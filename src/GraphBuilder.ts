import { Value } from './Value';

/**
 * INCREMENTAL GRAPH BUILDER WITH HASH-AS-YOU-BUILD
 *
 * Eliminates the need for separate graph traversal by tracking:
 * 1. Incremental hash computation during graph construction
 * 2. Leaf Value objects (parameters + constants) as they're encountered
 *
 * Performance target: <1ms for typical residual graphs
 *
 * @internal
 */

/**
 * Dual 32-bit hash for 64-bit hash space without BigInt overhead
 */
interface Hash64 {
  h1: number;
  h2: number;
}

/**
 * Graph signature with hash and input mapping
 */
export interface GraphSignature {
  /** Hash string for kernel matching */
  hash: string;
  /** Ordered list of leaf Value objects (params first, then constants) */
  leaves: Value[];
  /** Map from Value to index in leaves array */
  leafIndexMap: Map<Value, number>;
}

// 32-bit primes for mixing
const PRIMES_32 = [
  2654435761, // Golden ratio prime
  805459861,
  3266489917,
  4101842887,
  2484345967,
  3369493747,
  1505335857,
  1267890027,
];

// Operation hash constants (same as GraphHashCanonicalizer)
const OP_HASHES: { [op: string]: Hash64 } = {
  '+': { h1: 0x9e3779b9, h2: 0x85ebca6b },
  '-': { h1: 0xc2b2ae35, h2: 0x27d4eb2f },
  '*': { h1: 0x165667b1, h2: 0xd3a2646c },
  '/': { h1: 0xfd7046c5, h2: 0xb55a4f09 },
  'pow': { h1: 0x5f356495, h2: 0x3c6ef372 },
  'powValue': { h1: 0x5f356495, h2: 0x3c6ef372 },
  'sqrt': { h1: 0x8f1bbcdc, h2: 0x42a5c977 },
  'exp': { h1: 0x1e3779b9, h2: 0x95ebca6b },
  'log': { h1: 0x2e3779b9, h2: 0xa5ebca6b },
  'abs': { h1: 0x3e3779b9, h2: 0xb5ebca6b },
  'square': { h1: 0x4e3779b9, h2: 0xc5ebca6b },
  'sin': { h1: 0x6e3779b9, h2: 0xe5ebca6b },
  'cos': { h1: 0x7e3779b9, h2: 0xf5ebca6b },
  'tan': { h1: 0x8e3779b9, h2: 0x15ebca6b },
  'asin': { h1: 0x9e3779b9, h2: 0x25ebca6b },
  'acos': { h1: 0xae3779b9, h2: 0x35ebca6b },
  'atan': { h1: 0xbe3779b9, h2: 0x45ebca6b },
  'atan2': { h1: 0xce3779b9, h2: 0x55ebca6b },
  'min': { h1: 0xde3779b9, h2: 0x65ebca6b },
  'max': { h1: 0xee3779b9, h2: 0x75ebca6b },
  'sigmoid': { h1: 0xfe3779b9, h2: 0x85ebca6b },
  'tanh': { h1: 0x1f3779b9, h2: 0x95ebca6b },
  'relu': { h1: 0x2f3779b9, h2: 0xa5ebca6b },
  'softplus': { h1: 0x3f3779b9, h2: 0xb5ebca6b },
  'sum': { h1: 0x9e3779b9, h2: 0x85ebca6b },
  'eigenvalue_custom': { h1: 0x4f3779b9, h2: 0xc5ebca6b },
};

function getOpHash(op: string): Hash64 {
  if (OP_HASHES[op]) {
    return OP_HASHES[op];
  }

  // Fallback for unknown ops
  let h1 = 2166136261;
  let h2 = 5381;
  for (let i = 0; i < op.length; i++) {
    const c = op.charCodeAt(i);
    h1 = (h1 ^ c) >>> 0;
    h1 = Math.imul(h1, 16777619);
    h2 = ((h2 << 5) + h2 + c) >>> 0;
  }
  return { h1: h1 >>> 0, h2: h2 >>> 0 };
}

function combineHashes(h1: Hash64, h2: Hash64, position: number = 0): Hash64 {
  const p1 = PRIMES_32[position % PRIMES_32.length];
  const p2 = PRIMES_32[(position + 1) % PRIMES_32.length];

  return {
    h1: (h1.h1 ^ Math.imul(h2.h1, p1)) >>> 0,
    h2: (h1.h2 ^ Math.imul(h2.h2, p2)) >>> 0
  };
}

function hashLeaf(leafIndex: number, hasGrad: boolean): Hash64 {
  const gradMask = hasGrad ? 0x12345678 : 0;
  return {
    h1: (leafIndex * PRIMES_32[0] ^ gradMask) >>> 0,
    h2: (leafIndex * PRIMES_32[1] ^ (gradMask << 8)) >>> 0
  };
}

/**
 * Incremental graph builder that tracks structure during construction.
 *
 * Usage:
 * ```typescript
 * const builder = new GraphBuilder(knownParams);
 * const result = builder.build(() => {
 *   const a = V.add(params[0], params[1]);
 *   const b = V.mul(a, V.C(2));
 *   return b;
 * });
 * // result.signature.hash - for kernel lookup
 * // result.signature.leaves - for index mapping at execution time
 * ```
 */
export class GraphBuilder {
  private leaves: Value[] = [];
  private leafIndexMap = new Map<Value, number>();
  private knownParams: Set<Value>;
  private operations: Value[] = [];

  /**
   * Create a new graph builder.
   * @param knownParams - Parameters that are known upfront (will be indexed first)
   */
  constructor(knownParams: Value[] = []) {
    this.knownParams = new Set(knownParams);

    // Pre-register known parameters
    for (const param of knownParams) {
      this.registerLeaf(param);
    }
  }

  /**
   * Register a leaf Value and return its index.
   * If already registered, returns existing index.
   * New parameters are added dynamically.
   */
  private registerLeaf(value: Value): number {
    let index = this.leafIndexMap.get(value);
    if (index !== undefined) {
      return index;
    }

    // New leaf - add to end
    index = this.leaves.length;
    this.leaves.push(value);
    this.leafIndexMap.set(value, index);
    return index;
  }

  /**
   * Record an operation during graph building.
   * Called automatically by Value.make() when in tracked context.
   * @internal
   */
  recordOp(value: Value): void {
    const prev = (value as any).prev as Value[];

    // If this is a leaf, register it
    if (prev.length === 0) {
      this.registerLeaf(value);
      return;
    }

    // For intermediate nodes, register all leaf children
    for (const child of prev) {
      if (child.prev.length === 0) {
        this.registerLeaf(child);
      }
    }

    // Record this operation for later hash computation
    this.operations.push(value);
  }

  /**
   * Build a graph in tracked context and return output + signature.
   *
   * @param fn - Function that builds and returns the output Value
   * @returns Output value and graph signature
   */
  build(fn: () => Value): { output: Value; signature: GraphSignature } {
    const prevBuilder = Value.currentBuilder;
    Value.currentBuilder = this;

    try {
      const output = fn();

      // Finalize signature
      // Stable sort: params (with grad) first, then params (no grad), then constants
      const sortedLeaves = [...this.leaves].sort((a, b) => {
        const aIsParam = this.knownParams.has(a);
        const bIsParam = this.knownParams.has(b);

        if (aIsParam !== bIsParam) return aIsParam ? -1 : 1;
        if (a.requiresGrad !== b.requiresGrad) return b.requiresGrad ? 1 : -1;
        return a._id - b._id;
      });

      // Rebuild index map with sorted order
      const finalLeafIndexMap = new Map<Value, number>();
      for (let i = 0; i < sortedLeaves.length; i++) {
        finalLeafIndexMap.set(sortedLeaves[i], i);
      }

      // Compute hash based on graph structure (NOW that we know all leaf assignments)
      // Create stable node ID map: leaves get their indices, operations get sequential IDs
      const nodeIdMap = new Map<Value, number>();
      for (let i = 0; i < sortedLeaves.length; i++) {
        nodeIdMap.set(sortedLeaves[i], i);
      }

      let nextOpId = sortedLeaves.length;
      let hash: Hash64 = { h1: 2166136261, h2: 5381 };

      // Hash leaf configuration first
      for (let i = 0; i < sortedLeaves.length; i++) {
        const leafHash = hashLeaf(i, sortedLeaves[i].requiresGrad);
        hash = combineHashes(hash, leafHash, i);
      }

      // Hash operations in order they were created
      for (const op of this.operations) {
        const prev = (op as any).prev as Value[];

        // Assign ID to this operation
        if (!nodeIdMap.has(op)) {
          nodeIdMap.set(op, nextOpId++);
        }

        let opName = op._op || 'unknown';

        // Normalize operations
        if (opName === 'sum') opName = '+';
        if (opName === 'powValue' && prev.length === 2) {
          const exponent = prev[1];
          if (exponent.prev.length === 0 && exponent.data === 2 && !exponent.requiresGrad) {
            opName = 'square';
          }
        }

        const opHash = getOpHash(opName);
        hash = combineHashes(hash, opHash, 0);

        // Hash children by their stable IDs
        for (let i = 0; i < prev.length; i++) {
          const child = prev[i];
          if (!nodeIdMap.has(child)) {
            nodeIdMap.set(child, nextOpId++);
          }
          const childId = nodeIdMap.get(child)!;
          const childHash: Hash64 = {
            h1: childId * PRIMES_32[2],
            h2: childId * PRIMES_32[3]
          };
          hash = combineHashes(hash, childHash, i + 1);
        }
      }

      const hashString = hash.h1.toString(16).padStart(8, '0') +
                         hash.h2.toString(16).padStart(8, '0');

      return {
        output,
        signature: {
          hash: hashString,
          leaves: sortedLeaves,
          leafIndexMap: finalLeafIndexMap
        }
      };
    } finally {
      Value.currentBuilder = prevBuilder;
    }
  }

  /**
   * Get current signature without finalizing (for debugging)
   */
  getCurrentSignature(): Partial<GraphSignature> {
    return {
      hash: 'incomplete',
      leaves: [...this.leaves]
    };
  }
}
