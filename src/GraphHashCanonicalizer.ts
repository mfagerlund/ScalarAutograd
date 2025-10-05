import { Value } from './Value';

/**
 * HASH-BASED CANONICAL REPRESENTATION
 *
 * Fast canonicalization using dual 32-bit integer hashing.
 *
 * Design:
 * - Each node gets a hash based on its operation and children
 * - Position-dependent hashing (preserves order and multiplicity)
 * - Direct integer operations (NO STRING HASHING!)
 * - Parameters maintain topological order in signature
 *
 * Format: "0g,1g,2g,...|<hex_hash>"
 *
 * Benefits:
 * - O(n) complexity (no sorting!)
 * - Dual 32-bit hash (64-bit space, no BigInt overhead)
 * - Preserves multiplicity: add(a,a,b) != add(a,b)
 * - Recursion-free (stack-based iteration)
 * - Pure integer operations (no string conversion!)
 *
 * Note: add(a,b) != add(b,a) - not order-independent
 *
 * @internal
 */

export interface HashCanonicalResult {
  /** Canonical string with parameter list and expression hash */
  canon: string;
  /** Maps Value nodes to their parameter indices */
  paramMap: Map<Value, number>;
  /** Debug: full string expression (if enabled) */
  debugExpr?: string;
}

/**
 * Dual 32-bit hash for 64-bit hash space without BigInt overhead
 */
interface Hash64 {
  h1: number;  // First 32 bits
  h2: number;  // Second 32 bits
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

// Operation hash constants (pre-computed for speed)
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
  'sum': { h1: 0x9e3779b9, h2: 0x85ebca6b }, // Alias for '+'
  'eigenvalue_custom': { h1: 0x4f3779b9, h2: 0xc5ebca6b },
};

/**
 * Create hash from leaf ID and gradient flag - PURE INTEGER OPS
 */
function hashLeaf(id: number, hasGrad: boolean): Hash64 {
  const gradMask = hasGrad ? 0x12345678 : 0;
  return {
    h1: (id * PRIMES_32[0] ^ gradMask) >>> 0,
    h2: (id * PRIMES_32[1] ^ (gradMask << 8)) >>> 0
  };
}

/**
 * Get operation hash - direct lookup or fallback hash
 */
function getOpHash(op: string): Hash64 {
  if (OP_HASHES[op]) {
    return OP_HASHES[op];
  }

  // Fallback for unknown ops - hash the string characters as integers
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

/**
 * Combine two dual hashes with position information - PURE INTEGER OPS
 */
function combineHashes(h1: Hash64, h2: Hash64, position: number = 0): Hash64 {
  const p1 = PRIMES_32[position % PRIMES_32.length];
  const p2 = PRIMES_32[(position + 1) % PRIMES_32.length];

  // Mix each 32-bit component separately with different primes
  return {
    h1: (h1.h1 ^ Math.imul(h2.h1, p1)) >>> 0,
    h2: (h1.h2 ^ Math.imul(h2.h2, p2)) >>> 0
  };
}

/**
 * Canonicalize a computation graph using hash-based signatures.
 *
 * Much faster than string-based canonicalization - O(n) with no sorting.
 *
 * @param output - The output Value of the computation graph
 * @param params - Array of parameter Values for this residual/objective
 * @param debug - If true, also generate string expression for debugging
 * @returns Canonical string with hash and parameter mapping
 *
 * @internal
 */
export function canonicalizeGraphHash(
  output: Value,
  params: Value[],
  debug: boolean = false
): HashCanonicalResult {
  const t0 = performance.now();
  const leafToId = new Map<Value, number>();
  const visited = new Set<Value>();
  const allLeaves = new Set<Value>();

  // Phase 1: Discover all leaves in the graph
  function discoverLeaves(node: Value) {
    if (visited.has(node)) return;
    visited.add(node);

    const prev = (node as any).prev as Value[];
    if (prev.length === 0) {
      allLeaves.add(node);
    } else {
      for (const child of prev) {
        discoverLeaves(child);
      }
    }
  }

  discoverLeaves(output);
  const t1 = performance.now();

  // Phase 2: Assign IDs in stable, deterministic order
  let nextId = 0;
  const paramSet = new Set(params);

  for (const param of params) {
    if (allLeaves.has(param)) {
      leafToId.set(param, nextId++);
    }
  }

  const remainingLeaves = Array.from(allLeaves)
    .filter(leaf => !paramSet.has(leaf));

  for (const leaf of remainingLeaves) {
    leafToId.set(leaf, nextId++);
  }
  const t2 = performance.now();

  // Phase 3: Compute hashes recursively with memoization
  const memoized = new Map<Value, Hash64>();
  const debugExprs = debug ? new Map<Value, string>() : null;

  function computeHash(node: Value): Hash64 {
    // Return cached if already computed
    if (memoized.has(node)) {
      return memoized.get(node)!;
    }

    // Leaf node
    if (leafToId.has(node)) {
      const id = leafToId.get(node)!;
      const result = hashLeaf(id, node.requiresGrad);
      memoized.set(node, result);
      if (debugExprs) {
        const gradFlag = node.requiresGrad ? 'g' : '';
        debugExprs.set(node, `${id}${gradFlag}`);
      }
      return result;
    }

    const prev = (node as any).prev as Value[];
    if (prev.length === 0) {
      // Empty prev (shouldn't happen but handle it)
      const result = hashLeaf(0, false);
      memoized.set(node, result);
      return result;
    }

    let op = node._op || 'unknown';

    // Normalize pow(x, const_2) -> square(x)
    if (op === 'powValue' && prev.length === 2) {
      const exponent = prev[1];
      if (leafToId.has(exponent) && exponent.data === 2 && !exponent.requiresGrad) {
        op = 'square';
        const childHash = computeHash(prev[0]);  // Recursive
        const opHash = getOpHash(op);
        const result = combineHashes(opHash, childHash, 0);
        memoized.set(node, result);
        if (debugExprs) {
          debugExprs.set(node, `(${op},${debugExprs.get(prev[0])})`);
        }
        return result;
      }
    }

    // Convert 'sum' to '+'
    if (op === 'sum') {
      op = '+';
    }

    // Recursively hash children
    const childHashes: Hash64[] = prev.map(c => computeHash(c));

    // Combine hashes
    const opHash = getOpHash(op);
    let combinedHash = opHash;
    for (let i = 0; i < childHashes.length; i++) {
      combinedHash = combineHashes(combinedHash, childHashes[i], i);
    }

    memoized.set(node, combinedHash);

    if (debugExprs) {
      const childDebugStrs = prev.map(c => debugExprs.get(c)!);
      debugExprs.set(node, `(${op},${childDebugStrs.join(',')})`);
    }

    return combinedHash;
  }

  computeHash(output);  // Compute hash for output node
  const t3 = performance.now();

  // Phase 4: Build param list and get output hash
  const paramList: string[] = [];
  for (let i = 0; i < params.length; i++) {
    const param = params[i];
    if (!allLeaves.has(param)) continue;
    const gradFlag = param.requiresGrad ? 'g' : '';
    paramList.push(`${i}${gradFlag}`);
  }

  const exprHash = memoized.get(output)!;

  // Convert dual hash to hex string (16 characters for 64-bit hash)
  const hashHex = exprHash.h1.toString(16).padStart(8, '0') + exprHash.h2.toString(16).padStart(8, '0');
  const canon = `${paramList.join(',')}|${hashHex}`;

  // Build reverse map
  const paramMap = new Map<Value, number>();
  for (const [leaf, id] of leafToId) {
    paramMap.set(leaf, id);
  }

  const t4 = performance.now();
  const total = t4 - t0;

  const result: HashCanonicalResult = { canon, paramMap };

  if (debug && debugExprs) {
    result.debugExpr = debugExprs.get(output);
  }

  if (total > 10) {
    console.log(`[GraphHashCanonicalizer] Total: ${total.toFixed(0)}ms | Phase1(discover): ${(t1-t0).toFixed(0)}ms | Phase2(assign): ${(t2-t1).toFixed(0)}ms | Phase3(hash): ${(t3-t2).toFixed(0)}ms | Phase4(build): ${(t4-t3).toFixed(0)}ms | Nodes: ${visited.size} | keyLength: ${canon.length}`);
  }

  return result;
}
