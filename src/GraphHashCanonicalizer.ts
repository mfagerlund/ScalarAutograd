import { Value } from './Value';
import { createHash } from 'crypto';

/**
 * HASH-BASED CANONICAL REPRESENTATION
 *
 * Fast canonicalization using dual 32-bit hashing.
 *
 * Design:
 * - Each node gets a hash based on its operation and children
 * - Position-dependent hashing (preserves order and multiplicity)
 * - Commutative ops (add, mul, min, max): flatten nested same-ops
 * - Non-commutative ops (div, sub, pow): position-dependent hashing
 * - Parameters maintain topological order in signature
 *
 * Format: "0g,1g,2g,...|<hex_hash>"
 *
 * Benefits:
 * - O(n) complexity (no sorting!)
 * - Dual 32-bit hash (64-bit space, no BigInt overhead)
 * - Preserves multiplicity: add(a,a,b) != add(a,b)
 * - Recursion-free (stack-based iteration)
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

/**
 * Fast 32-bit hash using FNV-1a
 */
function hash32a(str: string): number {
  let hash = 2166136261; // FNV-1a offset basis (32-bit)
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 16777619); // FNV prime (32-bit)
  }
  return hash >>> 0; // Ensure unsigned
}

/**
 * Fast 32-bit hash using DJB2
 */
function hash32b(str: string): number {
  let hash = 5381; // DJB2 seed
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash) ^ str.charCodeAt(i); // hash * 33 ^ char
  }
  return hash >>> 0; // Ensure unsigned
}

/**
 * Create dual hash from string
 */
function hash64(str: string): Hash64 {
  return {
    h1: hash32a(str),
    h2: hash32b(str)
  };
}

// 32-bit primes for position mixing
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

/**
 * Combine two dual hashes with position information
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

// Removed xorHashes - now using position encoding for all ops to preserve multiplicity

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

  // Phase 3: Build hash-based canonical expression (iterative, no recursion)
  const memoized = new Map<Value, Hash64>();
  const debugExprs = debug ? new Map<Value, string>() : null;
  const opHashCache = new Map<string, Hash64>(); // Cache operation hashes

  // Topological sort to process nodes bottom-up
  const topoOrder: Value[] = [];
  const topoVisited = new Set<Value>();
  const topoStack: Value[] = [output];

  while (topoStack.length > 0) {
    const node = topoStack[topoStack.length - 1];

    if (topoVisited.has(node)) {
      topoStack.pop();
      continue;
    }

    const prev = (node as any).prev as Value[];

    // If leaf, mark visited and add to order
    if (prev.length === 0 || leafToId.has(node)) {
      topoVisited.add(node);
      topoOrder.push(node);
      topoStack.pop();
      continue;
    }

    // Check if all children processed
    let allChildrenProcessed = true;
    for (let i = prev.length - 1; i >= 0; i--) {
      if (!topoVisited.has(prev[i])) {
        allChildrenProcessed = false;
        topoStack.push(prev[i]);
      }
    }

    if (allChildrenProcessed) {
      topoVisited.add(node);
      topoOrder.push(node);
      topoStack.pop();
    }
  }

  // Process nodes in topological order (leaves first)
  for (const node of topoOrder) {
    // Skip if already memoized
    if (memoized.has(node)) continue;

    // Leaf node
    if (leafToId.has(node)) {
      const id = leafToId.get(node)!;
      const gradFlag = node.requiresGrad ? 'g' : '';
      const hashStr = `${id}${gradFlag}`;
      const result = hash64(hashStr);
      memoized.set(node, result);

      if (debugExprs) {
        debugExprs.set(node, hashStr);
      }
      continue;
    }

    const prev = (node as any).prev as Value[];
    if (prev.length === 0) continue; // Skip empty prev

    let op = node._op || 'unknown';

    // Normalize pow(x, const_2) -> square(x)
    if (op === 'powValue' && prev.length === 2) {
      const exponent = prev[1];
      if (leafToId.has(exponent) && exponent.data === 2 && !exponent.requiresGrad) {
        op = 'square';
        const childHash = memoized.get(prev[0])!;
        let opHash = opHashCache.get(op);
        if (!opHash) {
          opHash = hash64(op);
          opHashCache.set(op, opHash);
        }
        const result = combineHashes(opHash, childHash, 0);
        memoized.set(node, result);

        if (debugExprs) {
          debugExprs.set(node, `(${op},${debugExprs.get(prev[0])})`);
        }
        continue;
      }
    }

    // Convert 'sum' to '+'
    if (op === 'sum') {
      op = '+';
    }

    // All operations: position matters for each child (no flattening)
    const childHashes: Hash64[] = [];
    const childDebugStrs: string[] = debugExprs ? [] : [];

    for (let i = 0; i < prev.length; i++) {
      const childHash = memoized.get(prev[i])!;
      // Mix position into THIS level's child
      const positionedHash = combineHashes(childHash, { h1: i, h2: i }, i);
      childHashes.push(positionedHash);

      if (debugExprs && childDebugStrs) {
        childDebugStrs.push(debugExprs.get(prev[i])!);
      }
    }

    // Combine all positioned child hashes
    let opHash = opHashCache.get(op);
    if (!opHash) {
      opHash = hash64(op);
      opHashCache.set(op, opHash);
    }
    let combinedHash = opHash;
    for (const h of childHashes) {
      combinedHash = combineHashes(combinedHash, h, 0);
    }

    const result = combineHashes(combinedHash, { h1: 0, h2: 0 }, 0);
    memoized.set(node, result);

    if (debugExprs && childDebugStrs) {
      debugExprs.set(node, `(${op},${childDebugStrs.join(',')})`);
    }
  }

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
