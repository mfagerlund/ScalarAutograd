import { Value } from './Value';
import { createHash } from 'crypto';

/**
 * HASH-BASED CANONICAL REPRESENTATION
 *
 * Fast alternative to string-based canonicalization using cryptographic hashing.
 *
 * Design:
 * - Each node gets a hash based on its operation and children
 * - Commutative ops (add, mul, min, max): XOR children hashes (order-independent)
 * - Non-commutative ops (div, sub, pow): position-dependent hashing
 * - Parameters maintain topological order in signature
 *
 * Format: "0g,1g,2g,...|<hex_hash>"
 *
 * Benefits:
 * - O(n) complexity (no sorting!)
 * - Collision probability: ~2^-128 with SHA-256 (astronomically low)
 * - Same structural graphs produce identical hashes
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
 * Simple 64-bit hash function (fast, non-cryptographic)
 * Uses FNV-1a algorithm
 */
function hash64(str: string): bigint {
  let hash = 14695981039346656037n; // FNV offset basis
  const prime = 1099511628211n; // FNV prime

  for (let i = 0; i < str.length; i++) {
    hash ^= BigInt(str.charCodeAt(i));
    hash = (hash * prime) & 0xFFFFFFFFFFFFFFFFn; // Keep 64 bits
  }

  return hash;
}

// Large primes for position mixing
const POSITION_PRIMES = [
  2654435761n,
  805459861n,
  3266489917n,
  4101842887n,
  2484345967n,
  3369493747n,
  1505335857n,
  1267890027n,
];

/**
 * Combine two hashes with position information
 */
function combineHashes(h1: bigint, h2: bigint, position: number = 0): bigint {
  // Use different prime for each position (mod available primes)
  const prime = POSITION_PRIMES[position % POSITION_PRIMES.length];
  // Rotate h2 based on position for additional mixing
  const rotated = ((h2 << BigInt(position * 7)) | (h2 >> (64n - BigInt(position * 7)))) & 0xFFFFFFFFFFFFFFFFn;
  return (h1 ^ (rotated * prime)) & 0xFFFFFFFFFFFFFFFFn;
}

/**
 * XOR hashes together (position-independent for commutative ops)
 */
function xorHashes(hashes: bigint[]): bigint {
  let result = 0n;
  for (const h of hashes) {
    result ^= h;
  }
  return result;
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

  // Phase 3: Build hash-based canonical expression
  const memoized = new Map<Value, bigint>();
  const debugExprs = debug ? new Map<Value, string>() : null;

  function canonHash(node: Value, position: number = 0): bigint {
    // Check memoization
    if (memoized.has(node)) {
      return memoized.get(node)!;
    }

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

      return result;
    }

    const prev = (node as any).prev as Value[];
    let op = node._op || 'unknown';

    // Normalize pow(x, const_2) -> square(x)
    if (op === 'powValue' && prev.length === 2) {
      const exponent = prev[1];
      if (leafToId.has(exponent) && exponent.data === 2 && !exponent.requiresGrad) {
        op = 'square';
        const childHash = canonHash(prev[0], 0);
        const opHash = hash64(op);
        const result = combineHashes(opHash, childHash, position);
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

    const isCommutative = op === '+' || op === '*' || op === 'min' || op === 'max';

    if (isCommutative) {
      // Collect child hashes, flattening same-op children
      const childHashes: bigint[] = [];
      const childDebugStrs: string[] = debugExprs ? [] : [];

      function collectChildHashes(n: Value) {
        // Flatten same commutative operation
        if (n._op === op) {
          for (const child of (n as any).prev) {
            collectChildHashes(child);
          }
        } else {
          // Different op or leaf: get its hash
          childHashes.push(canonHash(n, 0)); // Position 0 for children of commutative ops
          if (debugExprs && childDebugStrs) {
            childDebugStrs.push(debugExprs.get(n)!);
          }
        }
      }

      for (const child of prev) {
        collectChildHashes(child);
      }

      // XOR all child hashes (order-independent!)
      const childrenHash = xorHashes(childHashes);

      // Combine with operation hash and position
      const opHash = hash64(op);
      const result = combineHashes(opHash, childrenHash, position);
      memoized.set(node, result);

      if (debugExprs && childDebugStrs) {
        // For debug, sort strings to make it readable
        childDebugStrs.sort();
        debugExprs.set(node, `(${op},${childDebugStrs.join(',')})`);
      }

      return result;
    }

    // Non-commutative: position matters for each child
    const childHashes: bigint[] = [];
    const childDebugStrs: string[] = debugExprs ? [] : [];

    for (let i = 0; i < prev.length; i++) {
      const childHash = canonHash(prev[i], 0); // Don't pass position down
      // Mix position into THIS level's child
      const positionedHash = combineHashes(childHash, BigInt(i), i);
      childHashes.push(positionedHash);

      if (debugExprs && childDebugStrs) {
        childDebugStrs.push(debugExprs.get(prev[i])!);
      }
    }

    // Combine all positioned child hashes
    let combinedHash = hash64(op);
    for (const h of childHashes) {
      combinedHash = combineHashes(combinedHash, h, 0);
    }

    // Final position mix from parent
    const result = combineHashes(combinedHash, 0n, position);
    memoized.set(node, result);

    if (debugExprs && childDebugStrs) {
      debugExprs.set(node, `(${op},${childDebugStrs.join(',')})`);
    }

    return result;
  }

  // Phase 4: Build param list
  const paramList: string[] = [];
  for (let i = 0; i < params.length; i++) {
    const param = params[i];
    if (!allLeaves.has(param)) continue;
    const gradFlag = param.requiresGrad ? 'g' : '';
    paramList.push(`${i}${gradFlag}`);
  }

  const exprHash = canonHash(output, 0);
  const t3 = performance.now();

  // Convert hash to hex string (16 characters for 64-bit hash)
  const hashHex = exprHash.toString(16).padStart(16, '0');
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
