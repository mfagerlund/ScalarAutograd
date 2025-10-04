import { Value } from './Value';

/**
 * CANONICAL STRING REPRESENTATION
 *
 * Design goals:
 * 1. Same topology + same grad requirements = same kernel
 * 2. Commutative ops automatically match regardless of order
 * 3. Flatten nested commutative ops for maximal reuse
 * 4. Encode which parameters need gradients
 *
 * Format: "param_ids|expression"
 *
 * Examples:
 *   "0g,1g,2|(+,(*,0g,1g),2)"
 *    ^^^^^^  ^^^^^^^^^^^^^^^^
 *    params   expression
 *
 * - 0g,1g: parameters that need gradients
 * - 2: constant or non-grad parameter
 * - Expression uses Lisp-style: (op,arg1,arg2,...)
 * - Commutative ops sort children by operator name, then leaves by ID
 *
 * @internal
 */

export interface CanonicalResult {
  /** Canonical string uniquely identifying graph structure */
  canon: string;
  /** Maps Value nodes to their parameter indices */
  paramMap: Map<Value, number>;
}

/**
 * Split a comma-separated expression, respecting nested parentheses.
 * Example: "0g,(*,1,2g),3" -> ["0g", "(*,1,2g)", "3"]
 * @internal
 */
function splitExpr(expr: string): string[] {
  const parts: string[] = [];
  let current = '';
  let depth = 0;

  for (const char of expr) {
    if (char === '(') {
      depth++;
      current += char;
    } else if (char === ')') {
      depth--;
      current += char;
    } else if (char === ',' && depth === 0) {
      if (current) parts.push(current);
      current = '';
    } else {
      current += char;
    }
  }

  if (current) parts.push(current);
  return parts;
}

/**
 * Sort expressions canonically:
 * 1. Leaves (params/constants) before operations
 * 2. Within leaves: by ID (0g, 1g, 2, 3g, ...)
 * 3. Within operations: by operator name, then recursively by children
 * @internal
 */
function canonicalSort(exprs: string[]): string[] {
  return exprs.sort((a, b) => {
    const aIsLeaf = !a.startsWith('(');
    const bIsLeaf = !b.startsWith('(');

    // Leaves before operations
    if (aIsLeaf && !bIsLeaf) return -1;
    if (!aIsLeaf && bIsLeaf) return 1;

    // Both leaves: numeric comparison of IDs
    if (aIsLeaf && bIsLeaf) {
      const aId = parseInt(a);
      const bId = parseInt(b);
      return aId - bId;
    }

    // Both operations: extract operator names
    const aOp = a.slice(1, a.indexOf(','));
    const bOp = b.slice(1, b.indexOf(','));

    if (aOp !== bOp) {
      return aOp.localeCompare(bOp);
    }

    // Same operator: compare full expression
    return a.localeCompare(b);
  });
}

/**
 * Canonicalize a computation graph to produce a unique string signature.
 *
 * Graphs with identical topology and gradient requirements will produce
 * the same canonical string, enabling kernel reuse.
 *
 * Features:
 * - Flattens nested commutative operations (+, *, min, max)
 * - Sorts children of commutative ops for order independence
 * - Encodes gradient requirements in parameter list
 * - Normalizes common patterns (e.g., pow(x,2) -> square(x))
 *
 * @param output - The output Value of the computation graph
 * @param params - Array of parameter Values for this residual/objective
 * @returns Canonical string and parameter mapping
 *
 * @example
 * ```typescript
 * const x = V.W(1), y = V.W(2);
 * const r1 = V.add(V.square(x), V.square(y));
 * const r2 = V.add(V.square(y), V.square(x)); // Reordered
 *
 * const { canon: c1 } = canonicalizeGraph(r1, [x, y]);
 * const { canon: c2 } = canonicalizeGraph(r2, [x, y]);
 * // c1 === c2 (same kernel can be reused!)
 * ```
 *
 * @internal
 */
export function canonicalizeGraph(output: Value, params: Value[]): CanonicalResult {
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
  // Primary params (from argument) get 0..n-1
  // Remaining leaves (constants, etc.) get n..m
  let nextId = 0;
  const paramSet = new Set(params);

  for (const param of params) {
    if (allLeaves.has(param)) {
      leafToId.set(param, nextId++);
    }
  }

  // Remaining leaves (constants) are assigned IDs in order of discovery
  // We don't sort by value - constants with different values should map to
  // the same structural position for kernel reuse
  const remainingLeaves = Array.from(allLeaves)
    .filter(leaf => !paramSet.has(leaf));

  for (const leaf of remainingLeaves) {
    leafToId.set(leaf, nextId++);
  }
  const t2 = performance.now();

  // Phase 3: Build canonical expression
  let sortCalls = 0;
  let sortTimeMs = 0;
  const memoized = new Map<Value, string>();

  function canonExpr(node: Value): string {
    // Check memoization first - nodes can appear multiple times in the graph
    if (memoized.has(node)) {
      return memoized.get(node)!;
    }

    if (leafToId.has(node)) {
      const id = leafToId.get(node)!;
      const gradFlag = node.requiresGrad ? 'g' : '';
      const result = `${id}${gradFlag}`;
      memoized.set(node, result);
      return result;
    }

    const prev = (node as any).prev as Value[];
    let op = node._op || 'unknown';

    // Normalize pow(x, const_2) -> square(x)
    if (op === 'powValue' && prev.length === 2) {
      const exponent = prev[1];
      if (leafToId.has(exponent) && exponent.data === 2 && !exponent.requiresGrad) {
        op = 'square';
        const result = `(${op},${canonExpr(prev[0])})`;
        memoized.set(node, result);
        return result;
      }
    }

    // Convert 'sum' to '+'
    if (op === 'sum') {
      op = '+';
    }

    const isCommutative = op === '+' || op === '*' || op === 'min' || op === 'max';

    if (isCommutative) {
      // Collect all child expressions, flattening same-op children
      const childStrs: string[] = [];

      function collectChildStrs(n: Value) {
        // Check if this child is the same commutative op - flatten it
        if (n._op === op) {
          // Same op: recursively collect its children's strings (skip this node)
          for (const child of (n as any).prev) {
            collectChildStrs(child);
          }
        } else {
          // Different op or leaf: get its canonical string
          childStrs.push(canonExpr(n));
        }
      }

      // Collect all child strings recursively
      for (const child of prev) {
        collectChildStrs(child);
      }

      // Fast sort using default string comparison
      // For large arrays, this is much faster than custom comparison functions
      if (childStrs.length > 1) {
        const sortStart = performance.now();
        childStrs.sort();
        sortTimeMs += performance.now() - sortStart;
        sortCalls++;
      }

      const result = `(${op},${childStrs.join(',')})`;
      memoized.set(node, result);
      return result;
    }

    // Non-commutative: preserve order
    const childStrs = prev.map(p => canonExpr(p));
    const result = `(${op},${childStrs.join(',')})`;
    memoized.set(node, result);
    return result;
  }

  // Phase 4: Build param list (ONLY primary params, not internal constants)
  const paramList: string[] = [];
  for (let i = 0; i < params.length; i++) {
    const param = params[i];
    if (!allLeaves.has(param)) continue; // Skip unused params
    const gradFlag = param.requiresGrad ? 'g' : '';
    paramList.push(`${i}${gradFlag}`);
  }

  const expr = canonExpr(output);
  const t3 = performance.now();

  const canon = `${paramList.join(',')}|${expr}`;

  // Build reverse map
  const paramMap = new Map<Value, number>();
  for (const [leaf, id] of leafToId) {
    paramMap.set(leaf, id);
  }

  const t4 = performance.now();
  const total = t4 - t0;
  if (total > 10) {
    console.log(`[GraphCanonicalizer] Total: ${total.toFixed(0)}ms | Phase1(discover): ${(t1-t0).toFixed(0)}ms | Phase2(assign): ${(t2-t1).toFixed(0)}ms | Phase3(expr): ${(t3-t2).toFixed(0)}ms [sorts: ${sortCalls} calls, ${sortTimeMs.toFixed(0)}ms] | Phase4(build): ${(t4-t3).toFixed(0)}ms | Nodes: ${visited.size}`);
  }

  return { canon, paramMap };
}
