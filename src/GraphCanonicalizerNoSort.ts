import { Value } from './Value';

/**
 * CANONICAL STRING REPRESENTATION (NO SORTING)
 *
 * Fast variant that doesn't sort children of commutative operations.
 * This means fewer kernel matches (add(a,b) != add(b,a)), but much faster.
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
 * Canonicalize a computation graph WITHOUT sorting (fast, fewer matches).
 *
 * This is faster than the sorted version but produces different signatures
 * for reordered commutative operations.
 *
 * @param output - The output Value of the computation graph
 * @param params - Array of parameter Values for this residual/objective
 * @returns Canonical string and parameter mapping
 *
 * @internal
 */
export function canonicalizeGraphNoSort(output: Value, params: Value[]): CanonicalResult {
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

  // Phase 3: Build canonical expression (NO SORTING!)
  // NEW APPROACH: Build string lazily only when needed for final output
  // During traversal, just assign node IDs
  const nodeToId = new Map<Value, number>();
  let nextNodeId = nextId; // Start after leaf IDs
  let getNodeIdCalls = 0;
  let cacheHits = 0;

  // Build a compact representation: array of [nodeId, op, childIds...]
  const nodeExprs: Array<[number, string, number[]]> = [];

  function getNodeId(node: Value): number {
    getNodeIdCalls++;
    // Check if it's a leaf
    const leafId = leafToId.get(node);
    if (leafId !== undefined) return leafId;

    // Check if already processed
    const existingId = nodeToId.get(node);
    if (existingId !== undefined) {
      cacheHits++;
      return existingId;
    }

    // Assign new ID
    const myId = nextNodeId++;
    nodeToId.set(node, myId);

    const prev = (node as any).prev as Value[];
    let op = node._op || 'unknown';

    // Normalize pow(x, const_2) -> square(x)
    if (op === 'powValue' && prev.length === 2) {
      const exponent = prev[1];
      const exponentId = leafToId.get(exponent);
      if (exponentId !== undefined && exponent.data === 2 && !exponent.requiresGrad) {
        op = 'square';
        const childId = getNodeId(prev[0]);
        nodeExprs.push([myId, op, [childId]]);
        return myId;
      }
    }

    if (op === 'sum') op = '+';

    const isCommutative = op === '+' || op === '*' || op === 'min' || op === 'max';

    if (isCommutative) {
      // Flatten same-op children
      const childIds: number[] = [];

      function collectChildIds(n: Value) {
        if (n._op === op) {
          const nPrev = (n as any).prev as Value[];
          for (let i = 0; i < nPrev.length; i++) {
            collectChildIds(nPrev[i]);
          }
        } else {
          childIds.push(getNodeId(n));
        }
      }

      for (let i = 0; i < prev.length; i++) {
        collectChildIds(prev[i]);
      }

      nodeExprs.push([myId, op, childIds]);
      return myId;
    }

    // Non-commutative
    const childIds = prev.map(p => getNodeId(p));
    nodeExprs.push([myId, op, childIds]);
    return myId;
  }

  const tTraversal = performance.now();
  const outputId = getNodeId(output);
  const tTraversalEnd = performance.now();

  // Now build the string ONCE from the compact representation
  const tStringStart = performance.now();

  // Build leaf lookup for gradient flags (id -> 'g' suffix or '')
  const leafGradFlags = new Map<number, string>();
  for (const [leaf, id] of leafToId) {
    leafGradFlags.set(id, leaf.requiresGrad ? 'g' : '');
  }

  // Build expression using node IDs (no string embedding!)
  // Format: n123:(op,child1,child2) where children are IDs with optional 'g' or 'n' prefix
  const exprParts: string[] = [];

  for (const [nodeId, op, childIds] of nodeExprs) {
    const childRefs = childIds.map(id => {
      const gradFlag = leafGradFlags.get(id);
      if (gradFlag !== undefined) {
        // It's a leaf
        return `${id}${gradFlag}`;
      }
      // It's a node - prefix with 'n'
      return `n${id}`;
    }).join(',');
    exprParts.push(`n${nodeId}:(${op},${childRefs})`);
  }

  let expr = exprParts.join(';');

  // Special case: if output is a leaf node itself, expr will be empty
  // In that case, represent it as just the leaf ID
  if (expr === '' && leafGradFlags.has(outputId)) {
    const gradFlag = leafGradFlags.get(outputId);
    expr = `${outputId}${gradFlag}`;
  }

  const tStringEnd = performance.now();
  const t3 = performance.now();

  // Phase 4: Build param list
  const paramList: string[] = [];
  for (let i = 0; i < params.length; i++) {
    const param = params[i];
    if (!allLeaves.has(param)) continue;
    const gradFlag = param.requiresGrad ? 'g' : '';
    paramList.push(`${i}${gradFlag}`);
  }

  const canon = `${paramList.join(',')}|${expr}`;

  // Build reverse map
  const paramMap = new Map<Value, number>();
  for (const [leaf, id] of leafToId) {
    paramMap.set(leaf, id);
  }

  const t4 = performance.now();
  const total = t4 - t0;
  if (total > 10) {
    const traversalTime = tTraversalEnd - tTraversal;
    const stringBuildTime = tStringEnd - tStringStart;
    const revisitRatio = getNodeIdCalls > 0 ? (cacheHits / getNodeIdCalls * 100).toFixed(1) : '0';
    console.log(`[GraphCanonicalizerNoSort] Total: ${total.toFixed(0)}ms | Phase1(discover): ${(t1-t0).toFixed(0)}ms | Phase2(assign): ${(t2-t1).toFixed(0)}ms | Phase3(expr): ${(t3-t2).toFixed(0)}ms [traversal: ${traversalTime.toFixed(0)}ms, stringBuild: ${stringBuildTime.toFixed(0)}ms] | Phase4(build): ${(t4-t3).toFixed(0)}ms | Nodes: ${visited.size} | NodeExprs: ${nodeExprs.length} | getNodeId calls: ${getNodeIdCalls}, cacheHits: ${cacheHits} (${revisitRatio}%), keyLength: ${canon.length}`);
  }

  return { canon, paramMap };
}
