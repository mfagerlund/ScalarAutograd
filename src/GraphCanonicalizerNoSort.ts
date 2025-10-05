import { Value } from './Value';

/**
 * CANONICAL STRING REPRESENTATION (NO SORTING)
 *
 * Fast variant that doesn't sort children of commutative operations.
 * This means fewer kernel matches (add(a,b) != add(b,a)), but much faster.
 *
 * Uses traverse counter optimization: instead of Map<Value, number> lookups,
 * stamps each Value with a traverse ID for O(1) property access.
 *
 * @internal
 */

// Global traverse counter for memoization
let globalTraverseId = 0;

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
  const allLeaves = new Set<Value>();

  // Increment global traverse ID for this canonicalization
  const traverseId = ++globalTraverseId;

  // Phase 1: Discover all leaves in the graph (iterative)
  const discoverStack: Value[] = [output];
  const discoverVisited = new Set<Value>();

  while (discoverStack.length > 0) {
    const node = discoverStack.pop()!;
    if (discoverVisited.has(node)) continue;
    discoverVisited.add(node);

    const prev = (node as any).prev as Value[];
    if (prev.length === 0) {
      allLeaves.add(node);
    } else {
      for (const child of prev) {
        discoverStack.push(child);
      }
    }
  }

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

  // Phase 3: Build canonical expression (NO SORTING, NO RECURSION!)
  let nextNodeId = nextId; // Start after leaf IDs

  // Build a compact representation: array of [nodeId, op, childIds...]
  const nodeExprs: Array<[number, string, number[]]> = [];

  // Topological sort to process nodes bottom-up (iterative)
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

  const tTraversal = performance.now();

  // Process nodes in topological order (leaves first)
  // Use traverse counter stamping instead of Map lookups
  for (const node of topoOrder) {
    const nodeAny = node as any;

    // Skip if already processed in this traverse
    if (nodeAny._canonTraverseId === traverseId) continue;

    // Leaf node
    const leafId = leafToId.get(node);
    if (leafId !== undefined) {
      nodeAny._canonTraverseId = traverseId;
      nodeAny._canonNodeId = leafId;
      continue;
    }

    const prev = (node as any).prev as Value[];
    if (prev.length === 0) continue; // Skip empty prev

    // Assign new ID and stamp it
    const myId = nextNodeId++;
    nodeAny._canonTraverseId = traverseId;
    nodeAny._canonNodeId = myId;

    let op = node._op || 'unknown';

    // Normalize pow(x, const_2) -> square(x)
    if (op === 'powValue' && prev.length === 2) {
      const exponent = prev[1];
      const exponentId = leafToId.get(exponent);
      if (exponentId !== undefined && exponent.data === 2 && !exponent.requiresGrad) {
        op = 'square';
        const childId = (prev[0] as any)._canonNodeId;
        nodeExprs.push([myId, op, [childId]]);
        continue;
      }
    }

    if (op === 'sum') op = '+';

    // All operations: just use children as-is (no flattening)
    const childIds = prev.map(p => (p as any)._canonNodeId);
    nodeExprs.push([myId, op, childIds]);
  }

  const outputId = (output as any)._canonNodeId;
  const tTraversalEnd = performance.now();

  // Build compact canonical signature from node expressions
  const tStringStart = performance.now();

  // Build leaf lookup for gradient flags (id -> 'g' suffix or '')
  const leafGradFlags = new Map<number, string>();
  for (const [leaf, id] of leafToId) {
    leafGradFlags.set(id, leaf.requiresGrad ? 'g' : '');
  }

  // Build expression parts (compact format)
  const exprParts: string[] = [];
  for (const [nodeId, op, childIds] of nodeExprs) {
    const childRefs = childIds.map(id => {
      const gradFlag = leafGradFlags.get(id);
      if (gradFlag !== undefined) {
        return `${id}${gradFlag}`;
      }
      return `n${id}`;
    }).join(',');
    exprParts.push(`n${nodeId}:(${op},${childRefs})`);
  }

  const expr = exprParts.join(';');

  // Phase 4: Build param list
  const paramList: string[] = [];
  for (let i = 0; i < params.length; i++) {
    const param = params[i];
    if (!allLeaves.has(param)) continue;
    const gradFlag = param.requiresGrad ? 'g' : '';
    paramList.push(`${i}${gradFlag}`);
  }

  // For single-node leaf graphs, include the leaf ID
  const finalExpr = expr.length > 0 ? expr : `${outputId}${leafGradFlags.get(outputId) || ''}`;
  const canon = `${paramList.join(',')}|${finalExpr}`;

  const tStringEnd = performance.now();
  const t3 = performance.now();

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
    console.log(`[GraphCanonicalizerNoSort] Total: ${total.toFixed(0)}ms | Phase1(discover): ${(t1-t0).toFixed(0)}ms | Phase2(assign): ${(t2-t1).toFixed(0)}ms | Phase3(expr): ${(t3-t2).toFixed(0)}ms [traversal: ${traversalTime.toFixed(0)}ms, stringBuild: ${stringBuildTime.toFixed(0)}ms] | Phase4(build): ${(t4-t3).toFixed(0)}ms | Nodes: ${discoverVisited.size} | NodeExprs: ${nodeExprs.length} | keyLength: ${canon.length}`);
  }

  return { canon, paramMap };
}
