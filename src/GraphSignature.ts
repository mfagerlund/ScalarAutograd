import { Value } from './Value';

/**
 * Canonical representation of a computation graph structure.
 * Two graphs with identical topology will have the same signature.
 * @internal
 */
export interface GraphSignature {
  /** Ordered list of operations in topological order */
  operations: string[];

  /** Parent node indices for each node: [[parent1_idx, parent2_idx], ...] */
  topology: number[][];

  /** Hash for fast lookup */
  hash: string;
}

/**
 * Canonicalize a computation graph to identify its structure.
 *
 * Graphs with identical topology (same operations in same order with same connections)
 * will produce the same signature, regardless of:
 * - Value data (3 vs 5)
 * - Labels/names
 * - Specific Value object instances
 *
 * @param output - Output Value of the graph
 * @returns Canonical signature identifying the graph structure
 * @internal
 */
export function canonicalizeGraph(output: Value): GraphSignature {
  const visited = new Set<Value>();
  const topoOrder: Value[] = [];

  // Build topological order
  function traverse(node: Value) {
    if (visited.has(node)) return;
    visited.add(node);

    const prev = (node as any).prev as Value[];
    for (const child of prev) {
      traverse(child);
    }

    topoOrder.push(node);
  }

  traverse(output);

  // Build canonical representation
  const nodeToIndex = new Map<Value, number>();
  topoOrder.forEach((node, i) => nodeToIndex.set(node, i));

  const operations: string[] = [];
  const topology: number[][] = [];

  for (const node of topoOrder) {
    const prev = (node as any).prev as Value[];

    // Operation: use _op for computed nodes, 'leaf' for inputs
    const op = prev.length === 0 ? 'leaf' : (node._op || 'unknown');
    operations.push(op);

    // Topology: parent indices in topoOrder
    const parentIndices = prev.map(p => nodeToIndex.get(p)!);
    topology.push(parentIndices);
  }

  // Generate hash - fast custom implementation
  const hash = generateHash(operations, topology);

  return { operations, topology, hash };
}

/**
 * Fast hash generation for graph signatures.
 * Uses simple string concatenation - much faster than JSON.stringify.
 */
function generateHash(operations: string[], topology: number[][]): string {
  // Format: "op1,op2,op3|[0,1],[2,3]"
  const opsStr = operations.join(',');
  const topoStr = topology.map(arr => `[${arr.join(',')}]`).join(',');
  return `${opsStr}|${topoStr}`;
}
