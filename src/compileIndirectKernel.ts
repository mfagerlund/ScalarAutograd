import { Value } from './Value';
import { ValueRegistry } from './ValueRegistry';

/**
 * Compiles a residual function with indirect value indexing for kernel reuse.
 *
 * Instead of hardcoding parameter positions, this generates code that:
 * 1. Accepts a global value array
 * 2. Accepts an indices array specifying which values this graph uses
 * 3. Looks up values via indices: allValues[indices[i]]
 *
 * This allows topologically identical graphs to share the same kernel.
 *
 * @param residual - Output Value of the residual computation
 * @param params - Parameter Values (for Jacobian computation)
 * @param registry - ValueRegistry tracking all unique values
 * @returns Compiled function: (allValues, indices, gradientIndices, gradient) => number
 * @internal
 */
export function compileIndirectKernel(
  residual: Value,
  params: Value[],
  registry: ValueRegistry
): (allValues: number[], indices: number[], gradientIndices: number[], gradient: number[]) => number {
  const visited = new Set<Value>();
  const topoOrder: Value[] = [];
  const forwardCode: string[] = [];
  const backwardCode: string[] = [];
  let varCounter = 0;
  const nodeToVar = new Map<Value, string>();
  const nodeToIndexVar = new Map<Value, string>(); // Maps Value -> index variable name

  // Build topo order and register only leaf nodes (inputs)
  function buildTopoOrder(node: Value) {
    if (visited.has(node)) return;
    visited.add(node);

    for (const child of (node as any).prev as Value[]) {
      buildTopoOrder(child);
    }

    topoOrder.push(node);

    // Only register leaf nodes (graph inputs: constants and parameters)
    // Intermediates (prev.length > 0) are computed within kernel, not in registry
    if ((node as any).prev.length === 0) {
      registry.register(node);
    }
  }

  buildTopoOrder(residual);

  // Build index mapping for graph inputs (leaf nodes in topoOrder)
  const graphInputs: Value[] = [];
  for (const node of topoOrder) {
    const prev = (node as any).prev as Value[];
    if (prev.length === 0 && !graphInputs.includes(node)) {
      graphInputs.push(node);
    }
  }

  // Assign index variables to inputs
  graphInputs.forEach((input, i) => {
    nodeToIndexVar.set(input, `idx_${i}`);
  });

  function getVarName(node: Value): string {
    if (!nodeToVar.has(node)) {
      nodeToVar.set(node, `_v${varCounter++}`);
    }
    return nodeToVar.get(node)!;
  }

  // Generate forward code for each node in topo order
  for (const node of topoOrder) {
    const prev = (node as any).prev as Value[];

    if (prev.length === 0) {
      // Leaf node - load from allValues via index
      const indexVar = nodeToIndexVar.get(node)!;
      forwardCode.push(`const ${getVarName(node)} = allValues[${indexVar}];`);
    } else {
      // Computed node
      const childCodes = prev.map(c => getVarName(c));
      const code = node.getForwardCode(childCodes);
      forwardCode.push(`const ${getVarName(node)} = ${code};`);
    }
  }

  // Generate gradient declarations
  const gradDeclarations = Array.from(nodeToVar.entries())
    .map(([node, varName]) => `let grad_${varName} = 0;`);

  // Generate backward pass - iterate REVERSE topological order (same as Value.backward)
  for (let i = topoOrder.length - 1; i >= 0; i--) {
    const node = topoOrder[i];
    const prev = (node as any).prev as Value[];
    if (prev.length === 0 || !node.requiresGrad) continue;

    const gradVar = `grad_${getVarName(node)}`;
    const childGrads = prev.map((c: Value) => `grad_${getVarName(c)}`);
    const childVars = prev.map((c: Value) => getVarName(c));

    backwardCode.push(node.getBackwardCode(gradVar, childGrads, childVars));
  }

  const outputVar = getVarName(residual);

  // Extract indices for graph inputs
  const indexExtractions = graphInputs
    .map((input, i) => `const ${nodeToIndexVar.get(input)} = indices[${i}];`)
    .join('\n    ');

  // Build gradient updates - only for inputs that require gradients
  // Skip constants and any values with requiresGrad=false
  const gradientUpdates = graphInputs
    .map((input, inputIdx) => {
      if (!input.requiresGrad) {
        return null; // Skip - this input doesn't need gradients
      }
      const gradVar = `grad_${getVarName(input)}`;
      return `gradient[gradientIndices[${inputIdx}]] += ${gradVar};`;
    })
    .filter((line): line is string => line !== null)
    .join('\n    ');

  const functionBody = `
    ${indexExtractions}
    ${forwardCode.join('\n    ')}
    ${gradDeclarations.join('\n    ')}
    grad_${outputVar} = 1;
    ${backwardCode.join('\n    ')}
    ${gradientUpdates}
    return ${outputVar};
  `;

  try {
    return new Function('allValues', 'indices', 'gradientIndices', 'gradient', functionBody) as (
      allValues: number[],
      indices: number[],
      gradientIndices: number[],
      gradient: number[]
    ) => number;
  } catch (error) {
    console.error(`[compileIndirectKernel] COMPILATION ERROR:`, error);
    console.error(`[compileIndirectKernel] Function body:\n${functionBody}`);
    throw error;
  }
}

/**
 * Extract input indices for a residual graph.
 * Returns array of registry IDs for all leaf nodes in topological order.
 */
export function extractInputIndices(residual: Value, registry: ValueRegistry): number[] {
  const visited = new Set<Value>();
  const inputs: Value[] = [];

  function collectInputs(node: Value) {
    if (visited.has(node)) return;
    visited.add(node);

    const prev = (node as any).prev as Value[];
    if (prev.length === 0) {
      if (!inputs.includes(node)) {
        inputs.push(node);
      }
    } else {
      for (const child of prev) {
        collectInputs(child);
      }
    }
  }

  collectInputs(residual);

  return inputs.map(v => registry.getId(v));
}
