import { Value } from './Value';

export function applyCompiledGradients(
  compiledFn: (...args: number[]) => number[],
  params: Value[]
): number[] {
  const grads = compiledFn(...params.map(p => p.data));
  params.forEach((p, i) => p.grad = grads[i]);
  return grads;
}

export function compileGradientFunction(output: Value, params: Value[]) {
  const paramNames = params.map(p => p.paramName!);

  const visited = new Set<Value>();
  const topoOrder: Value[] = [];
  const forwardCode: string[] = [];
  const backwardCode: string[] = [];
  let varCounter = 0;
  const nodeToVar = new Map<Value, string>();

  function getVarName(node: Value): string {
    if (node.paramName) return node.paramName;
    if (!nodeToVar.has(node)) {
      nodeToVar.set(node, `_v${varCounter++}`);
    }
    return nodeToVar.get(node)!;
  }

  function compileForward(node: Value) {
    if (visited.has(node)) return;
    visited.add(node);

    for (const child of (node as any).prev as Value[]) {
      compileForward(child);
    }

    topoOrder.push(node);

    if (!node.paramName) {
      if ((node as any).prev.length > 0) {
        const childCodes = ((node as any).prev as Value[]).map(c => getVarName(c));
        const code = node.getForwardCode(childCodes);
        forwardCode.push(`const ${getVarName(node)} = ${code};`);
      } else {
        forwardCode.push(`const ${getVarName(node)} = ${node.data};`);
      }
    }
  }

  compileForward(output);

  const gradDeclarations = Array.from(nodeToVar.entries())
    .map(([node, varName]) => `let grad_${varName} = 0;`)
    .concat(paramNames.map(p => `let grad_${p} = 0;`));

  for (let i = topoOrder.length - 1; i >= 0; i--) {
    const node = topoOrder[i];
    if ((node as any).prev.length === 0) continue;

    const gradVar = `grad_${getVarName(node)}`;
    const childGrads = ((node as any).prev as Value[]).map((c: Value) => `grad_${getVarName(c)}`);
    const childVars = ((node as any).prev as Value[]).map((c: Value) => getVarName(c));

    backwardCode.push(node.getBackwardCode(gradVar, childGrads, childVars));
  }

  const functionBody = `
    ${forwardCode.join('\n    ')}
    ${gradDeclarations.join('\n    ')}
    grad_${getVarName(output)} = 1;
    ${backwardCode.join('\n    ')}
    return [${paramNames.map(p => `grad_${p}`).join(', ')}];
  `;

  return new Function(...paramNames, functionBody) as (...args: number[]) => number[];
}

/**
 * Compiles a residual function for efficient Jacobian computation.
 *
 * Returns a compiled function that takes parameter values and the Jacobian matrix,
 * then computes the residual value and directly updates its row in the Jacobian.
 *
 * **IMPORTANT LIMITATION**: This only works when the residual computation graph
 * structure is FIXED. If your residual function rebuilds the graph on each call
 * (e.g., `new Vec2(params[0], params[1])`), compilation will fail because it
 * captures only the initial graph structure.
 *
 * **When to use:**
 * - Residual functions that use the same Value objects throughout optimization
 * - Fixed computation graphs where only parameter .data values change
 * - Performance-critical applications with many residual evaluations
 *
 * **When NOT to use:**
 * - Residual functions that reconstruct Value objects from params array
 * - Dynamic graphs that change structure based on parameter values
 *
 * **Performance:** Eliminates Value.zeroGradTree() traversal and backward()
 * closure execution, providing 2-10x speedup for large Jacobian computations.
 *
 * @param residual - A single residual Value (output of residual function)
 * @param params - Array of parameter Values
 * @param rowIdx - Row index in the Jacobian matrix (baked into compiled code)
 * @returns Compiled function: (paramValues: number[], J: number[][]) => number
 */
export function compileResidualJacobian(residual: Value, params: Value[], rowIdx: number): (paramValues: number[], J: number[][]) => number {
  const paramNames = params.map(p => p.paramName!);

  const visited = new Set<Value>();
  const topoOrder: Value[] = [];
  const forwardCode: string[] = [];
  const backwardCode: string[] = [];
  let varCounter = 0;
  const nodeToVar = new Map<Value, string>();

  function getVarName(node: Value): string {
    if (node.paramName) return node.paramName;
    if (!nodeToVar.has(node)) {
      nodeToVar.set(node, `_v${varCounter++}`);
    }
    return nodeToVar.get(node)!;
  }

  function compileForward(node: Value) {
    if (visited.has(node)) return;
    visited.add(node);

    for (const child of (node as any).prev as Value[]) {
      compileForward(child);
    }

    topoOrder.push(node);

    if (!node.paramName) {
      if ((node as any).prev.length > 0) {
        const childCodes = ((node as any).prev as Value[]).map(c => getVarName(c));
        const code = node.getForwardCode(childCodes);
        forwardCode.push(`const ${getVarName(node)} = ${code};`);
      } else {
        forwardCode.push(`const ${getVarName(node)} = ${node.data};`);
      }
    }
  }

  compileForward(residual);

  const gradDeclarations = Array.from(nodeToVar.entries())
    .map(([node, varName]) => `let grad_${varName} = 0;`)
    .concat(paramNames.map(p => `let grad_${p} = 0;`));

  for (let i = topoOrder.length - 1; i >= 0; i--) {
    const node = topoOrder[i];
    if ((node as any).prev.length === 0) continue;

    const gradVar = `grad_${getVarName(node)}`;
    const childGrads = ((node as any).prev as Value[]).map((c: Value) => `grad_${getVarName(c)}`);
    const childVars = ((node as any).prev as Value[]).map((c: Value) => getVarName(c));

    backwardCode.push(node.getBackwardCode(gradVar, childGrads, childVars));
  }

  const outputVar = getVarName(residual);

  // Extract parameter values from array
  const paramExtractions = paramNames.map((p, i) => `const ${p} = paramValues[${i}];`).join('\n    ');

  // Build code to update Jacobian matrix in-place (rowIdx is baked in)
  const jacobianUpdates = paramNames.map((p, i) => `J[${rowIdx}][${i}] = grad_${p};`).join('\n    ');

  const functionBody = `
    ${paramExtractions}
    ${forwardCode.join('\n    ')}
    ${gradDeclarations.join('\n    ')}
    grad_${outputVar} = 1;
    ${backwardCode.join('\n    ')}
    ${jacobianUpdates}
    return ${outputVar};
  `;

  return new Function('paramValues', 'J', functionBody) as ((paramValues: number[], J: number[][]) => number);
}
