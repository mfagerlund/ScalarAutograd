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
