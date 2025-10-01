import { Value } from './Value';

export class CompilableValue extends Value {
  paramName?: string;
  _prev: CompilableValue[] = [];
  _op?: string;
  _backward: () => void = () => {};

  constructor(data: number, paramName?: string) {
    super(data, '', true);
    this.paramName = paramName;
  }

  backward(): void {
    const topo: CompilableValue[] = [];
    const visited = new Set<CompilableValue>();

    const buildTopo = (v: CompilableValue) => {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._prev) {
          buildTopo(child);
        }
        topo.push(v);
      }
    };

    buildTopo(this);
    this.grad = 1;

    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }

  getForwardCode(childCodes: string[]): string {
    if (this.paramName) return this.paramName;

    const [left, right] = childCodes;
    switch (this._op) {
      case '+': return `(${left} + ${right})`;
      case '-': return `(${left} - ${right})`;
      case '*': return `(${left} * ${right})`;
      case '/': return `(${left} / ${right})`;
      default: return String(this.data);
    }
  }

  getBackwardCode(gradVar: string, childGrads: [string, string]): string {
    const [leftGrad, rightGrad] = childGrads;

    switch (this._op) {
      case '+':
        return `${leftGrad} += ${gradVar}; ${rightGrad} += ${gradVar};`;
      case '-':
        return `${leftGrad} += ${gradVar}; ${rightGrad} -= ${gradVar};`;
      case '*': {
        const [left, right] = this._prev.map(v => (v as CompilableValue).getForwardCode([]));
        return `${leftGrad} += ${gradVar} * ${right}; ${rightGrad} += ${gradVar} * ${left};`;
      }
      case '/': {
        const [left, right] = this._prev.map(v => (v as CompilableValue).getForwardCode([]));
        return `${leftGrad} += ${gradVar} / ${right}; ${rightGrad} -= ${gradVar} * ${left} / (${right} * ${right});`;
      }
      default:
        return '';
    }
  }

  add(other: CompilableValue): CompilableValue {
    const out = new CompilableValue(this.data + other.data);
    out._prev = [this, other];
    out._op = '+';
    out._backward = () => {
      this.grad += out.grad;
      other.grad += out.grad;
    };
    return out;
  }

  sub(other: CompilableValue): CompilableValue {
    const out = new CompilableValue(this.data - other.data);
    out._prev = [this, other];
    out._op = '-';
    out._backward = () => {
      this.grad += out.grad;
      other.grad -= out.grad;
    };
    return out;
  }

  mul(other: CompilableValue): CompilableValue {
    const out = new CompilableValue(this.data * other.data);
    out._prev = [this, other];
    out._op = '*';
    out._backward = () => {
      this.grad += out.grad * other.data;
      other.grad += out.grad * this.data;
    };
    return out;
  }

  div(other: CompilableValue): CompilableValue {
    const out = new CompilableValue(this.data / other.data);
    out._prev = [this, other];
    out._op = '/';
    out._backward = () => {
      this.grad += out.grad / other.data;
      other.grad -= out.grad * this.data / (other.data * other.data);
    };
    return out;
  }
}

export function compileGradientFunction(output: CompilableValue, params: CompilableValue[]) {
  const paramNames = params.map(p => p.paramName!);

  const visited = new Set<CompilableValue>();
  const forwardCode: string[] = [];
  const backwardCode: string[] = [];
  let varCounter = 0;
  const nodeToVar = new Map<CompilableValue, string>();

  function getVarName(node: CompilableValue): string {
    if (node.paramName) return node.paramName;
    if (!nodeToVar.has(node)) {
      nodeToVar.set(node, `v${varCounter++}`);
    }
    return nodeToVar.get(node)!;
  }

  function compileForward(node: CompilableValue) {
    if (visited.has(node)) return;
    visited.add(node);

    for (const child of node._prev) {
      compileForward(child as CompilableValue);
    }

    if (!node.paramName && node._prev.length > 0) {
      const childCodes = node._prev.map(c => getVarName(c as CompilableValue));
      const code = node.getForwardCode(childCodes);
      forwardCode.push(`const ${getVarName(node)} = ${code};`);
    }
  }

  function compileBackward(node: CompilableValue) {
    if (node._prev.length === 0) return;

    const gradVar = `grad_${getVarName(node)}`;
    const childGrads: [string, string] = [
      `grad_${getVarName(node._prev[0] as CompilableValue)}`,
      `grad_${getVarName(node._prev[1] as CompilableValue)}`
    ];

    backwardCode.push(node.getBackwardCode(gradVar, childGrads));

    for (const child of node._prev) {
      compileBackward(child as CompilableValue);
    }
  }

  compileForward(output);

  const gradDeclarations = Array.from(nodeToVar.entries())
    .map(([node, varName]) => `let grad_${varName} = 0;`)
    .concat(paramNames.map(p => `let grad_${p} = 0;`));

  backwardCode.push(`grad_${getVarName(output)} = 1;`);
  compileBackward(output);

  const functionBody = `
    ${forwardCode.join('\n    ')}
    ${gradDeclarations.join('\n    ')}
    grad_${getVarName(output)} = 1;
    ${backwardCode.join('\n    ')}
    return [${paramNames.map(p => `grad_${p}`).join(', ')}];
  `;

  return new Function(...paramNames, functionBody) as (...args: number[]) => number[];
}
