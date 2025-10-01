import { Value } from './Value';

export class ValueActivation {
  static relu(x: Value): Value {
    const r = Math.max(0, x.data);
    return Value.make(
      r,
      x, null,
      (out) => () => {
        if (x.requiresGrad) x.grad += (x.data > 0 ? 1 : 0) * out.grad;
      },
      `relu(${x.label})`
    );
  }

  static softplus(x: Value): Value {
    const s = Math.log(1 + Math.exp(x.data));
    return Value.make(
      s,
      x, null,
      (out) => () => {
        x.grad += 1 / (1 + Math.exp(-x.data)) * out.grad;
      },
      `softplus(${x.label})`
    );
  }

  static tanh(x: Value): Value {
    const t = Math.tanh(x.data);
    return Value.make(
      t,
      x, null,
      (out) => () => {
        if (x.requiresGrad) x.grad += (1 - t ** 2) * out.grad;
      },
      `tanh(${x.label})`
    );
  }

  static sigmoid(x: Value): Value {
    const s = 1 / (1 + Math.exp(-x.data));
    return Value.make(
      s,
      x, null,
      (out) => () => {
        if (x.requiresGrad) x.grad += s * (1 - s) * out.grad;
      },
      `sigmoid(${x.label})`
    );
  }
}