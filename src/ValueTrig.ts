import { Value } from './Value';

export class ValueTrig {
  static sin(x: Value): Value {
    const s = Math.sin(x.data);
    return Value.make(
      s,
      x, null,
      (out) => () => {
        if (x.requiresGrad) x.grad += Math.cos(x.data) * out.grad;
      },
      `sin(${x.label})`
    );
  }
  static cos(x: Value): Value {
    const c = Math.cos(x.data);
    return Value.make(
      c,
      x, null,
      (out) => () => {
        if (x.requiresGrad) x.grad += -Math.sin(x.data) * out.grad;
      },
      `cos(${x.label})`
    );
  }
  static tan(x: Value): Value {
    const t = Math.tan(x.data);
    return Value.make(
      t,
      x, null,
      (out) => () => {
        if (x.requiresGrad) x.grad += (1 / (Math.cos(x.data) ** 2)) * out.grad;
      },
      `tan(${x.label})`
    );
  }
  static asin(x: Value): Value {
    const s = Math.asin(x.data);
    return Value.make(
      s,
      x, null,
      (out) => () => {
        if (x.requiresGrad) x.grad += (1 / Math.sqrt(1 - x.data * x.data)) * out.grad;
      },
      `asin(${x.label})`
    );
  }
  static acos(x: Value): Value {
    const c = Math.acos(x.data);
    return Value.make(
      c,
      x, null,
      (out) => () => {
        if (x.requiresGrad) x.grad += (-1 / Math.sqrt(1 - x.data * x.data)) * out.grad;
      },
      `acos(${x.label})`
    );
  }
  static atan(x: Value): Value {
    const a = Math.atan(x.data);
    return Value.make(
      a,
      x, null,
      (out) => () => {
        if (x.requiresGrad) x.grad += (1 / (1 + x.data * x.data)) * out.grad;
      },
      `atan(${x.label})`
    );
  }
}