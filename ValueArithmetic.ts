import { Value } from './Value';

export class ValueArithmetic {
  static add(a: Value, b: Value): Value {
    return Value.make(
      a.data + b.data,
      a, b,
      (out) => () => {
        if (a.requiresGrad) a.grad += 1 * out.grad;
        if (b.requiresGrad) b.grad += 1 * out.grad;
      },
      `(${a.label}+${b.label})`
    );
}
  static sqrt(a: Value): Value {
    if (a.data < 0) {
      throw new Error(`Cannot take sqrt of negative number: ${a.data}`);
    }
    const root = Math.sqrt(a.data);
    return Value.make(
      root,
      a, null,
      (out) => () => {
        if (a.requiresGrad) a.grad += 0.5 / root * out.grad;
      },
      `sqrt(${a.label})`
    );
  }

  static mul(a: Value, b: Value): Value {
    return Value.make(
      a.data * b.data,
      a, b,
      (out) => () => {
        if (a.requiresGrad) a.grad += b.data * out.grad;
        if (b.requiresGrad) b.grad += a.data * out.grad;
      },
      `(${a.label}*${b.label})`
    );
  }

  static sub(a: Value, b: Value): Value {
    return Value.make(
      a.data - b.data,
      a, b,
      (out) => () => {
        if (a.requiresGrad) a.grad += 1 * out.grad;
        if (b.requiresGrad) b.grad -= 1 * out.grad;
      },
      `(${a.label}-${b.label})`
    );
  }

  static div(a: Value, b: Value, eps = 1e-12): Value {
    if (Math.abs(b.data) < eps) {
      throw new Error(`Division by zero or near-zero encountered in div: denominator=${b.data}`);
    }
    const safe = b.data;
    return Value.make(
      a.data / safe,
      a, b,
      (out) => () => {
        if (a.requiresGrad) a.grad += (1 / safe) * out.grad;
        if (b.requiresGrad) b.grad -= (a.data / (safe ** 2)) * out.grad;
      },
      `(${a.label}/${b.label})`
    );
  }

  static pow(a: Value, exp: number): Value {
    if (typeof exp !== "number" || Number.isNaN(exp) || !Number.isFinite(exp)) {
      throw new Error(`Exponent must be a finite number, got ${exp}`);
    }
    if (a.data < 0 && Math.abs(exp % 1) > 1e-12) {
      throw new Error(`Cannot raise negative base (${a.data}) to non-integer exponent (${exp})`);
    }
    const safeBase = a.data;
    return Value.make(
      Math.pow(safeBase, exp),
      a, null,
      (out) => () => {
        if (a.requiresGrad) a.grad += exp * Math.pow(safeBase, exp - 1) * out.grad;
      },
      `(${a.label}^${exp})`
    );
  }

  static powValue(a: Value, b: Value, eps = 1e-12): Value {
    if (a.data < 0 && Math.abs(b.data % 1) > eps) {
      throw new Error(`Cannot raise negative base (${a.data}) to non-integer exponent (${b.data})`);
    }
    if (a.data === 0 && b.data <= 0) {
      throw new Error(`0 cannot be raised to zero or negative power: ${b.data}`);
    }
    const safeBase = a.data;
    return Value.make(
      Math.pow(safeBase, b.data),
      a, b,
      (out) => () => {
        a.grad += b.data * Math.pow(safeBase, b.data - 1) * out.grad;
        b.grad += Math.log(Math.max(eps, safeBase)) * Math.pow(safeBase, b.data) * out.grad;
      },
      `(${a.label}^${b.label})`
    );
  }

  static mod(a: Value, b: Value): Value {
    if (typeof b.data !== 'number' || b.data === 0) {
      throw new Error(`Modulo by zero encountered`);
    }
    return Value.make(
      a.data % b.data,
      a, b,
      (out) => () => {
        a.grad += 1 * out.grad;
        // No grad to b (modulus not used in most diff cases)
      },
      `(${a.label}%${b.label})`
    );
  }

  static abs(a: Value): Value {
    const d = Math.abs(a.data);
    return Value.make(
      d,
      a, null,
      (out) => () => {
        if (a.requiresGrad) a.grad += (a.data >= 0 ? 1 : -1) * out.grad;
      },
      `abs(${a.label})`
    );
  }

  static exp(a: Value): Value {
    const e = Math.exp(a.data);
    return Value.make(
      e,
      a, null,
      (out) => () => {
        if (a.requiresGrad) a.grad += e * out.grad;
      },
      `exp(${a.label})`
    );
  }

  static log(a: Value, eps = 1e-12): Value {
    if (a.data <= 0) {
      throw new Error(`Logarithm undefined for non-positive value: ${a.data}`);
    }
    const safe = Math.max(a.data, eps);
    const l = Math.log(safe);
    return Value.make(
      l,
      a, null,
      (out) => () => {
        if (a.requiresGrad) a.grad += (1 / safe) * out.grad;
      },
      `log(${a.label})`
    );
  }

  static min(a: Value, b: Value): Value {
    const d = Math.min(a.data, b.data);
    return Value.make(
      d,
      a, b,
      (out) => () => {
        if (a.requiresGrad) a.grad += (a.data < b.data ? 1 : 0) * out.grad;
        if (b.requiresGrad) b.grad += (b.data < a.data ? 1 : 0) * out.grad;
      },
      `min(${a.label},${b.label})`
    );
  }

  static max(a: Value, b: Value): Value {
    const d = Math.max(a.data, b.data);
    return Value.make(
      d,
      a, b,
      (out) => () => {
        if (a.requiresGrad) a.grad += (a.data > b.data ? 1 : 0) * out.grad;
        if (b.requiresGrad) b.grad += (b.data > a.data ? 1 : 0) * out.grad;
      },
      `max(${a.label},${b.label})`
    );
  }

  static floor(a: Value): Value {
    const fl = Math.floor(a.data);
    return Value.make(
      fl,
      a, null,
      () => () => {},
      `floor(${a.label})`
    );
  }

  static ceil(a: Value): Value {
    const cl = Math.ceil(a.data);
    return Value.make(
      cl,
      a, null,
      () => () => {},
      `ceil(${a.label})`
    );
  }

  static round(a: Value): Value {
    const rd = Math.round(a.data);
    return Value.make(
      rd,
      a, null,
      () => () => {},
      `round(${a.label})`
    );
  }

  static square(a: Value): Value {
    return ValueArithmetic.pow(a, 2);
  }

  static cube(a: Value): Value {
    return ValueArithmetic.pow(a, 3);
  }

  static reciprocal(a: Value, eps = 1e-12): Value {
    if (Math.abs(a.data) < eps) {
      throw new Error(`Reciprocal of zero or near-zero detected`);
    }
    return Value.make(
      1 / a.data,
      a, null,
      (out) => () => {
        if (a.requiresGrad) a.grad += -1 / (a.data * a.data) * out.grad;
      },
      `reciprocal(${a.label})`
    );
  }

  static clamp(a: Value, min: number, max: number): Value {
    let val = Math.max(min, Math.min(a.data, max));
    return Value.make(
      val,
      a, null,
      (out) => () => {
        a.grad += (a.data > min && a.data < max ? 1 : 0) * out.grad;
      },
      `clamp(${a.label},${min},${max})`
    );
  }

  static sum(vals: Value[]): Value {
    if (!vals.length) return new Value(0);
    return vals.reduce((a, b) => a.add(b));
  }

  static mean(vals: Value[]): Value {
    if (!vals.length) return new Value(0);
    return ValueArithmetic.sum(vals).div(vals.length);
  }

  static neg(a: Value): Value {
    return Value.make(
      -a.data,
      a, null,
      (out) => () => {
        if (a.requiresGrad) a.grad -= out.grad;
      },
      `(-${a.label})`
    );
  }

  static sign(a: Value): Value {
    const s = Math.sign(a.data);
    return Value.make(
      s,
      a, null,
      (out) => () => {
        // The derivative of sign(x) is 0 for x != 0.
        // At x = 0, the derivative is undefined (Dirac delta), but for practical purposes in ML,
        // we can define it as 0.
        if (a.requiresGrad) a.grad += 0 * out.grad;
      },
      `sign(${a.label})`
    );
  }
}
