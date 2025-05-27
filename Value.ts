export type BackwardFn = () => void;
export { V } from './V';
export { Optimizer, SGD, Adam, AdamW, OptimizerOptions, AdamOptions } from './Optimizers';
export { Losses } from './Losses';

const EPS = 1e-12;


import { ValueTrig } from './ValueTrig';
import { ValueActivation } from './ValueActivation';
import { ValueArithmetic } from './ValueArithmetic';
import { ValueComparison } from './ValueComparison';

export class Value {
  static no_grad_mode = false;
  data: number;
  grad: number = 0;
  requiresGrad: boolean;
  private backwardFn: BackwardFn = () => {};
  private prev: Value[] = [];
  public label: string;

  constructor(data: number, label = "", requiresGrad = false) {
    if (typeof data !== 'number' || Number.isNaN(data) || !Number.isFinite(data)) {
      throw new Error(`Invalid number passed to Value: ${data}`);
    }
    this.data = data;
    this.label = label;
    this.requiresGrad = requiresGrad;
  }
  
  private static ensureValue(x: Value | number): Value {
    return typeof x === 'number' ? new Value(x) : x;
  }

  /**
   * Returns sin(this).
   * @returns New Value with sin.
   */
  sin(): Value {
    return ValueTrig.sin(this);
  }

  /**
   * Returns cos(this).
   * @returns New Value with cos.
   */
  cos(): Value {
    return ValueTrig.cos(this);
  }

  /**
   * Returns tan(this).
   * @returns New Value with tan.
   */
  tan(): Value {
    return ValueTrig.tan(this);
  }

  /**
   * Returns asin(this).
   * @returns New Value with asin.
   */
  asin(): Value {
    return ValueTrig.asin(this);
  }

  /**
   * Returns acos(this).
   * @returns New Value with acos.
   */
  acos(): Value {
    return ValueTrig.acos(this);
  }

  /**
   * Returns atan(this).
   * @returns New Value with atan.
   */
  atan(): Value {
    return ValueTrig.atan(this);
  }

  /**
   * Returns relu(this).
   * @returns New Value with relu.
   */
  relu(): Value {
    return ValueActivation.relu(this);
  }

  /**
   * Returns abs(this).
   * @returns New Value with abs.
   */
  abs(): Value {
    return ValueArithmetic.abs(this);
  }

  /**
   * Returns exp(this).
   * @returns New Value with exp.
   */
  exp(): Value {
    return ValueArithmetic.exp(this);
  }

  /**
   * Returns log(this).
   * @returns New Value with log.
   */
  log(): Value {
    return ValueArithmetic.log(this, EPS);
  }

  /**
   * Returns min(this, other).
   * @param other Value to compare
   * @returns New Value with min.
   */
  min(other: Value): Value {
    return ValueArithmetic.min(this, other);
  }

  /**
   * Returns max(this, other).
   * @param other Value to compare
   * @returns New Value with max.
   */
  max(other: Value): Value {
    return ValueArithmetic.max(this, other);
  }

  /**
   * Adds this and other.
   * @param other Value or number to add
   * @returns New Value with sum.
   */
  add(other: Value | number): Value {
    return ValueArithmetic.add(this, Value.ensureValue(other));
  }
  /**
   * Multiplies this and other.
   * @param other Value or number to multiply
   * @returns New Value with product.
   */
  mul(other: Value | number): Value {
    return ValueArithmetic.mul(this, Value.ensureValue(other));
  }

  /**
   * Subtracts other from this.
   * @param other Value or number to subtract
   * @returns New Value with difference.
   */
  sub(other: Value | number): Value {
    return ValueArithmetic.sub(this, Value.ensureValue(other));
  }

  /**
   * Divides this by other.
   * @param other Value or number divisor
   * @returns New Value with quotient.
   */
  div(other: Value | number): Value {
    return ValueArithmetic.div(this, Value.ensureValue(other), EPS);
  }

  /**
   * Raises this to the power exp.
   * @param exp Exponent
   * @returns New Value with pow(this, exp)
   */
  pow(exp: number): Value {
    return ValueArithmetic.pow(this, exp);
  }

  /**
   * Raises this to a dynamic Value (other).
   * @param other Exponent Value or number
   * @returns New Value with pow(this, other)
   */
  powValue(other: Value | number): Value {
    return ValueArithmetic.powValue(this, Value.ensureValue(other), EPS);
  }

  /**
   * Returns this modulo other.
   * @param other Divisor Value
   * @returns New Value with modulo.
   */
  mod(other: Value): Value {
    return ValueArithmetic.mod(this, other);
  }

  /**
   * Returns Value indicating if this equals other.
   * @param other Value to compare
   * @returns New Value (1 if equal, else 0)
   */
  eq(other: Value): Value {
    return ValueComparison.eq(this, other);
  }
  /**
   * Returns Value indicating if this not equals other.
   * @param other Value to compare
   * @returns New Value (1 if not equal, else 0)
   */
  neq(other: Value): Value {
    return ValueComparison.neq(this, other);
  }
  /**
   * Returns Value indicating if this greater than other.
   * @param other Value to compare
   * @returns New Value (1 if true, else 0)
   */
  gt(other: Value): Value {
    return ValueComparison.gt(this, other);
  }
  /**
   * Returns Value indicating if this less than other.
   * @param other Value to compare
   * @returns New Value (1 if true, else 0)
   */
  lt(other: Value): Value {
    return ValueComparison.lt(this, other);
  }
  /**
   * Returns Value indicating if this greater than or equal to other.
   * @param other Value to compare
   * @returns New Value (1 if true, else 0)
   */
  gte(other: Value): Value {
    return ValueComparison.gte(this, other);
  }
  /**
   * Returns Value indicating if this less than or equal to other.
   * @param other Value to compare
   * @returns New Value (1 if true, else 0)
   */
  lte(other: Value): Value {
    return ValueComparison.lte(this, other);
  }

  /**
   * Returns softplus(this).
   * @returns New Value with softplus.
   */
  softplus(): Value {
    return ValueActivation.softplus(this);
  }

  /**
   * Returns the floor of this Value.
   * @returns New Value with floor(data).
   */
  floor(): Value {
    return ValueArithmetic.floor(this);
  }
  /**
   * Returns the ceiling of this Value.
   * @returns New Value with ceil(data).
   */
  ceil(): Value {
    return ValueArithmetic.ceil(this);
  }
  /**
   * Returns the rounded value of this Value.
   * @returns New Value with rounded data.
   */
  round(): Value {
    return ValueArithmetic.round(this);
  }
  /**
   * Returns the square of this Value.
   * @returns New Value with squared data.
   */
  square(): Value {
    return ValueArithmetic.square(this);
  }
  /**
   * Returns the cube of this Value.
   * @returns New Value with cubed data.
   */
  cube(): Value {
    return ValueArithmetic.cube(this);
  }
  /**
   * Returns the reciprocal (1/x) of this Value.
   * @returns New Value with reciprocal.
   */
  reciprocal(): Value {
    return ValueArithmetic.reciprocal(this, EPS);
  }

  /**
   * Clamps this between min and max.
   * @param min Minimum value
   * @param max Maximum value
   * @returns New clamped Value
   */
  clamp(min: number, max: number): Value {
    return ValueArithmetic.clamp(this, min, max);
  }

  /**
   * Returns the negation (-this) Value.
   * @returns New Value which is the negation.
   */
  neg(): Value {
    return ValueArithmetic.neg(this);
  }

  /**
   * Returns the sum of the given Values.
   * @param vals Array of Value objects
   * @returns New Value holding their sum.
   */
  static sum(vals: Value[]): Value {
    return ValueArithmetic.sum(vals);
  }

  /**
   * Returns the mean of the given Values.
   * @param vals Array of Value objects
   * @returns New Value holding their mean.
   */
  static mean(vals: Value[]): Value {
    return ValueArithmetic.mean(vals);
  }

  /**
   * Returns tanh(this).
   * @returns New Value with tanh.
   */
  tanh(): Value {
    return ValueActivation.tanh(this);
  }

  /**
   * Returns sigmoid(this).
   * @returns New Value with sigmoid.
   */
  sigmoid(): Value {
    return ValueActivation.sigmoid(this);
  }

  /**
   * Performs a reverse-mode autodiff backward pass from this Value.
   * @param zeroGrad If true, zeroes all grads in the graph before backward
   */
  backward(zeroGrad = false): void {
    // Only allow backward on scalars (not arrays), i.e. single value outputs
    // (output shape check is redundant for this codebase, but keep to scalar-by-convention)
    if (zeroGrad) Value.zeroGradTree(this);

    const topo: Value[] = [];
    const visited = new Set<Value>();

    const buildTopo = (v: Value) => {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v.prev) {
          buildTopo(child);
        }
        topo.push(v);
      }
    };

    buildTopo(this);
    this.grad = 1;

    for (let i = topo.length - 1; i >= 0; i--) {
      if (topo[i].requiresGrad) {
        topo[i].backwardFn();
      }
    }
  }

  /**
   * Sets all grad fields in the computation tree (from root) to 0.
   * @param root Value to zero tree from
   */
  static zeroGradTree(root: Value): void {
    const visited = new Set<Value>();
    const visit = (v: Value) => {
      if (!visited.has(v)) {
        visited.add(v);
        v.grad = 0;
        for (const child of v.prev) visit(child);
      }
    };
    visit(root);
  }

  /**
   * Sets all grad fields in all supplied trees to 0.
   * @param vals Values whose trees to zero
   */
  static zeroGradAll(vals: Value[]): void {
    const visited = new Set<Value>();
    for (const v of vals) {
      const visit = (u: Value) => {
        if (!visited.has(u)) {
          visited.add(u);
          u.grad = 0;
          for (const child of u.prev) visit(child);
        }
      };
      visit(v);
    }
  }

  /**
   * Internal helper to construct a Value with correct backward fn and grads.
   * @param data Output value data
   * @param left Left operand Value
   * @param right Right operand Value or null
   * @param backwardFnBuilder Function to create backward closure
   * @param label Node label for debugging
   * @returns New Value node
   */
  static make(
    data: number,
    left: Value,
    right: Value | null,
    backwardFnBuilder: (out: Value) => BackwardFn,
    label: string
  ): Value {
    const requiresGrad = !Value.no_grad_mode && [left, right].filter(Boolean).some(v => v!.requiresGrad);
    const out = new Value(data, label, requiresGrad);
    out.prev = Value.no_grad_mode ? [] : ([left, right].filter(Boolean) as Value[]);
    if (requiresGrad) {
      out.backwardFn = backwardFnBuilder(out);
    }
    return out;
  }

  /**
   * Returns string representation for debugging.
   * @returns String summary of Value
   */
  toString(): string {
    return `Value(data=${this.data.toFixed(4)}, grad=${this.grad.toFixed(4)}, label=${this.label})`;
  }

  /**
   * Temporarily disables gradient tracking within the callback scope, like torch.no_grad().
   * Restores the previous state after running fn.
   */
  static withNoGrad<T>(fn: () => T): T {
    const prev = Value.no_grad_mode;
    Value.no_grad_mode = true;
    try {
      return fn();
    } finally {
      Value.no_grad_mode = prev;
    }
  }
}
