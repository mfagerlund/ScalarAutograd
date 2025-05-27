import { Value } from './Value';

export class ValueComparison {
  static eq(a: Value, b: Value): Value {
    return Value.make(
      a.data === b.data ? 1 : 0,
      a, b,
      (out) => () => {
        // No gradient - discrete operation
      },
      `(${a.label}==${b.label})`
    );
  }

  static ifThenElse(cond: Value, thenVal: Value, elseVal: Value): Value {
    return Value.make(
      cond.data ? thenVal.data : elseVal.data,
      cond,
      cond.data ? thenVal : elseVal,
      (out) => () => {
        if (cond.data) {
          thenVal.grad += out.grad;
        } else {
          elseVal.grad += out.grad;
        }
      },
      `if(${cond.label}){${thenVal.label}}else{${elseVal.label}}`
    );
  }

  static neq(a: Value, b: Value): Value {
    return Value.make(
      a.data !== b.data ? 1 : 0,
      a, b,
      (out) => () => {
        // No gradient - discrete operation
      },
      `(${a.label}!=${b.label})`
    );
  }
  
  static gt(a: Value, b: Value): Value {
    return Value.make(
      a.data > b.data ? 1 : 0,
      a, b,
      (out) => () => {
        // No gradient - discrete operation
      },
      `(${a.label}>${b.label})`
    );
  }
  
  static lt(a: Value, b: Value): Value {
    return Value.make(
      a.data < b.data ? 1 : 0,
      a, b,
      (out) => () => {
        // No gradient - discrete operation
      },
      `(${a.label}<${b.label})`
    );
  }
  
  static gte(a: Value, b: Value): Value {
    return Value.make(
      a.data >= b.data ? 1 : 0,
      a, b,
      (out) => () => {
        // No gradient - discrete operation
      },
      `(${a.label}>=${b.label})`
    );
  }
  
  static lte(a: Value, b: Value): Value {
    return Value.make(
      a.data <= b.data ? 1 : 0,
      a, b,
      (out) => () => {
        // No gradient - discrete operation
      },
      `(${a.label}<=${b.label})`
    );
  }
}
