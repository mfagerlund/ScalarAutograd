import { ValueArithmetic } from './ValueArithmetic';
import { ValueTrig } from './ValueTrig';
import { ValueActivation } from './ValueActivation';
import { ValueComparison } from './ValueComparison';
import { Value } from './Value';

export class V {
 private static ensureValue(x: Value | number): Value {
   return typeof x === 'number' ? new Value(x) : x;
 }

 /**
  * Creates a constant Value (non-differentiable).
  * @param value The numeric value
  * @param label Optional label for the value
  * @returns New constant Value
  */
 static C(value: number, label = ""): Value {
   return new Value(value, label, false);
 }
  
 /**
  * Creates a weight Value (differentiable).
  * @param value The numeric value
  * @param label Optional label for the value
  * @returns New differentiable Value
  */
 static W(value: number, label = ""): Value {
   return new Value(value, label, true);
 }

 /**
  * Addition operation.
  * @param a First operand
  * @param b Second operand
  * @returns New Value with sum
  */
 static add(a: Value | number, b: Value | number): Value {
   return ValueArithmetic.add(V.ensureValue(a), V.ensureValue(b));
 }

 /**
  * Multiplication operation.
  * @param a First operand
  * @param b Second operand
  * @returns New Value with product
  */
 static mul(a: Value | number, b: Value | number): Value {
   return ValueArithmetic.mul(V.ensureValue(a), V.ensureValue(b));
 }

 /**
  * Subtraction operation.
  * @param a First operand
  * @param b Second operand
  * @returns New Value with difference
  */
 static sub(a: Value | number, b: Value | number): Value {
   return ValueArithmetic.sub(V.ensureValue(a), V.ensureValue(b));
 }

 /**
  * Division operation.
  * @param a Dividend
  * @param b Divisor
  * @param eps Small epsilon to prevent division by zero
  * @returns New Value with quotient
  */
 static div(a: Value | number, b: Value | number, eps = 1e-12): Value {
   return ValueArithmetic.div(V.ensureValue(a), V.ensureValue(b), eps);
 }

 /**
  * Power operation with numeric exponent.
  * @param a Base
  * @param exp Exponent
  * @returns New Value with result
  */
 static pow(a: Value | number, exp: number): Value {
   return ValueArithmetic.pow(V.ensureValue(a), exp);
 }

 /**
  * Power operation with Value exponent.
  * @param a Base
  * @param b Exponent
  * @param eps Small epsilon for logarithm
  * @returns New Value with result
  */
 static powValue(a: Value | number, b: Value | number, eps = 1e-12): Value {
   return ValueArithmetic.powValue(V.ensureValue(a), V.ensureValue(b), eps);
 }

 /**
  * Modulo operation.
  * @param a Dividend
  * @param b Divisor
  * @returns New Value with remainder
  */
 static mod(a: Value | number, b: Value | number): Value {
   return ValueArithmetic.mod(V.ensureValue(a), V.ensureValue(b));
 }

 /**
  * Absolute value operation.
  * @param a Input value
  * @returns New Value with absolute value
  */
 static abs(a: Value | number): Value {
   return ValueArithmetic.abs(V.ensureValue(a));
 }

 /**
  * Exponential function.
  * @param a Input value
  * @returns New Value with e^a
  */
 static exp(a: Value | number): Value {
   return ValueArithmetic.exp(V.ensureValue(a));
 }

 /**
  * Natural logarithm.
  * @param a Input value
  * @param eps Small epsilon for numerical stability
  * @returns New Value with ln(a)
  */
 static log(a: Value | number, eps = 1e-12): Value {
   return ValueArithmetic.log(V.ensureValue(a), eps);
 }

 /**
  * Minimum of two values.
  * @param a First value
  * @param b Second value
  * @returns New Value with minimum
  */
 static min(a: Value | number, b: Value | number): Value {
   return ValueArithmetic.min(V.ensureValue(a), V.ensureValue(b));
 }

 /**
  * Maximum of two values.
  * @param a First value
  * @param b Second value
  * @returns New Value with maximum
  */
 static max(a: Value | number, b: Value | number): Value {
   return ValueArithmetic.max(V.ensureValue(a), V.ensureValue(b));
 }

 /**
  * Floor function.
  * @param a Input value
  * @returns New Value with floor
  */
 static floor(a: Value | number): Value {
   return ValueArithmetic.floor(V.ensureValue(a));
 }

 /**
  * Ceiling function.
  * @param a Input value
  * @returns New Value with ceiling
  */
 static ceil(a: Value | number): Value {
   return ValueArithmetic.ceil(V.ensureValue(a));
 }

 /**
  * Round function.
  * @param a Input value
  * @returns New Value rounded to nearest integer
  */
 static round(a: Value | number): Value {
   return ValueArithmetic.round(V.ensureValue(a));
 }

 /**
  * Square function.
  * @param a Input value
  * @returns New Value with a B2
  */
 static square(a: Value | number): Value {
   return ValueArithmetic.square(V.ensureValue(a));
 }

 /**
  * Cube function.
  * @param a Input value
  * @returns New Value with a B3
  */
 static cube(a: Value | number): Value {
   return ValueArithmetic.cube(V.ensureValue(a));
 }

 /**
  * Reciprocal function.
  * @param a Input value
  * @param eps Small epsilon to prevent division by zero
  * @returns New Value with 1/a
  */
 static reciprocal(a: Value | number, eps = 1e-12): Value {
   return ValueArithmetic.reciprocal(V.ensureValue(a), eps);
 }

 /**
  * Clamp function.
  * @param a Input value
  * @param min Minimum bound
  * @param max Maximum bound
  * @returns New Value clamped between min and max
  */
 static clamp(a: Value | number, min: number, max: number): Value {
   return ValueArithmetic.clamp(V.ensureValue(a), min, max);
 }

 /**
  * Negation operation.
  * @param a Input value
  * @returns New Value which is negation
  */
 static neg(a: Value | number): Value {
   return ValueArithmetic.neg(V.ensureValue(a));
 }

 /**
  * Sum of array of values.
  * @param vals Array of values
  * @returns New Value with sum
  */
 static sum(vals: (Value | number)[]): Value {
   return ValueArithmetic.sum(vals.map(V.ensureValue));
 }

 /**
  * Mean of array of values.
  * @param vals Array of values
  * @returns New Value with mean
  */
 static mean(vals: (Value | number)[]): Value {
   return ValueArithmetic.mean(vals.map(V.ensureValue));
 }

 /**
  * Sine function.
  * @param x Input value
  * @returns New Value with sin(x)
  */
 static sin(x: Value | number): Value {
   return ValueTrig.sin(V.ensureValue(x));
 }

 /**
  * Cosine function.
  * @param x Input value
  * @returns New Value with cos(x)
  */
 static cos(x: Value | number): Value {
   return ValueTrig.cos(V.ensureValue(x));
 }

 /**
  * Tangent function.
  * @param x Input value
  * @returns New Value with tan(x)
  */
 static tan(x: Value | number): Value {
   return ValueTrig.tan(V.ensureValue(x));
 }

 /**
  * Arcsine function.
  * @param x Input value
  * @returns New Value with asin(x)
  */
 static asin(x: Value | number): Value {
   return ValueTrig.asin(V.ensureValue(x));
 }

 /**
  * Arccosine function.
  * @param x Input value
  * @returns New Value with acos(x)
  */
 static acos(x: Value | number): Value {
   return ValueTrig.acos(V.ensureValue(x));
 }

 /**
  * Arctangent function.
  * @param x Input value
  * @returns New Value with atan(x)
  */
 static atan(x: Value | number): Value {
   return ValueTrig.atan(V.ensureValue(x));
 }

 /**
  * ReLU activation function.
  * @param x Input value
  * @returns New Value with max(0, x)
  */
 static relu(x: Value | number): Value {
   return ValueActivation.relu(V.ensureValue(x));
 }

 /**
  * Softplus activation function.
  * @param x Input value
  * @returns New Value with ln(1 + e^x)
  */
 static softplus(x: Value | number): Value {
   return ValueActivation.softplus(V.ensureValue(x));
 }

 /**
  * Hyperbolic tangent function.
  * @param x Input value
  * @returns New Value with tanh(x)
  */
 static tanh(x: Value | number): Value {
   return ValueActivation.tanh(V.ensureValue(x));
 }

 /**
  * Sigmoid activation function.
  * @param x Input value
  * @returns New Value with 1/(1 + e^(-x))
  */
 static sigmoid(x: Value | number): Value {
   return ValueActivation.sigmoid(V.ensureValue(x));
 }

 /**
  * Equal comparison operation.
  * @param a First operand
  * @param b Second operand
  * @returns New Value with 1 if equal, 0 otherwise
  */
 static eq(a: Value | number, b: Value | number): Value {
   return ValueComparison.eq(V.ensureValue(a), V.ensureValue(b));
 }

 /**
  * Not equal comparison operation.
  * @param a First operand
  * @param b Second operand
  * @returns New Value with 1 if not equal, 0 otherwise
  */
 static neq(a: Value | number, b: Value | number): Value {
   return ValueComparison.neq(V.ensureValue(a), V.ensureValue(b));
 }

 /**
  * Greater than comparison operation.
  * @param a First operand
  * @param b Second operand
  * @returns New Value with 1 if a > b, 0 otherwise
  */
 static gt(a: Value | number, b: Value | number): Value {
   return ValueComparison.gt(V.ensureValue(a), V.ensureValue(b));
 }

 /**
  * Less than comparison operation.
  * @param a First operand
  * @param b Second operand
  * @returns New Value with 1 if a < b, 0 otherwise
  */
 static lt(a: Value | number, b: Value | number): Value {
   return ValueComparison.lt(V.ensureValue(a), V.ensureValue(b));
 }

 /**
  * Greater than or equal comparison operation.
  * @param a First operand
  * @param b Second operand
  * @returns New Value with 1 if a >= b, 0 otherwise
  */
 static gte(a: Value | number, b: Value | number): Value {
   return ValueComparison.gte(V.ensureValue(a), V.ensureValue(b));
 }

 /**
  * Less than or equal comparison operation.
  * @param a First operand
  * @param b Second operand
  * @returns New Value with 1 if a <= b, 0 otherwise
  */
 static lte(a: Value | number, b: Value | number): Value {
   return ValueComparison.lte(V.ensureValue(a), V.ensureValue(b));
 }

 static ifThenElse(cond: Value | number, thenVal: Value | number, elseVal: Value | number): Value {
    // cond: Value, thenVal: Value, elseVal: Value
    return ValueComparison.ifThenElse(V.ensureValue(cond), V.ensureValue(thenVal), V.ensureValue(elseVal));
  }

   /**
   * Square root function.
   * @param a Input value
   * @returns New Value with sqrt(a)
   */
   static sqrt(a: Value | number): Value {
    return ValueArithmetic.sqrt(V.ensureValue(a));
  }

  /**
   * Sign function.
   * @param a Input value
   * @returns New Value with sign(a)
   */
  static sign(a: Value | number): Value {
    return ValueArithmetic.sign(V.ensureValue(a));
  }
}
