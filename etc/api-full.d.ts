/**
 * Function type for backward pass computations in automatic differentiation.
 * @public
 */
export type BackwardFn = () => void;
export { V } from './V';
export { Optimizer, SGD, Adam, AdamW } from './Optimizers';
export type { OptimizerOptions, AdamOptions } from './Optimizers';
export { Losses } from './Losses';
export type { NonlinearLeastSquaresOptions, NonlinearLeastSquaresResult } from './NonlinearLeastSquares';
export { Vec2 } from './Vec2';
export { Vec3 } from './Vec3';
/**
 * Represents a scalar value in the computational graph for automatic differentiation.
 * Supports forward computation and reverse-mode autodiff (backpropagation).
 * @public
 */
export declare class Value {
    /**
     * Global flag to disable gradient tracking. Use Value.withNoGrad() instead of setting directly.
     * @public
     */
    static no_grad_mode: boolean;
    /**
     * The numeric value stored in this node.
     * @public
     */
    data: number;
    /**
     * The gradient of the output with respect to this value.
     * @public
     */
    grad: number;
    /**
     * Whether this value participates in gradient computation.
     * @public
     */
    requiresGrad: boolean;
    private backwardFn;
    private prev;
    /**
     * Optional label for debugging and visualization.
     * @public
     */
    label: string;
    constructor(data: number, label?: string, requiresGrad?: boolean);
    private static ensureValue;
    /**
     * Returns sin(this).
     * @returns New Value with sin.
     */
    sin(): Value;
    /**
     * Returns cos(this).
     * @returns New Value with cos.
     */
    cos(): Value;
    /**
     * Returns tan(this).
     * @returns New Value with tan.
     */
    tan(): Value;
    /**
     * Returns asin(this).
     * @returns New Value with asin.
     */
    asin(): Value;
    /**
     * Returns acos(this).
     * @returns New Value with acos.
     */
    acos(): Value;
    /**
     * Returns atan(this).
     * @returns New Value with atan.
     */
    atan(): Value;
    /**
     * Returns relu(this).
     * @returns New Value with relu.
     */
    relu(): Value;
    /**
     * Returns abs(this).
     * @returns New Value with abs.
     */
    abs(): Value;
    /**
     * Returns exp(this).
     * @returns New Value with exp.
     */
    exp(): Value;
    /**
     * Returns log(this).
     * @returns New Value with log.
     */
    log(): Value;
    /**
     * Returns min(this, other).
     * @param other Value to compare
     * @returns New Value with min.
     */
    min(other: Value): Value;
    /**
     * Returns max(this, other).
     * @param other Value to compare
     * @returns New Value with max.
     */
    max(other: Value): Value;
    /**
     * Adds this and other.
     * @param other Value or number to add
     * @returns New Value with sum.
     */
    add(other: Value | number): Value;
    /**
     * Multiplies this and other.
     * @param other Value or number to multiply
     * @returns New Value with product.
     */
    mul(other: Value | number): Value;
    /**
     * Subtracts other from this.
     * @param other Value or number to subtract
     * @returns New Value with difference.
     */
    sub(other: Value | number): Value;
    /**
     * Divides this by other.
     * @param other Value or number divisor
     * @returns New Value with quotient.
     */
    div(other: Value | number): Value;
    /**
     * Raises this to the power exp.
     * @param exp Exponent
     * @returns New Value with pow(this, exp)
     */
    pow(exp: number): Value;
    /**
     * Raises this to a dynamic Value (other).
     * @param other Exponent Value or number
     * @returns New Value with pow(this, other)
     */
    powValue(other: Value | number): Value;
    /**
     * Returns this modulo other.
     * @param other Divisor Value
     * @returns New Value with modulo.
     */
    mod(other: Value): Value;
    /**
     * Returns Value indicating if this equals other.
     * @param other Value to compare
     * @returns New Value (1 if equal, else 0)
     */
    eq(other: Value): Value;
    /**
     * Returns Value indicating if this not equals other.
     * @param other Value to compare
     * @returns New Value (1 if not equal, else 0)
     */
    neq(other: Value): Value;
    /**
     * Returns Value indicating if this greater than other.
     * @param other Value to compare
     * @returns New Value (1 if true, else 0)
     */
    gt(other: Value): Value;
    /**
     * Returns Value indicating if this less than other.
     * @param other Value to compare
     * @returns New Value (1 if true, else 0)
     */
    lt(other: Value): Value;
    /**
     * Returns Value indicating if this greater than or equal to other.
     * @param other Value to compare
     * @returns New Value (1 if true, else 0)
     */
    gte(other: Value): Value;
    /**
     * Returns Value indicating if this less than or equal to other.
     * @param other Value to compare
     * @returns New Value (1 if true, else 0)
     */
    lte(other: Value): Value;
    /**
     * Returns softplus(this).
     * @returns New Value with softplus.
     */
    softplus(): Value;
    /**
     * Returns the floor of this Value.
     * @returns New Value with floor(data).
     */
    floor(): Value;
    /**
     * Returns the ceiling of this Value.
     * @returns New Value with ceil(data).
     */
    ceil(): Value;
    /**
     * Returns the rounded value of this Value.
     * @returns New Value with rounded data.
     */
    round(): Value;
    /**
     * Returns the square of this Value.
     * @returns New Value with squared data.
     */
    square(): Value;
    /**
     * Returns the cube of this Value.
     * @returns New Value with cubed data.
     */
    cube(): Value;
    /**
     * Returns the reciprocal (1/x) of this Value.
     * @returns New Value with reciprocal.
     */
    reciprocal(): Value;
    /**
     * Clamps this between min and max.
     * @param min Minimum value
     * @param max Maximum value
     * @returns New clamped Value
     */
    clamp(min: number, max: number): Value;
    /**
     * Returns the negation (-this) Value.
     * @returns New Value which is the negation.
     */
    neg(): Value;
    /**
     * Returns sign(this).
     * @returns New Value with sign.
     */
    sign(): Value;
    /**
     * Returns the sum of the given Values.
     * @param vals Array of Value objects
     * @returns New Value holding their sum.
     */
    static sum(vals: Value[]): Value;
    /**
     * Returns the mean of the given Values.
     * @param vals Array of Value objects
     * @returns New Value holding their mean.
     */
    static mean(vals: Value[]): Value;
    /**
     * Returns tanh(this).
     * @returns New Value with tanh.
     */
    tanh(): Value;
    /**
     * Returns sigmoid(this).
     * @returns New Value with sigmoid.
     */
    sigmoid(): Value;
    /**
     * Performs a reverse-mode autodiff backward pass from this Value.
     * @param zeroGrad If true, zeroes all grads in the graph before backward
     */
    backward(zeroGrad?: boolean): void;
    /**
     * Sets all grad fields in the computation tree (from root) to 0.
     * @param root Value to zero tree from
     */
    static zeroGradTree(root: Value): void;
    /**
     * Sets all grad fields in all supplied trees to 0.
     * @param vals Values whose trees to zero
     */
    static zeroGradAll(vals: Value[]): void;
    /**
     * Internal helper to construct a Value with correct backward fn and grads.
     * @param data Output value data
     * @param left Left operand Value
     * @param right Right operand Value or null
     * @param backwardFnBuilder Function to create backward closure
     * @param label Node label for debugging
     * @returns New Value node
     */
    static make(data: number, left: Value, right: Value | null, backwardFnBuilder: (out: Value) => BackwardFn, label: string): Value;
    /**
     * Returns string representation for debugging.
     * @returns String summary of Value
     */
    toString(): string;
    /**
     * Temporarily disables gradient tracking within the callback scope, like torch.no_grad().
     * Restores the previous state after running fn.
     */
    static withNoGrad<T>(fn: () => T): T;
}
import { Value } from './Value';
import { NonlinearLeastSquaresOptions, NonlinearLeastSquaresResult } from './NonlinearLeastSquares';
/**
 * Static factory and operation methods for Value objects.
 * Provides functional-style API for creating and manipulating Values.
 * @public
 */
export declare class V {
    private static ensureValue;
    /**
     * Creates a constant Value (non-differentiable).
     * @param value The numeric value
     * @param label Optional label for the value
     * @returns New constant Value
     */
    static C(value: number, label?: string): Value;
    /**
     * Creates a weight Value (differentiable).
     * @param value The numeric value
     * @param label Optional label for the value
     * @returns New differentiable Value
     */
    static W(value: number, label?: string): Value;
    /**
     * Addition operation.
     * @param a First operand
     * @param b Second operand
     * @returns New Value with sum
     */
    static add(a: Value | number, b: Value | number): Value;
    /**
     * Multiplication operation.
     * @param a First operand
     * @param b Second operand
     * @returns New Value with product
     */
    static mul(a: Value | number, b: Value | number): Value;
    /**
     * Subtraction operation.
     * @param a First operand
     * @param b Second operand
     * @returns New Value with difference
     */
    static sub(a: Value | number, b: Value | number): Value;
    /**
     * Division operation.
     * @param a Dividend
     * @param b Divisor
     * @param eps Small epsilon to prevent division by zero
     * @returns New Value with quotient
     */
    static div(a: Value | number, b: Value | number, eps?: number): Value;
    /**
     * Power operation with numeric exponent.
     * @param a Base
     * @param exp Exponent
     * @returns New Value with result
     */
    static pow(a: Value | number, exp: number): Value;
    /**
     * Power operation with Value exponent.
     * @param a Base
     * @param b Exponent
     * @param eps Small epsilon for logarithm
     * @returns New Value with result
     */
    static powValue(a: Value | number, b: Value | number, eps?: number): Value;
    /**
     * Modulo operation.
     * @param a Dividend
     * @param b Divisor
     * @returns New Value with remainder
     */
    static mod(a: Value | number, b: Value | number): Value;
    /**
     * Absolute value operation.
     * @param a Input value
     * @returns New Value with absolute value
     */
    static abs(a: Value | number): Value;
    /**
     * Exponential function.
     * @param a Input value
     * @returns New Value with e^a
     */
    static exp(a: Value | number): Value;
    /**
     * Natural logarithm.
     * @param a Input value
     * @param eps Small epsilon for numerical stability
     * @returns New Value with ln(a)
     */
    static log(a: Value | number, eps?: number): Value;
    /**
     * Minimum of two values.
     * @param a First value
     * @param b Second value
     * @returns New Value with minimum
     */
    static min(a: Value | number, b: Value | number): Value;
    /**
     * Maximum of two values.
     * @param a First value
     * @param b Second value
     * @returns New Value with maximum
     */
    static max(a: Value | number, b: Value | number): Value;
    /**
     * Floor function.
     * @param a Input value
     * @returns New Value with floor
     */
    static floor(a: Value | number): Value;
    /**
     * Ceiling function.
     * @param a Input value
     * @returns New Value with ceiling
     */
    static ceil(a: Value | number): Value;
    /**
     * Round function.
     * @param a Input value
     * @returns New Value rounded to nearest integer
     */
    static round(a: Value | number): Value;
    /**
     * Square function.
     * @param a Input value
     * @returns New Value with a B2
     */
    static square(a: Value | number): Value;
    /**
     * Cube function.
     * @param a Input value
     * @returns New Value with a B3
     */
    static cube(a: Value | number): Value;
    /**
     * Reciprocal function.
     * @param a Input value
     * @param eps Small epsilon to prevent division by zero
     * @returns New Value with 1/a
     */
    static reciprocal(a: Value | number, eps?: number): Value;
    /**
     * Clamp function.
     * @param a Input value
     * @param min Minimum bound
     * @param max Maximum bound
     * @returns New Value clamped between min and max
     */
    static clamp(a: Value | number, min: number, max: number): Value;
    /**
     * Negation operation.
     * @param a Input value
     * @returns New Value which is negation
     */
    static neg(a: Value | number): Value;
    /**
     * Sum of array of values.
     * @param vals Array of values
     * @returns New Value with sum
     */
    static sum(vals: (Value | number)[]): Value;
    /**
     * Mean of array of values.
     * @param vals Array of values
     * @returns New Value with mean
     */
    static mean(vals: (Value | number)[]): Value;
    /**
     * Sine function.
     * @param x Input value
     * @returns New Value with sin(x)
     */
    static sin(x: Value | number): Value;
    /**
     * Cosine function.
     * @param x Input value
     * @returns New Value with cos(x)
     */
    static cos(x: Value | number): Value;
    /**
     * Tangent function.
     * @param x Input value
     * @returns New Value with tan(x)
     */
    static tan(x: Value | number): Value;
    /**
     * Arcsine function.
     * @param x Input value
     * @returns New Value with asin(x)
     */
    static asin(x: Value | number): Value;
    /**
     * Arccosine function.
     * @param x Input value
     * @returns New Value with acos(x)
     */
    static acos(x: Value | number): Value;
    /**
     * Arctangent function.
     * @param x Input value
     * @returns New Value with atan(x)
     */
    static atan(x: Value | number): Value;
    /**
     * ReLU activation function.
     * @param x Input value
     * @returns New Value with max(0, x)
     */
    static relu(x: Value | number): Value;
    /**
     * Softplus activation function.
     * @param x Input value
     * @returns New Value with ln(1 + e^x)
     */
    static softplus(x: Value | number): Value;
    /**
     * Hyperbolic tangent function.
     * @param x Input value
     * @returns New Value with tanh(x)
     */
    static tanh(x: Value | number): Value;
    /**
     * Sigmoid activation function.
     * @param x Input value
     * @returns New Value with 1/(1 + e^(-x))
     */
    static sigmoid(x: Value | number): Value;
    /**
     * Equal comparison operation.
     * @param a First operand
     * @param b Second operand
     * @returns New Value with 1 if equal, 0 otherwise
     */
    static eq(a: Value | number, b: Value | number): Value;
    /**
     * Not equal comparison operation.
     * @param a First operand
     * @param b Second operand
     * @returns New Value with 1 if not equal, 0 otherwise
     */
    static neq(a: Value | number, b: Value | number): Value;
    /**
     * Greater than comparison operation.
     * @param a First operand
     * @param b Second operand
     * @returns New Value with 1 if a > b, 0 otherwise
     */
    static gt(a: Value | number, b: Value | number): Value;
    /**
     * Less than comparison operation.
     * @param a First operand
     * @param b Second operand
     * @returns New Value with 1 if a < b, 0 otherwise
     */
    static lt(a: Value | number, b: Value | number): Value;
    /**
     * Greater than or equal comparison operation.
     * @param a First operand
     * @param b Second operand
     * @returns New Value with 1 if a >= b, 0 otherwise
     */
    static gte(a: Value | number, b: Value | number): Value;
    /**
     * Less than or equal comparison operation.
     * @param a First operand
     * @param b Second operand
     * @returns New Value with 1 if a <= b, 0 otherwise
     */
    static lte(a: Value | number, b: Value | number): Value;
    /**
     * Conditional branching operation (if-then-else).
     * Evaluates condition and returns thenVal if cond is truthy, otherwise elseVal.
     * Gradients flow through the selected branch.
     * @param cond - Condition value (truthy if non-zero)
     * @param thenVal - Value returned if condition is true
     * @param elseVal - Value returned if condition is false
     * @returns New Value with the selected branch
     * @public
     */
    static ifThenElse(cond: Value | number, thenVal: Value | number, elseVal: Value | number): Value;
    /**
    * Square root function.
    * @param a Input value
    * @returns New Value with sqrt(a)
    */
    static sqrt(a: Value | number): Value;
    /**
     * Sign function.
     * @param a Input value
     * @returns New Value with sign(a)
     */
    static sign(a: Value | number): Value;
    /**
     * Nonlinear least squares solver using Levenberg-Marquardt algorithm.
     * Minimizes sum of squared residuals using automatic differentiation.
     * @param params Array of trainable Value parameters to optimize
     * @param residualFn Function that computes residuals from current parameters
     * @param options Optional solver configuration
     * @returns Result object with success status, iterations, and final cost
     */
    static nonlinearLeastSquares(params: Value[], residualFn: (params: Value[]) => Value[], options?: NonlinearLeastSquaresOptions): NonlinearLeastSquaresResult;
}
import { Value } from "./Value";
/**
 * Abstract base class for all optimizers.
 * Ensures only requiresGrad parameters are optimized.
 * @public
 */
export declare abstract class Optimizer {
    /**
     * Array of trainable Value parameters filtered to only those requiring gradients.
     * @public
     */
    protected trainables: Value[];
    /**
     * Learning rate for parameter updates.
     * @public
     */
    learningRate: number;
    /**
     * Constructs an Optimizer.
     * @param trainables - Array of Value parameters to optimize.
     * @param learningRate - Learning rate for updates.
     */
    constructor(trainables: Value[], learningRate: number);
    /**
     * Performs a parameter update step.
     * @public
     */
    abstract step(): void;
    /**
     * Resets optimizer state for a specific trainable parameter.
     * @param trainable - The Value parameter to reset state for.
     * @public
     */
    abstract resetStateFor(trainable: Value): void;
    /**
     * Sets grads of all trainables to zero.
     */
    zeroGrad(): void;
    /**
     * Clips global norm of gradients as regularization.
     * @param maxNorm Maximum allowed norm for gradients.
     */
    clipGradients(maxNorm: number): void;
}
/**
 * Optional arguments for basic optimizers.
 * @public
 */
export interface OptimizerOptions {
    /**
     * Overrides the step size for parameter updates (default varies by optimizer).
     * @public
     */
    learningRate?: number;
    /**
     * L2 regularization multiplier (default 0). Ignored for plain SGD.
     * @public
     */
    weightDecay?: number;
    /**
     * Maximum absolute value for gradient updates (default 0: no clipping).
     * @public
     */
    gradientClip?: number;
}
/**
 * Stochastic Gradient Descent (SGD) optimizer. Accepts weightDecay and gradientClip for API consistency (ignored).
 * @public
 */
export declare class SGD extends Optimizer {
    private weightDecay;
    private gradientClip;
    /**
     * Constructs an SGD optimizer.
     * @param trainables - Array of Value parameters to optimize.
     * @param opts - Optional parameters (learningRate, weightDecay, gradientClip).
     */
    constructor(trainables: Value[], opts?: OptimizerOptions);
    /**
     * Performs a parameter update using standard SGD.
     * @public
     */
    step(): void;
    /**
     * Resets optimizer state for a trainable (no-op for SGD).
     * @param trainable - The Value parameter to reset state for.
     * @public
     */
    resetStateFor(trainable: Value): void;
}
/**
 * Adam and AdamW optimizer parameters.
 * Extends OptimizerOptions.
 * @public
 */
export interface AdamOptions extends OptimizerOptions {
    /**
     * Exponential decay rate for 1st moment (default 0.9).
     * @public
     */
    beta1?: number;
    /**
     * Exponential decay rate for 2nd moment (default 0.999).
     * @public
     */
    beta2?: number;
    /**
     * Numerical stability fudge factor (default 1e-8).
     * @public
     */
    epsilon?: number;
}
/**
 * Adam optimizer, supports decoupled weight decay and gradient clipping.
 * @public
 */
export declare class Adam extends Optimizer {
    private beta1;
    private beta2;
    private epsilon;
    private weightDecay;
    private gradientClip;
    private m;
    private v;
    private stepCount;
    /**
     * Constructs an Adam optimizer.
     * @param trainables - Array of Value parameters to optimize.
     * @param opts - Optional parameters (learningRate, weightDecay, gradientClip, beta1, beta2, epsilon).
     */
    constructor(trainables: Value[], opts?: AdamOptions);
    /**
     * Performs a parameter update using Adam optimization.
     * @public
     */
    step(): void;
    /**
     * Resets optimizer state (momentum and velocity) for a specific trainable.
     * @param trainable - The Value parameter to reset state for.
     * @public
     */
    resetStateFor(trainable: Value): void;
}
/**
 * AdamW optimizer, supports decoupled weight decay and gradient clipping (same options as Adam).
 * @public
 */
export declare class AdamW extends Optimizer {
    private beta1;
    private beta2;
    private epsilon;
    private weightDecay;
    private gradientClip;
    private m;
    private v;
    private stepCount;
    /**
     * Constructs an AdamW optimizer.
     * @param trainables - Array of Value parameters to optimize.
     * @param opts - Optional parameters (learningRate, weightDecay, gradientClip, beta1, beta2, epsilon).
     */
    constructor(trainables: Value[], opts?: AdamOptions);
    /**
     * Performs a parameter update using AdamW optimization (decoupled weight decay).
     * @public
     */
    step(): void;
    /**
     * Resets optimizer state (momentum and velocity) for a specific trainable.
     * @param trainable - The Value parameter to reset state for.
     * @public
     */
    resetStateFor(trainable: Value): void;
}
import { Value } from "./Value";
/**
 * Collection of loss functions for training neural networks and optimization.
 * All methods return a scalar Value representing the loss.
 * @public
 */
export declare class Losses {
    /**
     * Computes mean squared error (MSE) loss between outputs and targets.
     * @param outputs Array of Value predictions.
     * @param targets Array of Value targets.
     * @returns Mean squared error as a Value.
     */
    static mse(outputs: Value[], targets: Value[]): Value;
    /**
     * Computes mean absolute error (MAE) loss between outputs and targets.
     * @param outputs Array of Value predictions.
     * @param targets Array of Value targets.
     * @returns Mean absolute error as a Value.
     */
    static mae(outputs: Value[], targets: Value[]): Value;
    /**
     * Small epsilon value for numerical stability in logarithmic computations.
     * @public
     */
    static EPS: number;
    /**
     * Computes binary cross-entropy loss between predicted outputs and targets (after sigmoid).
     * @param outputs Array of Value predictions (expected in (0,1)).
     * @param targets Array of Value targets (typically 0 or 1).
     * @returns Binary cross-entropy loss as a Value.
     */
    static binaryCrossEntropy(outputs: Value[], targets: Value[]): Value;
    /**
     * Computes categorical cross-entropy loss between outputs (logits) and integer target classes.
     * @param outputs Array of Value logits for each class.
     * @param targets Array of integer class indices (0-based, one per sample).
     * @returns Categorical cross-entropy loss as a Value.
     */
    static categoricalCrossEntropy(outputs: Value[], targets: number[]): Value;
    /**
     * Computes Huber loss between outputs and targets.
     * Combines quadratic loss for small residuals and linear loss for large residuals.
     * @param outputs Array of Value predictions.
     * @param targets Array of Value targets.
     * @param delta Threshold at which to switch from quadratic to linear (default: 1.0).
     * @returns Huber loss as a Value.
     */
    static huber(outputs: Value[], targets: Value[], delta?: number): Value;
    /**
     * Computes Tukey loss between outputs and targets.
     * This robust loss function saturates for large residuals.
     *
     * @param outputs Array of Value predictions.
     * @param targets Array of Value targets.
     * @param c Threshold constant (typically 4.685).
     * @returns Tukey loss as a Value.
     */
    static tukey(outputs: Value[], targets: Value[], c?: number): Value;
}
