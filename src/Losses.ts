import { V } from "./V";
import { Value } from "./Value";

/**
 * Throws an error if outputs and targets length do not match.
 * @param outputs Array of output Values.
 * @param targets Array of target Values.
 */
function checkLengthMatch(outputs: Value[], targets: Value[]): void {
  if (outputs.length !== targets.length) {
    throw new Error('Outputs and targets must have the same length');
  }
}

/**
 * Collection of loss functions for training neural networks and optimization.
 * All methods return a scalar Value representing the loss.
 * @public
 */
export class Losses {
  /**
   * Computes mean squared error (MSE) loss between outputs and targets.
   * @param outputs Array of Value predictions.
   * @param targets Array of Value targets.
   * @returns Mean squared error as a Value.
   */
  public static mse(outputs: Value[], targets: Value[]): Value {
    checkLengthMatch(outputs, targets);
    if (!Array.isArray(outputs) || !Array.isArray(targets)) throw new TypeError('mse expects Value[] for both arguments.');
    if (!outputs.length) return new Value(0);
    const diffs = outputs.map((out, i) => out.sub(targets[i]).square());
    return Value.mean(diffs);
  }

  /**
   * Computes mean absolute error (MAE) loss between outputs and targets.
   * @param outputs Array of Value predictions.
   * @param targets Array of Value targets.
   * @returns Mean absolute error as a Value.
   */
  public static mae(outputs: Value[], targets: Value[]): Value {
    checkLengthMatch(outputs, targets);
    if (!Array.isArray(outputs) || !Array.isArray(targets)) throw new TypeError('mae expects Value[] for both arguments.');
    if (!outputs.length) return new Value(0);
    const diffs = outputs.map((out, i) => out.sub(targets[i]).abs());
    return Value.mean(diffs);
  }

  /**
   * Small epsilon value for numerical stability in logarithmic computations.
   * @public
   */
  static EPS = 1e-12;

  /**
   * Computes binary cross-entropy loss between predicted outputs and targets (after sigmoid).
   * @param outputs Array of Value predictions (expected in (0,1)).
   * @param targets Array of Value targets (typically 0 or 1).
   * @returns Binary cross-entropy loss as a Value.
   */
  public static binaryCrossEntropy(outputs: Value[], targets: Value[]): Value {
    checkLengthMatch(outputs, targets);
    if (!Array.isArray(outputs) || !Array.isArray(targets)) throw new TypeError('binaryCrossEntropy expects Value[] for both arguments.');
    if (!outputs.length) return new Value(0);
    const eps = Losses.EPS;
    const one = new Value(1);
    const losses = outputs.map((out, i) => {
      const t = targets[i];
      const outClamped = out.clamp(eps, 1 - eps); // sigmoid should output (0,1)
      return t.mul(outClamped.log()).add(one.sub(t).mul(one.sub(outClamped).log()));
    });
    return Value.mean(losses).mul(-1);
  }

  /**
   * Computes categorical cross-entropy loss between outputs (logits) and integer target classes.
   * @param outputs Array of Value logits for each class.
   * @param targets Array of integer class indices (0-based, one per sample).
   * @returns Categorical cross-entropy loss as a Value.
   */
  public static categoricalCrossEntropy(outputs: Value[], targets: number[]): Value {
    // targets: integer encoded class indices
    if (!Array.isArray(outputs) || !Array.isArray(targets)) throw new TypeError('categoricalCrossEntropy expects Value[] and number[].');
    if (!outputs.length || !targets.length) return new Value(0);
    if (targets.some(t => typeof t !== 'number' || !isFinite(t) || t < 0 || t >= outputs.length || Math.floor(t) !== t)) {
      throw new Error('Target indices must be valid integers in [0, outputs.length)');
    }
    const eps = Losses.EPS;
    const maxLogit = outputs.reduce((a, b) => a.data > b.data ? a : b);
    const exps = outputs.map(out => out.sub(maxLogit).exp());
    const sumExp = Value.sum(exps).add(eps);
    const softmax = exps.map(e => e.div(sumExp));
    const tIndices = targets.map((t, i) => softmax[t]);
    return Value.mean(tIndices.map(sm => sm.add(eps).log().mul(-1)));
  }

  /**
   * Computes Huber loss between outputs and targets.
   * Combines quadratic loss for small residuals and linear loss for large residuals.
   * @param outputs Array of Value predictions.
   * @param targets Array of Value targets.
   * @param delta Threshold at which to switch from quadratic to linear (default: 1.0).
   * @returns Huber loss as a Value.
   */
  public static huber(outputs: Value[], targets: Value[], delta = 1.0): Value {
    checkLengthMatch(outputs, targets);
    if (!Array.isArray(outputs) || !Array.isArray(targets)) throw new TypeError('huber expects Value[] for both arguments.');
    if (!outputs.length) return new Value(0);
    
    const deltaValue = new Value(delta);
    const half = new Value(0.5);
    
    const losses = outputs.map((out, i) => {
      const residual = V.abs(V.sub(out, targets[i]));
      const condition = V.lt(residual, deltaValue);

      const quadraticLoss = V.mul(half, V.square(residual));
      const linearLoss = V.mul(deltaValue, V.sub(residual, V.mul(half, deltaValue)));

      return V.ifThenElse(condition, quadraticLoss, linearLoss);
    });

    return V.mean(losses);
  }

  /**
   * Computes Tukey loss between outputs and targets.
   * This robust loss function saturates for large residuals.
   * 
   * @param outputs Array of Value predictions.
   * @param targets Array of Value targets.
   * @param c Threshold constant (typically 4.685).
   * @returns Tukey loss as a Value.
   */
  public static tukey(outputs: Value[], targets: Value[], c: number = 4.685): Value {
    checkLengthMatch(outputs, targets);
    const c2_over_6 = (c * c) / 6;
    const cValue = V.C(c);
    const c2_over_6_Value = V.C(c2_over_6);

    const losses = outputs.map((out, i) => {
      const diff = V.abs(V.sub(out, targets[i]));
      const inlier = V.lte(diff, cValue);
      const rc = V.div(diff, cValue);
      const rc2 = V.square(rc);
      const oneMinusRC2 = V.sub(1, rc2);
      const inner = V.pow(oneMinusRC2, 3);
      const inlierLoss = V.mul(c2_over_6_Value, V.sub(1, inner));
      const loss = V.ifThenElse(inlier, inlierLoss, c2_over_6_Value);
      return loss;
    });
    return V.mean(losses);
}
}