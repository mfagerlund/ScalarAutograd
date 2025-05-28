// Optimizers.ts

import { Value } from "./Value";

/**
 * Abstract base class for all optimizers.
 * Ensures only requiresGrad parameters are optimized.
 */
export abstract class Optimizer {
  protected trainables: Value[];
  public learningRate: number;

  /**
   * Constructs an Optimizer.
   * @param trainables Array of Value parameters to optimize.
   * @param learningRate Learning rate for updates.
   */
  constructor(trainables: Value[], learningRate: number) {
    this.trainables = trainables.filter(v => v.requiresGrad);
    this.learningRate = learningRate;
  }

  /**
   * Performs a parameter update step.
   */
  abstract step(): void;

  abstract resetStateFor(trainable: Value): void;

  /**
   * Sets grads of all trainables to zero.
   */
  zeroGrad(): void {
    for (const v of this.trainables) v.grad = 0;
  }

  /**
   * Clips global norm of gradients as regularization.
   * @param maxNorm Maximum allowed norm for gradients.
   */
  clipGradients(maxNorm: number): void {
    const totalNorm = Math.sqrt(
      this.trainables.reduce((sum, v) => sum + v.grad * v.grad, 0)
    );
    if (totalNorm > maxNorm) {
      const scale = maxNorm / (totalNorm + 1e-6);
      for (const v of this.trainables) v.grad *= scale;
    }
  }
}

/**
 * Optional arguments for basic optimizers.
 * @property learningRate: Overrides the step size for parameter updates (default varies by optimizer).
 * @property weightDecay: L2 regularization multiplier (default 0). Ignored for plain SGD.
 * @property gradientClip: Maximum absolute value for gradient updates (default 0: no clipping).
 */
export interface OptimizerOptions {
  learningRate?: number;
  weightDecay?: number;
  gradientClip?: number;
}

/**
 * Stochastic Gradient Descent (SGD) optimizer. Accepts weightDecay and gradientClip for API consistency (ignored).
 */
export class SGD extends Optimizer {
  private weightDecay: number;
  private gradientClip: number;
  /**
   * Constructs an SGD optimizer.
   * @param trainables Array of Value parameters to optimize.
   * @param opts Optional parameters (learningRate, weightDecay, gradientClip).
   */
  constructor(trainables: Value[], opts: OptimizerOptions = {}) {
    super(
      trainables,
      opts.learningRate ?? 1e-2
    );
    this.weightDecay = opts.weightDecay ?? 0;
    this.gradientClip = opts.gradientClip ?? 0;
  }
  /**
   * Performs a parameter update using standard SGD.
   */
  step(): void {
    // Intentionally ignoring weightDecay/gradientClip for SGD
    for (const v of this.trainables) {
      v.data -= this.learningRate * v.grad;
    }
  }

   resetStateFor(trainable: Value): void{   
   }
}

/**
 * Adam and AdamW optimizer parameters.
 * Extends OptimizerOptions.
 * @property beta1: Exponential decay rate for 1st moment (default 0.9).
 * @property beta2: Exponential decay rate for 2nd moment (default 0.999).
 * @property epsilon: Numerical stability fudge factor (default 1e-8).
 */
export interface AdamOptions extends OptimizerOptions {
  beta1?: number;
  beta2?: number;
  epsilon?: number;
}

/**
 * Adam optimizer, supports decoupled weight decay and gradient clipping.
 */
export class Adam extends Optimizer {
  private beta1: number;
  private beta2: number;
  private epsilon: number;
  private weightDecay: number;
  private gradientClip: number;
  private m: Map<Value, number> = new Map();
  private v: Map<Value, number> = new Map();
  private stepCount: number = 0;
  /**
   * Constructs an Adam optimizer.
   * @param trainables Array of Value parameters to optimize.
   * @param opts Optional parameters (learningRate, weightDecay, gradientClip, beta1, beta2, epsilon).
   */
  constructor(
    trainables: Value[],
    opts: AdamOptions = {}
  ) {
    super(trainables, opts.learningRate ?? 0.001);
    this.beta1 = opts.beta1 ?? 0.9;
    this.beta2 = opts.beta2 ?? 0.999;
    this.epsilon = opts.epsilon ?? 1e-8;
    this.weightDecay = opts.weightDecay ?? 0;
    this.gradientClip = opts.gradientClip ?? 0;
    for (const v of this.trainables) {
      this.m.set(v, 0);
      this.v.set(v, 0);
    }
  }
  /**
   * Performs a parameter update using Adam optimization.
   */
  step(): void {
    this.stepCount++;
    for (const v of this.trainables) {
      let grad = v.grad;
      if (this.weightDecay > 0) grad += this.weightDecay * v.data;

      let m = this.m.get(v)!;
      let vVal = this.v.get(v)!;
      m = this.beta1 * m + (1 - this.beta1) * grad;
      vVal = this.beta2 * vVal + (1 - this.beta2) * grad * grad;

      const mHat = m / (1 - Math.pow(this.beta1, this.stepCount));
      const vHat = vVal / (1 - Math.pow(this.beta2, this.stepCount));
      let update = mHat / (Math.sqrt(vHat) + this.epsilon);

      if (this.gradientClip > 0) {
        update = Math.max(-this.gradientClip, Math.min(update, this.gradientClip));
      }
      v.data -= this.learningRate * update;

      this.m.set(v, m);
      this.v.set(v, vVal);
    }
  }

  resetStateFor(trainable: Value): void{   
    this.m.set(trainable, 0);
    this.v.set(trainable, 0);
  }
}

/**
 * AdamW optimizer, supports decoupled weight decay and gradient clipping (same options as Adam).
 */
export class AdamW extends Optimizer {
  private beta1: number;
  private beta2: number;
  private epsilon: number;
  private weightDecay: number;
  private gradientClip: number;
  private m: Map<Value, number> = new Map();
  private v: Map<Value, number> = new Map();
  private stepCount: number = 0;
  /**
   * Constructs an AdamW optimizer.
   * @param trainables Array of Value parameters to optimize.
   * @param opts Optional parameters (learningRate, weightDecay, gradientClip, beta1, beta2, epsilon).
   */
  constructor(
    trainables: Value[],
    opts: AdamOptions = {}
  ) {
    super(trainables, opts.learningRate ?? 0.001);
    this.beta1 = opts.beta1 ?? 0.9;
    this.beta2 = opts.beta2 ?? 0.999;
    this.epsilon = opts.epsilon ?? 1e-8;
    this.weightDecay = opts.weightDecay ?? 0.01;
    this.gradientClip = opts.gradientClip ?? 0;
    for (const v of this.trainables) {
      this.m.set(v, 0);
      this.v.set(v, 0);
    }
  }
  /**
   * Performs a parameter update using AdamW optimization (decoupled weight decay).
   */
  step(): void {
    this.stepCount++;
    for (const v of this.trainables) {
      let grad = v.grad;
      let m = this.m.get(v)!;
      let vVal = this.v.get(v)!;
      m = this.beta1 * m + (1 - this.beta1) * grad;
      vVal = this.beta2 * vVal + (1 - this.beta2) * grad * grad;

      const mHat = m / (1 - Math.pow(this.beta1, this.stepCount));
      const vHat = vVal / (1 - Math.pow(this.beta2, this.stepCount));
      let update = mHat / (Math.sqrt(vHat) + this.epsilon);
      if (this.gradientClip > 0) {
        update = Math.max(-this.gradientClip, Math.min(update, this.gradientClip));
      }
      // Weight decay is decoupled as in AdamW paper:
      v.data -= this.learningRate * update + this.learningRate * this.weightDecay * v.data;
      this.m.set(v, m);
      this.v.set(v, vVal);
    }
  }

  resetStateFor(trainable: Value): void{   
    this.m.set(trainable, 0);
    this.v.set(trainable, 0);
  }
}