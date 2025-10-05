import { Value } from "./Value";
import { CompiledFunctions } from "./CompiledFunctions";

/**
 * Pre-compiled residual functions for efficient repeated optimization.
 *
 * Wrapper around CompiledFunctions that provides a simpler API for
 * Levenberg-Marquardt optimization use cases.
 *
 * Compile once, reuse many times - ideal for IK, animation, or any scenario
 * where you solve the same structure with different parameter values.
 *
 * @example
 * ```typescript
 * // Compile once
 * const compiled = CompiledResiduals.compile(params, residualFn);
 *
 * // Solve many times with different initial values
 * for (let i = 0; i < 100; i++) {
 *   params.forEach(p => p.data = randomInitialValue());
 *   V.nonlinearLeastSquares(params, compiled);
 * }
 * ```
 */
export class CompiledResiduals {
  private compiledFunctions: CompiledFunctions;

  private constructor(compiledFunctions: CompiledFunctions) {
    this.compiledFunctions = compiledFunctions;
  }

  /**
   * Compile residual functions for reuse.
   *
   * @param params - Parameter Values (must have .paramName set)
   * @param residualFn - Function that builds residuals from params
   * @returns Compiled residual functions ready for optimization
   */
  static compile(
    params: Value[],
    residualFn: (params: Value[]) => Value[]
  ): CompiledResiduals {
    const compiledFunctions = CompiledFunctions.compile(params, residualFn);
    return new CompiledResiduals(compiledFunctions);
  }

  /**
   * Evaluate compiled residuals and Jacobian.
   *
   * @param params - Current parameter values
   * @returns Residuals, Jacobian, and cost
   */
  evaluate(params: Value[]): { residuals: number[]; J: number[][]; cost: number } {
    const { values, jacobian } = this.compiledFunctions.evaluateJacobian(params);

    const residuals = values;
    const J = jacobian;
    let cost = 0;
    for (const r of residuals) {
      cost += r * r;
    }

    return { residuals, J, cost };
  }

  /** Get number of residuals */
  get numResiduals(): number {
    return this.compiledFunctions.numFunctions;
  }
}
