import { Value } from "./Value";
import { compileResidualJacobian } from "./jit-compile-value";

/**
 * Pre-compiled residual functions for efficient repeated optimization.
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
  private compiledFunctions: ((paramValues: number[], row: number[]) => number)[];
  private numParams: number;

  private constructor(
    compiledFunctions: ((paramValues: number[], row: number[]) => number)[],
    numParams: number
  ) {
    this.compiledFunctions = compiledFunctions;
    this.numParams = numParams;
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
    // Ensure params have names for compilation
    params.forEach((p, i) => {
      if (!p.paramName) {
        p.paramName = `p${i}`;
      }
    });

    // Build computation graph
    const residualValues = residualFn(params);

    // Compile each residual with its row index
    const compiledFunctions = residualValues.map((r, i) =>
      compileResidualJacobian(r, params, i)
    );

    return new CompiledResiduals(compiledFunctions, params.length);
  }

  /**
   * Evaluate compiled residuals and Jacobian.
   *
   * @param params - Current parameter values
   * @returns Residuals, Jacobian, and cost
   */
  evaluate(params: Value[]): { residuals: number[]; J: number[][]; cost: number } {
    const paramValues = params.map(p => p.data);
    const numResiduals = this.compiledFunctions.length;

    const residuals: number[] = new Array(numResiduals);
    const J: number[][] = Array(numResiduals).fill(0).map(() => new Array(this.numParams).fill(0));
    let cost = 0;

    for (let i = 0; i < numResiduals; i++) {
      const value = this.compiledFunctions[i](paramValues, J[i]);
      cost += value * value;
      residuals[i] = value;
    }

    return { residuals, J, cost };
  }

  /** Get number of residuals */
  get numResiduals(): number {
    return this.compiledFunctions.length;
  }
}
