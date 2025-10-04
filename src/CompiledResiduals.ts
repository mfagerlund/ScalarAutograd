import { Value } from "./Value";
import { ValueRegistry } from "./ValueRegistry";
import { KernelPool } from "./KernelPool";
import { extractInputIndices } from "./compileIndirectKernel";

/**
 * Descriptor for a single residual: which kernel to use and what indices.
 * @internal
 */
interface ResidualDescriptor {
  kernelHash: string;
  inputIndices: number[];
}

/**
 * Pre-compiled residual functions for efficient repeated optimization.
 *
 * Uses kernel reuse: topologically identical residuals share the same compiled kernel.
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
  private registry: ValueRegistry;
  private kernelPool: KernelPool;
  private residualDescriptors: ResidualDescriptor[];
  private numParams: number;

  private constructor(
    registry: ValueRegistry,
    kernelPool: KernelPool,
    residualDescriptors: ResidualDescriptor[],
    numParams: number
  ) {
    this.registry = registry;
    this.kernelPool = kernelPool;
    this.residualDescriptors = residualDescriptors;
    this.numParams = numParams;
  }

  /**
   * Compile residual functions for reuse with kernel sharing.
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

    const registry = new ValueRegistry();
    const kernelPool = new KernelPool();

    // Register all params first
    params.forEach(p => registry.register(p));

    // Build residual graphs
    const residualValues = residualFn(params);

    // Compile kernels with reuse
    const residualDescriptors: ResidualDescriptor[] = residualValues.map(r => {
      // Get or compile kernel for this graph structure
      const descriptor = kernelPool.getOrCompile(r, params, registry);

      // Extract input indices for this specific residual
      const inputIndices = extractInputIndices(r, registry);

      return {
        kernelHash: descriptor.canonicalString,
        inputIndices
      };
    });

    return new CompiledResiduals(registry, kernelPool, residualDescriptors, params.length);
  }

  /**
   * Evaluate compiled residuals and Jacobian.
   *
   * @param params - Current parameter values
   * @returns Residuals, Jacobian, and cost
   */
  evaluate(params: Value[]): { residuals: number[]; J: number[][]; cost: number } {
    // Update registry with current param values
    const allValues = this.registry.getDataArray();
    params.forEach(p => {
      const id = this.registry.getId(p);
      allValues[id] = p.data;
    });

    const numResiduals = this.residualDescriptors.length;
    const residuals: number[] = new Array(numResiduals);
    const J: number[][] = Array(numResiduals).fill(0).map(() => new Array(this.numParams).fill(0));
    let cost = 0;

    for (let i = 0; i < numResiduals; i++) {
      const desc = this.residualDescriptors[i];

      // Get kernel for this residual
      const kernelDesc = this.kernelPool.kernels.get(desc.kernelHash)!;

      // Execute kernel with residual's specific indices
      const value = kernelDesc.kernel(allValues, desc.inputIndices, J[i]);
      cost += value * value;
      residuals[i] = value;
    }

    return { residuals, J, cost };
  }

  /** Get number of residuals */
  get numResiduals(): number {
    return this.residualDescriptors.length;
  }

  /** Get number of unique kernels (for metrics) */
  get kernelCount(): number {
    return this.kernelPool.size;
  }

  /** Get kernel reuse factor */
  get kernelReuseFactor(): number {
    return this.numResiduals / this.kernelPool.size;
  }
}
