/**
 * GPU-accelerated compiled functions for optimization
 *
 * Drop-in replacement for CompiledFunctions that uses WebGPU for gradient computation.
 * Compatible with L-BFGS, Levenberg-Marquardt, and other optimizers.
 */

import { Value } from '../Value';
import { WebGPUContext } from './WebGPUContext';
import { compileToWGSL, WGSLKernel } from './compileToWGSL';

/**
 * GPU-accelerated compiled functions.
 *
 * Implements the same interface as CompiledFunctions for drop-in compatibility
 * with L-BFGS and Levenberg-Marquardt optimizers.
 *
 * @example
 * ```typescript
 * const ctx = WebGPUContext.getInstance();
 * await ctx.initialize();
 *
 * const gpuCompiled = await GPUCompiledFunctions.compile(params, (p) => residuals);
 * const result = lbfgs(params, gpuCompiled, options);  // Works with L-BFGS!
 * ```
 */
export class GPUCompiledFunctions {
  private ctx: WebGPUContext;
  private forwardKernel: WGSLKernel;
  private gradientKernels: WGSLKernel[];  // One per residual for now
  private graphInputs: Value[];
  private numParams: number;
  private numResiduals: number;
  private residuals: Value[];
  private params: Value[];

  private constructor(
    ctx: WebGPUContext,
    forwardKernel: WGSLKernel,
    gradientKernels: WGSLKernel[],
    graphInputs: Value[],
    residuals: Value[],
    params: Value[]
  ) {
    this.ctx = ctx;
    this.forwardKernel = forwardKernel;
    this.gradientKernels = gradientKernels;
    this.graphInputs = graphInputs;
    this.numParams = params.length;
    this.numResiduals = residuals.length;
    this.residuals = residuals;
    this.params = params;
  }

  /**
   * Compile residuals for GPU execution.
   *
   * @param params - Parameter Values
   * @param functionsFn - Function that builds residual Values from params
   * @returns GPU-compiled functions ready for optimization
   */
  static async compile(
    params: Value[],
    functionsFn: (params: Value[]) => Value[]
  ): Promise<GPUCompiledFunctions> {
    const ctx = WebGPUContext.getInstance();
    if (!ctx.device) {
      throw new Error('WebGPU not initialized. Call WebGPUContext.getInstance().initialize() first.');
    }

    console.log(`[GPUCompiledFunctions] Compiling ${params.length} params...`);

    // Build residuals
    const residuals = functionsFn(params);
    console.log(`[GPUCompiledFunctions] Built ${residuals.length} residuals`);

    // Check if all residuals have the same structure (common case)
    const { wgslCode: forwardWGSL, graphInputs } = compileToWGSL(residuals[0]);
    const forwardKernel = new WGSLKernel(ctx.device, forwardWGSL, graphInputs);

    // For now, we'll compile gradient kernels separately
    // TODO: Optimize by detecting identical structures
    console.log(`[GPUCompiledFunctions] Compiling gradient kernels...`);
    const gradientKernels: WGSLKernel[] = [];

    for (let i = 0; i < residuals.length; i++) {
      if (i % 100 === 0) {
        console.log(`[GPUCompiledFunctions] Processing gradient ${i}/${residuals.length}`);
      }

      // For now, create a simple gradient kernel
      // This is a placeholder - we'll implement proper reverse-mode autodiff next
      const { wgslCode, graphInputs: gradInputs } = compileToWGSL(residuals[i]);
      gradientKernels.push(new WGSLKernel(ctx.device, wgslCode, gradInputs));
    }

    console.log(`[GPUCompiledFunctions] Compilation complete`);

    return new GPUCompiledFunctions(
      ctx,
      forwardKernel,
      gradientKernels,
      graphInputs,
      residuals,
      params
    );
  }

  /**
   * Evaluate sum of all residuals with accumulated gradient.
   * Compatible with L-BFGS optimizer.
   *
   * @param params - Current parameter values
   * @returns Sum of residuals and accumulated gradient
   */
  async evaluateSumWithGradient(params: Value[]): Promise<{ value: number; gradient: number[] }> {
    // Pack parameters into GPU buffer format
    const batchSize = this.numResiduals;
    const inputsPerResidual = this.graphInputs.length;
    const batchInputs = new Float32Array(batchSize * inputsPerResidual);

    // Pack data for each residual
    for (let i = 0; i < batchSize; i++) {
      // Map params to graph inputs order
      for (let j = 0; j < inputsPerResidual; j++) {
        const graphInput = this.graphInputs[j];
        const paramIdx = this.params.indexOf(graphInput);

        if (paramIdx !== -1) {
          batchInputs[i * inputsPerResidual + j] = params[paramIdx].data;
        } else {
          // Constant value
          batchInputs[i * inputsPerResidual + j] = graphInput.data;
        }
      }
    }

    // Execute forward pass on GPU (all residuals in parallel)
    const forwardValues = await this.forwardKernel.execute(batchInputs, batchSize);

    // Sum residuals
    let totalValue = 0;
    for (let i = 0; i < forwardValues.length; i++) {
      totalValue += forwardValues[i];
    }

    // TODO: Compute gradients on GPU
    // For now, fall back to CPU gradient computation
    const gradient = new Array(this.numParams).fill(0);

    // Numerical gradients as placeholder (TEMPORARY - will be replaced with analytical GPU gradients)
    const epsilon = 1e-7;
    for (let i = 0; i < this.numParams; i++) {
      const originalValue = params[i].data;

      // Perturb parameter
      params[i].data = originalValue + epsilon;

      // Pack perturbed inputs
      for (let r = 0; r < batchSize; r++) {
        for (let j = 0; j < inputsPerResidual; j++) {
          const graphInput = this.graphInputs[j];
          const paramIdx = this.params.indexOf(graphInput);

          if (paramIdx !== -1) {
            batchInputs[r * inputsPerResidual + j] = params[paramIdx].data;
          }
        }
      }

      const perturbedValues = await this.forwardKernel.execute(batchInputs, batchSize);
      let perturbedSum = 0;
      for (let r = 0; r < perturbedValues.length; r++) {
        perturbedSum += perturbedValues[r];
      }

      gradient[i] = (perturbedSum - totalValue) / epsilon;

      // Restore original value
      params[i].data = originalValue;
    }

    return { value: totalValue, gradient };
  }

  /**
   * Evaluate all residuals and their Jacobian matrix.
   * Compatible with Levenberg-Marquardt optimizer.
   *
   * @param params - Current parameter values
   * @returns Residual values and Jacobian matrix
   */
  async evaluateJacobian(params: Value[]): Promise<{ values: number[]; jacobian: number[][] }> {
    // Pack parameters
    const batchSize = this.numResiduals;
    const inputsPerResidual = this.graphInputs.length;
    const batchInputs = new Float32Array(batchSize * inputsPerResidual);

    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < inputsPerResidual; j++) {
        const graphInput = this.graphInputs[j];
        const paramIdx = this.params.indexOf(graphInput);

        if (paramIdx !== -1) {
          batchInputs[i * inputsPerResidual + j] = params[paramIdx].data;
        } else {
          batchInputs[i * inputsPerResidual + j] = graphInput.data;
        }
      }
    }

    // Execute forward pass
    const values = await this.forwardKernel.execute(batchInputs, batchSize);

    // TODO: Compute Jacobian on GPU
    // For now, use numerical differentiation
    const jacobian: number[][] = Array(this.numResiduals)
      .fill(0)
      .map(() => new Array(this.numParams).fill(0));

    const epsilon = 1e-7;
    for (let paramIdx = 0; paramIdx < this.numParams; paramIdx++) {
      const originalValue = params[paramIdx].data;
      params[paramIdx].data = originalValue + epsilon;

      // Repack with perturbed value
      for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < inputsPerResidual; j++) {
          const graphInput = this.graphInputs[j];
          const pIdx = this.params.indexOf(graphInput);

          if (pIdx !== -1) {
            batchInputs[i * inputsPerResidual + j] = params[pIdx].data;
          }
        }
      }

      const perturbedValues = await this.forwardKernel.execute(batchInputs, batchSize);

      for (let i = 0; i < this.numResiduals; i++) {
        jacobian[i][paramIdx] = (perturbedValues[i] - values[i]) / epsilon;
      }

      params[paramIdx].data = originalValue;
    }

    return { values: Array.from(values), jacobian };
  }

  /** Get number of compiled residuals */
  get numFunctions(): number {
    return this.numResiduals;
  }

  /** @deprecated Use numFunctions */
  get kernelCount(): number {
    return 1; // Single forward kernel for now
  }

  /** @deprecated Use numFunctions */
  get kernelReuseFactor(): number {
    return this.numResiduals;
  }
}
