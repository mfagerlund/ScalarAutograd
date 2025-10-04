import { Value } from "./Value";
import { ValueRegistry } from "./ValueRegistry";
import { KernelPool } from "./KernelPool";
import { extractInputIndices } from "./compileIndirectKernel";

/**
 * Descriptor for a single function: which kernel to use and what indices.
 * @internal
 */
interface FunctionDescriptor {
  kernelHash: string;
  inputIndices: number[];      // Maps kernel inputs → registry IDs
  gradientIndices: number[];   // Maps kernel inputs → gradient array positions (-1 if no grad)
}

/**
 * Pre-compiled scalar functions with automatic differentiation and kernel reuse.
 *
 * Uses kernel reuse: topologically identical functions share the same compiled kernel.
 *
 * Compile once, reuse many times - ideal for optimization, IK, animation, or any scenario
 * where you repeatedly evaluate the same structure with different parameter values.
 *
 * @example
 * ```typescript
 * // Single objective (L-BFGS, Adam, SGD)
 * const compiled = CompiledFunctions.compile(params, (p) => [loss(p)]);
 * const { value, gradient } = compiled.evaluateGradient(params);
 *
 * // Multiple residuals (Levenberg-Marquardt)
 * const compiled = CompiledFunctions.compile(params, (p) => residuals(p));
 * const { values, jacobian } = compiled.evaluateJacobian(params);
 * ```
 */
export class CompiledFunctions {
  private registry: ValueRegistry;
  private kernelPool: KernelPool;
  private functionDescriptors: FunctionDescriptor[];
  private numParams: number;

  private constructor(
    registry: ValueRegistry,
    kernelPool: KernelPool,
    functionDescriptors: FunctionDescriptor[],
    numParams: number
  ) {
    this.registry = registry;
    this.kernelPool = kernelPool;
    this.functionDescriptors = functionDescriptors;
    this.numParams = numParams;
  }

  /**
   * Compile scalar functions for reuse with kernel sharing.
   *
   * @param params - Parameter Values (must have .paramName set)
   * @param functionsFn - Function that builds scalar outputs from params
   * @returns Compiled functions ready for optimization
   */
  static compile(
    params: Value[],
    functionsFn: (params: Value[]) => Value[]
  ): CompiledFunctions {
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

    // Build function graphs
    const functionValues = functionsFn(params);

    // Build param index map for gradient mapping
    const paramIndexMap = new Map(params.map((p, i) => [registry.getId(p), i]));

    // Compile kernels with reuse
    console.log(`[CompiledFunctions] Compiling ${functionValues.length} functions...`);
    const functionDescriptors: FunctionDescriptor[] = functionValues.map((f, idx) => {
      if (idx % 100 === 0) {
        console.log(`[CompiledFunctions] Processing function ${idx}/${functionValues.length}`);
      }
      // Get or compile kernel for this graph structure
      const descriptor = kernelPool.getOrCompile(f, params, registry);

      // Extract input indices for this specific function
      const inputIndices = extractInputIndices(f, registry);

      // Build gradient indices: maps kernel local inputs → global gradient positions
      const gradientIndices = inputIndices.map(regId => {
        // Check if this input is a param (needs gradient)
        if (paramIndexMap.has(regId)) {
          return paramIndexMap.get(regId)!;
        }
        // Not a param (constant) - no gradient
        return -1;
      });

      return {
        kernelHash: descriptor.canonicalString,
        inputIndices,
        gradientIndices
      };
    });

    return new CompiledFunctions(registry, kernelPool, functionDescriptors, params.length);
  }

  /**
   * Compile scalar functions for reuse with kernel sharing (async version).
   * Yields to browser between chunks to prevent UI freezing on large problems.
   *
   * @param params - Parameter Values (must have .paramName set)
   * @param functionsFn - Function that builds scalar outputs from params
   * @param chunkSize - Number of functions to process per chunk (default: 50)
   * @returns Compiled functions ready for optimization
   */
  static async compileAsync(
    params: Value[],
    functionsFn: (params: Value[]) => Value[],
    chunkSize: number = 50
  ): Promise<CompiledFunctions> {
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

    // Build function graphs
    const functionValues = functionsFn(params);

    // Build param index map for gradient mapping
    const paramIndexMap = new Map(params.map((p, i) => [registry.getId(p), i]));

    // Compile kernels with reuse (in chunks to avoid freezing)
    console.log(`[CompiledFunctions] Compiling ${functionValues.length} functions...`);
    const functionDescriptors: FunctionDescriptor[] = [];
    const totalChunks = Math.ceil(functionValues.length / chunkSize);
    let lastLoggedPercent = 0;

    for (let i = 0; i < functionValues.length; i += chunkSize) {
      const chunkEnd = Math.min(i + chunkSize, functionValues.length);

      // Only log every 25% progress to reduce clutter
      const percentComplete = Math.floor((chunkEnd / functionValues.length) * 100);
      if (percentComplete >= lastLoggedPercent + 25) {
        console.log(`[CompiledFunctions] Processing ${chunkEnd}/${functionValues.length} (${percentComplete}%)`);
        lastLoggedPercent = percentComplete;
      }

      for (let j = i; j < chunkEnd; j++) {
        const f = functionValues[j];
        const descriptor = kernelPool.getOrCompile(f, params, registry);
        const inputIndices = extractInputIndices(f, registry);
        const gradientIndices = inputIndices.map(regId =>
          paramIndexMap.has(regId) ? paramIndexMap.get(regId)! : -1
        );

        functionDescriptors.push({
          kernelHash: descriptor.canonicalString,
          inputIndices,
          gradientIndices
        });
      }

      // Yield to browser
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    console.log(`[CompiledFunctions] Complete: ${kernelPool.kernels.size} unique kernels, ${(functionValues.length / kernelPool.kernels.size).toFixed(1)}x reuse`);

    return new CompiledFunctions(registry, kernelPool, functionDescriptors, params.length);
  }

  /**
   * Evaluate gradient of first function (for single objective optimization).
   *
   * @param params - Current parameter values
   * @returns Function value and gradient vector
   */
  evaluateGradient(params: Value[]): { value: number; gradient: number[] } {
    if (this.functionDescriptors.length === 0) {
      throw new Error('No functions compiled');
    }

    // Update registry with current param values
    const allValues = this.registry.getDataArray();
    params.forEach(p => {
      const id = this.registry.getId(p);
      allValues[id] = p.data;
    });

    const gradient = new Array(this.numParams).fill(0);
    const desc = this.functionDescriptors[0];
    const kernelDesc = this.kernelPool.kernels.get(desc.kernelHash)!;
    const value = kernelDesc.kernel(allValues, desc.inputIndices, desc.gradientIndices, gradient);

    return { value, gradient };
  }

  /**
   * Evaluate sum of all functions with accumulated gradient (for L-BFGS, Adam, etc).
   *
   * This is the key method for kernel reuse with gradient-based optimizers.
   * When you have N structurally identical residuals, this will:
   * - Compile ~1 kernel instead of N
   * - Evaluate all N residuals, accumulating their gradients
   * - Return total loss and accumulated gradient
   *
   * @param params - Current parameter values
   * @returns Sum of all function values and accumulated gradient
   */
  evaluateSumWithGradient(params: Value[]): { value: number; gradient: number[] } {
    // Update registry with current param values
    const allValues = this.registry.getDataArray();
    params.forEach(p => {
      const id = this.registry.getId(p);
      allValues[id] = p.data;
    });

    const gradient = new Array(this.numParams).fill(0);
    let totalValue = 0;

    for (const desc of this.functionDescriptors) {
      const kernelDesc = this.kernelPool.kernels.get(desc.kernelHash)!;
      totalValue += kernelDesc.kernel(allValues, desc.inputIndices, desc.gradientIndices, gradient);
    }

    return { value: totalValue, gradient };
  }

  /**
   * Evaluate all functions and their Jacobian matrix.
   *
   * @param params - Current parameter values
   * @returns Function values and Jacobian matrix
   */
  evaluateJacobian(params: Value[]): { values: number[]; jacobian: number[][] } {
    // Update registry with current param values
    const allValues = this.registry.getDataArray();
    params.forEach(p => {
      const id = this.registry.getId(p);
      allValues[id] = p.data;
    });

    const numFunctions = this.functionDescriptors.length;
    const values: number[] = new Array(numFunctions);
    const jacobian: number[][] = Array(numFunctions).fill(0).map(() => new Array(this.numParams).fill(0));

    for (let i = 0; i < numFunctions; i++) {
      const desc = this.functionDescriptors[i];
      const kernelDesc = this.kernelPool.kernels.get(desc.kernelHash)!;
      values[i] = kernelDesc.kernel(allValues, desc.inputIndices, desc.gradientIndices, jacobian[i]);
    }

    return { values, jacobian };
  }

  /**
   * Backward compatibility: evaluate as least-squares residuals.
   *
   * @deprecated Use evaluateJacobian() instead and compute cost yourself
   * @param params - Current parameter values
   * @returns Residuals, Jacobian, and sum of squared residuals
   */
  evaluate(params: Value[]): { residuals: number[]; J: number[][]; cost: number } {
    const { values, jacobian } = this.evaluateJacobian(params);
    const cost = values.reduce((sum, v) => sum + v * v, 0);
    return { residuals: values, J: jacobian, cost };
  }

  /** Get number of compiled functions */
  get numFunctions(): number {
    return this.functionDescriptors.length;
  }

  /** @deprecated Use numFunctions instead */
  get numResiduals(): number {
    return this.numFunctions;
  }

  /** Get number of unique kernels (for metrics) */
  get kernelCount(): number {
    return this.kernelPool.size;
  }

  /** Get kernel reuse factor */
  get kernelReuseFactor(): number {
    return this.numFunctions / this.kernelPool.size;
  }
}

/**
 * @deprecated Use CompiledFunctions instead
 */
export const CompiledResiduals = CompiledFunctions;
