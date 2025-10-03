import { Value } from './Value';
import { ValueRegistry } from './ValueRegistry';
import { GraphSignature, canonicalizeGraph } from './GraphSignature';
import { compileIndirectKernel, extractInputIndices } from './compileIndirectKernel';

/**
 * Descriptor for a compiled kernel with its signature and metadata.
 * @internal
 */
export interface KernelDescriptor {
  /** Graph signature for matching */
  signature: GraphSignature;

  /** Compiled kernel function */
  kernel: (allValues: number[], indices: number[], row: number[]) => number;

  /** Number of parameters this kernel expects in Jacobian row */
  numParams: number;
}

/**
 * Pool of compiled kernels, indexed by graph signature.
 * Enables kernel reuse across topologically identical graphs.
 * @internal
 */
export class KernelPool {
  /** @internal */
  public kernels = new Map<string, KernelDescriptor>();

  /**
   * Get or compile a kernel for the given residual graph.
   * If a kernel with identical topology exists, reuses it.
   * Otherwise, compiles a new kernel.
   *
   * Also ensures all leaf nodes in the graph are registered.
   *
   * @param residual - Output Value of residual computation
   * @param params - Parameter Values for Jacobian computation
   * @param registry - ValueRegistry for this compilation
   * @returns Kernel descriptor with compiled function
   */
  getOrCompile(
    residual: Value,
    params: Value[],
    registry: ValueRegistry
  ): KernelDescriptor {
    const signature = canonicalizeGraph(residual);

    // Ensure all leaf nodes in this graph are registered
    // (Even if kernel exists, this graph might have new constants)
    const visited = new Set<Value>();
    function registerLeaves(node: Value) {
      if (visited.has(node)) return;
      visited.add(node);

      const prev = (node as any).prev as Value[];
      if (prev.length === 0) {
        registry.register(node);
      } else {
        for (const child of prev) {
          registerLeaves(child);
        }
      }
    }
    registerLeaves(residual);

    // Check if we already have this kernel
    if (this.kernels.has(signature.hash)) {
      return this.kernels.get(signature.hash)!;
    }

    // Compile new kernel
    const kernel = compileIndirectKernel(residual, params, registry);

    const descriptor: KernelDescriptor = {
      signature,
      kernel,
      numParams: params.length
    };

    this.kernels.set(signature.hash, descriptor);
    return descriptor;
  }

  /**
   * Total number of unique kernels compiled
   */
  get size(): number {
    return this.kernels.size;
  }

  /**
   * Get all kernel signatures (for debugging)
   */
  getSignatures(): GraphSignature[] {
    return Array.from(this.kernels.values()).map(d => d.signature);
  }
}
