import { Value } from './Value';
import { ValueRegistry } from './ValueRegistry';
import { canonicalizeGraphHash } from './GraphHashCanonicalizer';
import { canonicalizeGraphNoSort } from './GraphCanonicalizerNoSort';
import { compileIndirectKernel, extractInputIndices } from './compileIndirectKernel';

/**
 * Descriptor for a compiled kernel with its signature and metadata.
 * @internal
 */
export interface KernelDescriptor {
  /** Canonical string signature for matching */
  canonicalString: string;

  /** Compiled kernel function */
  kernel: (allValues: number[], indices: number[], gradientIndices: number[], gradient: number[]) => number;

  /** Number of graph inputs (for validating indices arrays) */
  numInputs: number;
}

/**
 * Pool of compiled kernels, indexed by graph signature.
 * Enables kernel reuse across topologically identical graphs.
 * @internal
 */
export class KernelPool {
  /** @internal */
  public kernels = new Map<string, KernelDescriptor>();

  /** Cache canonical strings by output Value (weak references to avoid memory leaks during compilation) */
  private valueCanonCache = new WeakMap<Value, string>();

  /** Canonicalization mode: 'no-sort' | 'hash' */
  public canonMode: 'no-sort' | 'hash' = 'hash';

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
    // Find which params are actually used by this residual
    const usedParams: Value[] = [];
    const paramSet = new Set(params);
    const visited = new Set<Value>();

    function findUsedParams(node: Value) {
      if (visited.has(node)) return;
      visited.add(node);

      if (paramSet.has(node)) {
        usedParams.push(node);
      }

      const prev = (node as any).prev as Value[];
      for (const child of prev) {
        findUsedParams(child);
      }
    }

    findUsedParams(residual);

    // Sort by original param order to ensure deterministic canonicalization
    const paramIndexMap = new Map(params.map((p, i) => [p, i]));
    usedParams.sort((a, b) => paramIndexMap.get(a)! - paramIndexMap.get(b)!);

    // Check cache first (avoids re-canonicalizing same Value object)
    let canon = this.valueCanonCache.get(residual);
    let canonTime = 0;

    if (!canon) {
      // Canonicalize structure (param usage will be encoded in gradientIndices at runtime)
      const canonStart = performance.now();
      const result =
        this.canonMode === 'hash' ? canonicalizeGraphHash(residual, usedParams) :
        canonicalizeGraphNoSort(residual, usedParams);
      canon = result.canon;
      canonTime = performance.now() - canonStart;

      // Cache for future lookups
      this.valueCanonCache.set(residual, canon);

      if (canonTime > 10) {
        console.log(`[KernelPool] Canonicalization took ${canonTime.toFixed(0)}ms for graph with ${usedParams.length} params (mode: ${this.canonMode})`);
      }
    }

    // Ensure all leaf nodes in this graph are registered
    // (Even if kernel exists, this graph might have new constants)
    const visitedReg = new Set<Value>();
    function registerLeaves(node: Value) {
      if (visitedReg.has(node)) return;
      visitedReg.add(node);

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
    if (this.kernels.has(canon)) {
      if (canonTime > 10) {
        console.log(`[KernelPool] Cache HIT - reusing existing kernel (pool size: ${this.kernels.size})`);
        console.log(`  Canon: ${canon.substring(0, 300)}${canon.length > 300 ? '...' : ''}`);
      }
      return this.kernels.get(canon)!;
    }

    // Compile new kernel - it operates on local input indices
    const kernel = compileIndirectKernel(residual, params, registry);

    // Count number of inputs this kernel expects
    const numInputs = extractInputIndices(residual, registry).length;

    const descriptor: KernelDescriptor = {
      canonicalString: canon,
      kernel,
      numInputs
    };

    this.kernels.set(canon, descriptor);
    return descriptor;
  }

  /**
   * Set canonicalization mode for kernel matching.
   * - 'no-sort': ID-based (fast, exact graph matching)
   * - 'hash': Hash-based (fastest for very large graphs, uses 64-bit FNV-1a)
   * @param mode - Canonicalization mode
   */
  setCanonMode(mode: 'no-sort' | 'hash'): void {
    this.canonMode = mode;
  }

  /**
   * Total number of unique kernels compiled
   */
  get size(): number {
    return this.kernels.size;
  }

  /**
   * Get all canonical strings (for debugging)
   */
  getCanonicalStrings(): string[] {
    return Array.from(this.kernels.keys());
  }
}
