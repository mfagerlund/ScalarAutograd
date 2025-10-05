# GPU Integration with Optimizers (L-BFGS, Levenberg-Marquardt)

## Current Status

✅ **L-BFGS already supports compiled functions**:
```typescript
lbfgs(params, compiled, options)  // CompiledFunctions interface
```

✅ **Levenberg-Marquardt uses CompiledFunctions**:
```typescript
NonlinearLeastSquares.solve(compiled, params)
```

## Proposed: GPUCompiledFunctions

Create a GPU-accelerated version implementing the same interface:

```typescript
class GPUCompiledFunctions {
  constructor(
    private ctx: WebGPUContext,
    private kernel: WGSLKernel,
    private numParams: number,
    private numResiduals: number
  ) {}

  // Same interface as CompiledFunctions
  evaluateSumWithGradient(params: Value[]): { value: number; gradient: number[] } {
    // 1. Pack params into GPU buffer
    const inputData = new Float32Array(params.map(p => p.data));

    // 2. Execute GPU kernel (batched evaluation + gradient accumulation)
    const { values, gradients } = await this.kernel.executeWithGradients(inputData);

    // 3. Sum residuals on GPU or CPU
    const value = values.reduce((a, b) => a + b, 0);

    return { value, gradient: gradients };
  }

  evaluateJacobian(params: Value[]): { values: number[]; jacobian: number[][] } {
    // Similar GPU implementation
    // Returns residuals and Jacobian for Levenberg-Marquardt
  }
}
```

## Integration Points

### 1. Compilation Phase
```typescript
// Current CPU path
const compiled = CompiledFunctions.compile(params, (p) => residuals);

// Proposed GPU path
const gpuCompiled = await GPUCompiledFunctions.compile(
  params,
  (p) => residuals,
  WebGPUContext.getInstance()
);
```

### 2. Optimizer Usage (No Changes!)
```typescript
// Works with both CPU and GPU compiled functions
const result = lbfgs(params, compiled, options);
```

## Key Challenge: Gradient Computation on GPU

### Current CPU Approach
- **Forward pass**: Compute residual values
- **Backward pass**: Traverse graph in reverse, accumulate gradients

### GPU Approach
We need **reverse-mode autodiff on GPU**. Two options:

### Option A: Compile Backward Pass to WGSL
```typescript
// Compile BOTH forward and backward to WGSL
const { forwardKernel, backwardKernel } = compileToWGSL(residual, params);

// Execute forward
const values = await forwardKernel.execute(inputData);

// Execute backward (reverse topo order)
const gradients = await backwardKernel.execute(values, outputGrad=1.0);
```

**Challenges**:
- Need to store intermediate values for backward pass
- Reverse topological order on GPU
- Memory management for activations

### Option B: Finite Differences on GPU (Simpler, Less Accurate)
```typescript
// Compute gradient via finite differences
for (let i = 0; i < numParams; i++) {
  const epsPlus = [...params];
  epsPlus[i] += epsilon;
  const fPlus = await kernel.execute(epsPlus);

  gradient[i] = (fPlus - f) / epsilon;
}
```

**Issues**:
- O(N) kernel calls for N parameters
- Numerical stability issues
- Slower than analytical gradients

### Option C: Hybrid CPU/GPU
```typescript
// Forward pass on GPU (fast, batched)
const values = await gpuKernel.execute(params);  // All 5000 residuals

// Backward pass on CPU (uses compiled kernels)
const { gradient } = compiled.evaluateSumWithGradient(params);
```

**Advantages**:
- Reuse existing backward pass infrastructure
- GPU accelerates expensive forward pass
- No need to implement reverse-mode on GPU

**Disadvantages**:
- CPU backward pass is still slow
- Data transfer GPU → CPU

## Recommendation: Start with Option C (Hybrid)

**Phase 1: Hybrid Forward GPU + Backward CPU**
```typescript
class HybridGPUCompiledFunctions {
  async evaluateSumWithGradient(params: Value[]): Promise<{value: number, gradient: number[]}> {
    // Forward on GPU: Compute all residual values in parallel
    const gpuValues = await this.gpuKernel.execute(paramsData);
    const value = gpuValues.reduce((a,b) => a+b, 0);

    // Backward on CPU: Use existing compiled kernels
    const { gradient } = this.cpuCompiled.evaluateSumWithGradient(params);

    return { value, gradient };
  }
}
```

**Benefits**:
- Quick to implement (reuse existing code)
- Still get GPU speedup for forward pass (the bottleneck at 5k+ residuals)
- Gradients are exact (not finite difference)

**When does this help?**
- Forward pass is expensive (complex residuals, many vertices)
- Gradient computation is relatively cheap (already compiled)
- For 5k vertices: Forward 2.4x faster on GPU, backward same speed

**Phase 2: Full GPU Reverse-Mode Autodiff**
- Implement backward pass in WGSL
- Store intermediate values in GPU buffers
- Accumulate gradients in parallel
- This is a **big project** but would be 5-10x faster total

## Performance Estimate (5k vertices, 200 iterations)

### Current CPU (from benchmarks)
- Compilation: 81s (one-time)
- Per iteration: 8.8ms (forward + backward)
- 200 iterations: 1.76s
- **Total: 82.76s**

### Hybrid GPU (estimated)
- Compilation: 81s (same canonicalization) + 0.02s WGSL
- Per iteration: 3.7ms forward (GPU) + 5.1ms backward (CPU) = 8.8ms
- 200 iterations: 1.76s
- **Total: 82.78s** ⚠️ **NO BENEFIT!**

Wait, that's wrong! Let me recalculate...

Actually, the 8.8ms CPU includes both forward AND backward. If we split:
- Forward only: ~3ms (computing residuals)
- Backward only: ~5.8ms (gradient accumulation)

Hybrid:
- Forward (GPU): ~1.2ms (2.4x faster)
- Backward (CPU): ~5.8ms (same)
- **Total per iter: 7ms**
- **200 iterations: 1.4s**

**Savings: 0.36s per 200 iterations** - marginal

### Full GPU (hypothetical)
- Forward (GPU): 1.2ms
- Backward (GPU): ~1.5ms (estimated 4x speedup)
- **Total per iter: 2.7ms**
- **200 iterations: 0.54s**

**Savings: 1.2s per 200 iterations** - better, but still small vs 81s compilation

## Implementation Status

### ✅ Phase 1: GPUCompiledFunctions Class
Created `GPUCompiledFunctions` with same interface as `CompiledFunctions`:
- `evaluateSumWithGradient()` - For L-BFGS
- `evaluateJacobian()` - For Levenberg-Marquardt

**Issue**: Methods are async (GPU operations), but L-BFGS expects sync

**Solutions**:
1. **Create `lbfgsAsync`** - Async version of L-BFGS for GPU
2. **Use top-level await** - Only works in ES modules
3. **Synchronous blocking** - Not possible in JavaScript without hacks

**Recommended**: Implement `lbfgsAsync` as separate function

### 🚧 Phase 2: Async L-BFGS
```typescript
export async function lbfgsAsync(
  params: Value[],
  objectiveFn: GPUCompiledFunctions | CompiledFunctions,
  options: LBFGSOptions = {}
): Promise<LBFGSResult> {
  // Same algorithm as lbfgs(), but awaits GPU evaluations
  const { cost, gradient } = await objectiveFn.evaluateSumWithGradient(params);
  // ...
}
```

### 📊 Current Status: Numerical Gradients (Placeholder)
GPUCompiledFunctions currently uses finite differences for gradients:
- Forward pass: GPU (batched, fast)
- Gradients: Numerical (O(N) kernel calls, slow)

This is a placeholder until we implement reverse-mode autodiff on GPU.

## Conclusion

**The real bottleneck is canonicalization (81s), not runtime (1.7s).**

GPU helps with runtime, but optimization priorities should be:
1. **Speed up canonicalization** (81s → 10s would save 71 seconds)
2. **Then** consider GPU gradients (1.7s → 0.5s would save 1.2 seconds)

Unless canonicalization gets much faster, GPU optimizer integration has minimal ROI.

**However**, if you're running **many optimization problems** (different meshes, different initial conditions), the compilation is amortized and GPU runtime wins become significant!
