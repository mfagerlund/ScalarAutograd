# WebGPU Kernel Batching - Implementation Plan

## Vision

Compile Value computation graphs to WebGPU compute shaders, enabling parallel execution of all residuals sharing the same topology. This combines existing kernel reuse infrastructure with GPU parallelism for massive speedups on mesh optimization problems.

## Core Concept

```
CPU (Current):           GPU (Proposed):
┌─────────────┐         ┌──────────────────────────────┐
│ for residual│         │ Launch 1000 threads         │
│   kernel()  │  →      │ Each: kernel(threadId)      │
│ 1000 times  │         │ All run simultaneously      │
└─────────────┘         └──────────────────────────────┘
```

---

## Milestone 1: WebGPU Foundation
**Goal**: Establish basic WebGPU infrastructure and prove we can execute simple compute shaders.

### Tasks
- [ ] Research WebGPU API in Node.js (`@webgpu/types`, `@tensorflow/tfjs-backend-webgpu`)
- [ ] Create `src/gpu/WebGPUContext.ts` - singleton for device/queue management
- [ ] Write "Hello GPU" test: compute shader that squares an array of numbers
- [ ] Establish error handling and browser/Node.js compatibility strategy
- [ ] Document WebGPU setup requirements (Chrome flags, Node.js version, etc.)

### Success Criteria
- Can initialize WebGPU device in test environment
- Can execute trivial compute shader and read results back
- Clear documentation of platform requirements

---

## Milestone 2: Simple Kernel Compilation
**Goal**: Compile a single Value graph to WGSL compute shader.

### Tasks
- [ ] Create `src/gpu/compileToWGSL.ts` - Value graph → WGSL translator
- [ ] Implement WGSL code generation for basic ops (add, mul, div, pow)
- [ ] Handle forward pass only (no gradients yet)
- [ ] Create test: compile `V.add(V.mul(a, 2), b)` and verify output matches CPU
- [ ] Design data layout: how to pack input values into GPU buffers

### Example Target Output
```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let a = inputs[idx * 2 + 0];
  let b = inputs[idx * 2 + 1];
  let v0 = a * 2.0;
  let v1 = v0 + b;
  outputs[idx] = v1;
}
```

### Success Criteria
- Can compile simple arithmetic graphs to valid WGSL
- GPU results match CPU execution (within floating-point tolerance)
- Code is readable and maps clearly to Value graph structure

---

## Milestone 3: Kernel Batching Infrastructure
**Goal**: Execute multiple instances of the same kernel in parallel.

### Tasks
- [ ] Design batch data layout - how to interleave N residuals' inputs
- [ ] Modify WGSL compiler to use `global_invocation_id` for batch indexing
- [ ] Create `GPUBatchExecutor` class - manages buffers and dispatch
- [ ] Implement buffer pooling/reuse to avoid allocations
- [ ] Test: batch 100 instances of same graph, verify all outputs correct

### Data Layout Strategy
```
Option A (SoA - Structure of Arrays):
  inputs = [a0, a1, ..., aN, b0, b1, ..., bN]

Option B (AoS - Array of Structures):
  inputs = [a0, b0, a1, b1, ..., aN, bN]
```

### Success Criteria
- Can execute 1000 instances of same kernel in single GPU dispatch
- Correct indexing - each thread processes its assigned instance
- Performance baseline established (time vs CPU serial execution)

---

## Milestone 4: Gradient Computation on GPU
**Goal**: Generate backward pass in WGSL, compute gradients on GPU.

### Tasks
- [ ] Extend WGSL compiler to emit backward pass code
- [ ] Implement gradient accumulation in WGSL (atomic adds for shared params)
- [ ] Handle the gradient buffer layout - one gradient array per batch
- [ ] Test gradient correctness against CPU implementation
- [ ] Benchmark: gradient computation speedup vs CPU

### WGSL Gradient Challenge
```wgsl
// Problem: Multiple threads may write to same gradient
// Solution: Use atomic operations
atomicAdd(&gradients[param_idx], local_gradient);
```

### Success Criteria
- GPU gradients match CPU gradients (< 1e-10 difference)
- Backward pass executes in parallel across batch
- No race conditions in gradient accumulation

---

## Milestone 5: Integration with Existing Codebase
**Goal**: Make GPU execution a drop-in replacement for CompiledResiduals.

### Tasks
- [ ] Create `CompiledResidualsGPU` class parallel to `CompiledResiduals`
- [ ] Implement automatic batching - group residuals by graph signature
- [ ] Add fallback logic - use CPU for small batches or unsupported ops
- [ ] Integrate with NonlinearLeastSquares optimizer
- [ ] Add API for user control: `enableGPU()`, batch size tuning

### API Design
```typescript
const compiled = CompiledResiduals.compile(params, residualFn, {
  backend: 'gpu',      // 'cpu' | 'gpu' | 'auto'
  batchSize: 1024,     // Max instances per kernel
  fallbackThreshold: 10 // Use CPU if batch < threshold
});
```

### Success Criteria
- Icosphere test runs on GPU with no code changes
- Automatic fallback to CPU for edge cases works correctly
- Clear performance wins on large batch sizes (>100 residuals)

---

## Milestone 6: Extended Operations & Optimization
**Goal**: Support full operation set and optimize performance.

### Tasks
- [ ] Implement WGSL for all operations (trig, activation, comparisons, vectors)
- [ ] Add Vec2/Vec3 WGSL support (cross, dot, normalize)
- [ ] Optimize buffer transfers - minimize CPU↔GPU copying
- [ ] Implement async execution - don't block CPU while GPU works
- [ ] Profile and optimize shader occupancy
- [ ] Add telemetry - track kernel cache hits, batch sizes

### Advanced Operations
- [ ] Conditional operations (where, select)
- [ ] Min/max reductions
- [ ] Normalize with safe rsqrt
- [ ] Cross product (Vec3 specific)

### Success Criteria
- Full operation coverage - all Value ops work on GPU
- Async execution - CPU can prepare next batch while GPU runs
- Measurable performance improvements on real workloads

---

## Milestone 7: Real-World Validation
**Goal**: Prove significant speedups on actual mesh optimization problems.

### Tasks
- [ ] Benchmark icosphere optimization (compare CPU vs GPU)
- [ ] Test on complex mesh (1000+ vertices, multiple energy terms)
- [ ] Profile end-to-end: identify remaining bottlenecks
- [ ] Document performance characteristics (when GPU wins/loses)
- [ ] Create example showcasing GPU acceleration

### Target Benchmarks
- **Small mesh** (12 vertices): GPU likely slower (overhead)
- **Medium mesh** (100 vertices): Break-even point
- **Large mesh** (1000+ vertices): Expect 10-100x speedup

### Success Criteria
- Clear documentation of when to use GPU backend
- Reproducible benchmarks showing speedup on realistic problems
- Example in demos/ showcasing GPU acceleration

---

## Technical Considerations

### WebGPU Limitations
- **Max buffer size**: ~2GB typically
- **Workgroup limits**: 256 threads per workgroup (hardware dependent)
- **Atomic operations**: Limited to i32/u32 (need tricks for f32 gradients)
- **No recursion**: All loops must be statically bounded

### Gradient Accumulation Strategy
```wgsl
// Can't use atomicAdd on floats directly
// Option 1: Convert to int bits, atomic add, convert back
// Option 2: Use separate gradient buffer per thread, reduce after
// Option 3: Lock-free algorithms with compareExchange
```

### Browser vs Node.js
- **Browser**: Native WebGPU support (Chrome 113+)
- **Node.js**: Requires `dawn` bindings or headless Chrome
- Strategy: Abstract backend, support both environments

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| WebGPU not available in environment | High | Graceful fallback to CPU backend |
| Float32 precision insufficient | Medium | Compare against CPU, document limits |
| Overhead dominates for small batches | Medium | Auto-detect batch size threshold |
| Atomic gradient accumulation slow | High | Research lock-free alternatives early |
| Complex ops don't map to WGSL | Medium | Keep CPU fallback for edge cases |

---

## Future Extensions

### Beyond Initial Implementation
- **Multi-GPU support**: Distribute batches across devices
- **WebGL fallback**: Wider compatibility (older browsers)
- **Sparse gradients**: Only compute for requiresGrad=true params
- **Kernel fusion**: Merge multiple graph operations into single shader
- **Memory pooling**: Persistent GPU buffers across optimizer steps

### Research Questions
- Can we JIT-compile shaders and cache them?
- Can graph transformations improve GPU efficiency?
- What's the optimal granularity for parallelism?

---

## Getting Started

### Recommended Approach
1. Create worktree: `git worktree add ../ScalarAutograd-gpu webgpu-dev`
2. Start with Milestone 1 (foundation)
3. Validate each milestone with tests before proceeding
4. Keep main branch CPU-only until GPU backend is proven

### Dependencies to Add
```json
{
  "@webgpu/types": "^0.1.x",
  "gpu.js": "^2.x" // Alternative: higher-level abstraction
}
```

### First Session Goal
- Complete Milestone 1
- Prove we can run *something* on GPU
- Establish testing/benchmarking infrastructure

---

## Success Metrics

**Must Have**
- ✅ GPU backend produces identical results to CPU (within tolerance)
- ✅ Measurable speedup on batches >100 residuals
- ✅ Zero regressions on existing CPU code

**Nice to Have**
- 🎯 10x speedup on 1000-vertex mesh
- 🎯 Async execution pipeline
- 🎯 Support in both Node.js and browser

**Dream Scenario**
- 🚀 100x speedup on large meshes
- 🚀 Real-time mesh optimization in browser demos
- 🚀 GPU backend becomes default for large problems

---

## Questions to Resolve

1. **WebGPU vs WebGL vs gpu.js?** WebGPU is modern but limited availability. WebGL has better compat. gpu.js abstracts both.

2. **Gradient accumulation**: How to handle atomic f32 adds? This is THE critical technical challenge.

3. **Data transfer overhead**: When does GPU dispatch overhead dominate? Need early profiling.

4. **API surface**: Should GPU be explicit opt-in or automatic based on batch size?

---

**Status**: 📋 Planning Phase
**Next Action**: Create worktree, begin Milestone 1
**Owner**: TBD
**Target**: Proof-of-concept in 2-3 weeks
