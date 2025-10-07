# Graph Hasher Refactoring Plan

## Current Naming Issues

The "canonicalization" terminology is misleading. The system performs **graph signature generation** through structural hashing to enable kernel reuse, not true canonicalization (operations aren't even sorted for commutativity).

**Proposed Rename:**
- `GraphHashCanonicalizer.ts` → `GraphHasher.ts`
- `GraphCanonicalizerNoSort.ts` → `GraphHasherNoSort.ts` (or merge as modes)
- `canonicalizeGraphHash()` → `hashGraph()` or `computeGraphHash()`

## The Fundamental Problem: Static vs Dynamic Graphs

### Static Graphs (Current Optimization)

When graphs are **structurally identical** across optimization iterations, we can:
1. Build the graph once
2. Hash it once to get a kernel signature
3. Compile a kernel function
4. Reuse that kernel for all subsequent iterations
5. Pass parameters in a fixed order with index mappings

**Example:** A simple mesh where each vertex has 6 neighbors in consistent order. We might find ~30 unique kernel signatures due to neighbor count/topology differences, but these remain stable throughout optimization.

**Performance:** Very fast. No graph rebuilding, no rehashing, direct kernel execution.

### Dynamic Graphs (Current Problem)

When graph **structure changes** based on runtime conditions (e.g., optimization progress, conditionals, neighbor filtering), the precompiled kernel becomes invalid.

**Example:** `PaperCovarianceEnergyELambda.ts:118`
```typescript
// Skip if normals are nearly parallel (cross product ≈ 0)
if (cross1Mag.data < 1e-12) continue;
```

This conditional means:
- Early in optimization: normals might be nearly parallel → graph has N-2 terms
- Later in optimization: normals diverge → graph has N terms
- The kernel signature changes mid-optimization
- Precompiled kernel expects wrong number of inputs → **WRONG RESULTS**

### The Three Costs of Dynamic Graphs

If we precompile expecting static graphs but the graph changes:

1. **Graph Rebuild Cost:** Must rebuild the entire Value graph each iteration to detect changes
2. **Hash Cost:** Must run the graph hasher (currently slow: 10-100ms for complex graphs)
3. **Kernel Cache Management:** Must maintain dynamic kernel pool with:
   - Kernel signatures
   - Compiled kernel functions
   - Input index mappings (kernel param 4 → values array position 3056)
   - Gradient index mappings

**Performance Impact:** 10-100x slower than static case.

## Potential Solutions

### Option 1: Hash-As-You-Build + Leaf Tracking

Build graph hash incrementally AND track inputs during construction:
- Each `Value.make()` call updates a running hash
- Simultaneously collect leaf Value objects (params + constants) in a Set
- Graph returns: `{ hash, leaves: Value[] }`
- No separate traversal needed!
- At kernel call time, map Value objects to indices in global array

**Key insight:** We don't need to know indices during graph building - we just need the Value objects themselves. Index mapping happens at kernel execution time.

**Challenge:** Value object identity vs semantic equality
- Same constant `V.C(1e-12)` might create different objects
- Solution: Intern constants, or hash by numeric value not object identity

**Benefit:** Eliminates entire second traversal pass
**Performance:** Could reduce hashing from 10ms to <1ms

### Option 2: True JIT Compilation

Instead of precompiling, defer compilation decision:
1. Build graph on first execution
2. Hash and compile kernel
3. On subsequent calls, **quickly detect** if structure might have changed
4. If potentially changed: rehash and select/recompile kernel
5. If definitely unchanged: reuse existing kernel

**Challenge:** "Quickly detect" is hard. Need a fast structural fingerprint that's cheaper than full rehashing.

### Option 3: Static Graph Declaration

Require users to declare graph mutability:
```typescript
// Static graph: compile once, reuse forever
energy.computeStatic(mesh);

// Dynamic graph: rebuild + rehash every call
energy.computeDynamic(mesh);
```

Pros: Explicit control, clear performance characteristics
Cons: Burden on user, easy to get wrong

### Option 4: Hybrid Approach

- Use static compilation by default
- Add runtime validation: check input count matches expected
- On mismatch: fall back to dynamic mode (rebuild + rehash + recompile)
- Warn user about performance impact

### Option 5: WebGPU Considerations

If targeting WebGPU:
- Might execute each graph once for thousands of vertices in parallel
- Graph compilation becomes less critical (batched execution)
- But still need fast graph signature matching to select the right GPU kernel
- The "identify which graph we have" problem remains

## Benchmark Requirements

Before deciding on a solution, we need:

1. **Hash Speed Target:** How fast must hashing be for "true JIT"?
   - Target: <1ms for typical residual graphs
   - Current: 10-100ms (too slow)

2. **Graph Build Cost:** Measure Value graph construction time
   - Is it dominated by hashing or by graph building itself?

3. **Cache Hit Rate:** In practice, how often do graphs actually change?
   - If rare: hybrid validation approach works
   - If frequent: need fundamental rethinking

4. **Incremental Hash Viability:** Can hash-as-you-build achieve <1ms target?

## Implementation Results (2025-01-06)

### GraphBuilder Prototype - COMPLETE ✓

Implemented `GraphBuilder` with hash-as-you-build + leaf tracking:
- Tracks leaf Values during graph construction (no second traversal)
- Computes hash after knowing all leaves (ensures stability)
- Constant interning in `V.C()` for identical Value objects
- Zero overhead when not tracking (`Value.currentBuilder === null`)

**Benchmark Results:**
```
Simple graph (3 params, 4 ops):      1.29x speedup (0.006ms → 0.005ms)
Medium graph (10 params, ~30 ops):   1.19x speedup (0.026ms → 0.022ms)
Complex graph (20 params, ~100 ops): 1.38x speedup (0.094ms → 0.068ms)
Hinge-like graph (8 params, ~50 ops): 1.41x speedup (0.104ms → 0.073ms)

Hash collision rate: 0.00% (1000 diverse graphs)
Signature stability: 100% (stable across rebuilds)
```

**Performance Analysis:**
- Speedup is modest (1.2x-1.4x) because graph build dominates total time
- For 1000-vertex mesh: ~73ms per iteration (vs ~104ms old way)
- Still need ~73 seconds to signature 1000 vertices per optimization iteration
- **Conclusion:** Hash-as-you-build helps but doesn't solve the fundamental cost

### Next Steps

The real optimization requires **reducing graph builds**, not just faster hashing:

1. **Topology-based grouping** - Group by valence without hashing (instant)
2. **Signature caching** - Assume signatures stable across iterations
3. **Native module** - Move critical path to Rust/C++ if needed
4. **WebGPU strategy** - Batch by signature with fast pre-selection

## Action Items

1. ✓ Prototype incremental hashing approach
2. ✓ Add benchmark suite for graph hashing performance
3. ✓ Implement GraphBuilder with leaf tracking
4. Measure real-world dynamic graph frequency in developable-sphere demo
5. Design API for static vs dynamic graph declaration
6. Consider topology-based grouping as alternative to full hashing
7. Document tradeoffs for users
