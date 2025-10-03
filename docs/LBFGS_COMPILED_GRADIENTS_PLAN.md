# L-BFGS Compiled Gradients - Implementation Plan

## Overview

Upgrade L-BFGS optimizer to use compiled gradients instead of graph backward(), leveraging the kernel reuse infrastructure built for CompiledResiduals.

## Current State

**L-BFGS gradient computation** (src/LBFGS.ts:51-69):
```typescript
function computeObjectiveAndGradient(params, objectiveFn) {
  params.forEach(p => p.grad = 0);
  const objective = objectiveFn(params);
  Value.zeroGradTree(objective);  // ❌ Recursive traversal
  objective.backward();            // ❌ Closure execution
  return { cost: objective.data, gradient: params.map(p => p.grad) };
}
```

**Problem**: Called many times per iteration (5-20x in line search), making it a performance bottleneck.

## Architecture Options

### Option A: Reuse CompiledResiduals ⭐ RECOMMENDED

Treat objective function as a single-residual system.

**Approach**:
```typescript
// Wrap objective as residual function
const residualFn = (p: Value[]) => [objectiveFn(p)];
const compiled = CompiledResiduals.compile(params, residualFn);

// Evaluate to get cost and gradient
const { residuals, J, cost } = compiled.evaluate(params);
const gradient = J[0];  // First (only) row of Jacobian
```

**Pros**:
- ✅ Zero new code - reuses existing infrastructure
- ✅ Works immediately with all kernel reuse benefits
- ✅ Proves concept before committing to specialized implementation
- ✅ Simpler to test and validate

**Cons**:
- ⚠️ Minor overhead: single-element residual array, 1-row Jacobian matrix
- ⚠️ API feels slightly awkward (wrapping scalar in array)

**Verdict**: Start here. Profile later to see if overhead matters.

---

### Option B: Create CompiledObjective (Specialized)

New class specifically for scalar objective functions.

**API Design**:
```typescript
class CompiledObjective {
  static compile(
    params: Value[],
    objectiveFn: (params: Value[]) => Value
  ): CompiledObjective;

  evaluate(params: Value[]): {
    cost: number;
    gradient: number[];
  };
}
```

**Implementation**: Nearly identical to CompiledResiduals, but:
- Single kernel instead of array
- Returns `{ cost, gradient }` instead of `{ residuals, J, cost }`
- No need for residual descriptors array

**Pros**:
- ✅ Clean, purpose-built API
- ✅ No array overhead
- ✅ Clear semantics (objective vs residuals)

**Cons**:
- ❌ Code duplication from CompiledResiduals
- ❌ More work to implement and maintain
- ❌ Need to decide when to use which

**Verdict**: Only pursue if profiling shows Option A overhead is significant.

---

### Option C: Generalize to CompiledFunction (Unified)

Abstract base class for compiled computation graphs.

**Architecture**:
```typescript
abstract class CompiledFunction {
  protected registry: ValueRegistry;
  protected kernelPool: KernelPool;
  // Shared compilation logic
}

class CompiledObjective extends CompiledFunction {
  // Returns { cost, gradient }
}

class CompiledResiduals extends CompiledFunction {
  // Returns { residuals, J, cost }
}
```

**Pros**:
- ✅ Shared infrastructure, DRY principle
- ✅ Extensible to other compilation needs

**Cons**:
- ❌ Over-engineering for current needs
- ❌ Adds complexity without clear benefit
- ❌ Premature abstraction

**Verdict**: YAGNI. Reject unless we find 3+ similar compilation patterns.

---

## Implementation Plan

### Phase 1: Reuse CompiledResiduals (Quick Win)

**Steps**:

1. **Update L-BFGS signature** (src/LBFGS.ts):
   ```typescript
   export function lbfgs(
     params: Value[],
     objectiveFn: ((params: Value[]) => Value) | CompiledResiduals,
     options?: LBFGSOptions
   ): LBFGSResult
   ```

2. **Add compiled path to computeObjectiveAndGradient**:
   ```typescript
   function computeObjectiveAndGradient(
     params: Value[],
     objectiveFn: ((params: Value[]) => Value) | CompiledResiduals
   ): { cost: number; gradient: number[] } {
     if (objectiveFn instanceof CompiledResiduals) {
       const { residuals, J, cost } = objectiveFn.evaluate(params);
       return { cost, gradient: J[0] };
     }

     // Existing graph backward path...
   }
   ```

3. **Add helper function for compilation**:
   ```typescript
   // In V.ts or LBFGS.ts
   function compileObjective(
     params: Value[],
     objectiveFn: (params: Value[]) => Value
   ): CompiledResiduals {
     const residualFn = (p: Value[]) => [objectiveFn(p)];
     return CompiledResiduals.compile(params, residualFn);
   }
   ```

4. **Update documentation**:
   - Add example showing compiled usage
   - Note performance benefits (5-10x speedup expected)

5. **Add tests** (test/LBFGS-compiled.spec.ts):
   - Verify compiled matches uncompiled results
   - Test with various objective functions (Rosenbrock, quadratic, etc.)
   - Benchmark performance improvement

**Expected Results**:
- Line search: 5-20 gradient evaluations per iteration → 5-10x faster
- Overall L-BFGS: ~2-5x faster (depends on line search overhead vs iteration overhead)

---

### Phase 2: Profile and Optimize (If Needed)

**Only proceed if profiling shows >5% overhead from array wrapping.**

1. Profile Option A implementation
2. Measure overhead: array allocation, matrix indexing
3. If significant, implement Option B (CompiledObjective)
4. Benchmark both, compare

**Threshold for Phase 2**: Overhead >5% of total runtime on representative problems.

---

## Design Decisions

### Why Reuse First?

1. **Validate approach** with minimal code
2. **Measure real-world benefit** before optimizing
3. **Avoid premature optimization** - array overhead may be negligible
4. **Faster to production** - can ship immediately

### When to Create CompiledObjective?

Only if:
- Profiling shows measurable overhead (>5%)
- Used frequently enough to justify maintenance
- API confusion from wrapping causes user friction

### What About Other Optimizers?

**SGD, Adam, AdamW**: Currently use graph backward in `opt.step()`.

These could also benefit from compiled gradients:
```typescript
// Future work - compile loss function
const compiledLoss = compileObjective(params, lossFn);

for (let epoch = 0; epoch < 1000; epoch++) {
  const { cost, gradient } = compiledLoss.evaluate(params);
  // Apply gradients...
}
```

But defer until L-BFGS proves the pattern works.

---

## Success Criteria

**Phase 1 Complete When**:
- ✅ L-BFGS accepts CompiledResiduals
- ✅ Tests pass (compiled matches uncompiled)
- ✅ Benchmark shows >2x speedup on medium problems
- ✅ Documentation updated with examples

**Phase 2 Triggered If**:
- ⚠️ Profiling shows >5% overhead from wrapping
- ⚠️ User feedback indicates API confusion

---

## Open Questions

1. **Should we auto-compile?**
   - Option: Detect if objectiveFn is fixed structure, compile automatically
   - Decision: No - explicit is better than implicit, matches NonlinearLeastSquares pattern

2. **Expose compileObjective as public API?**
   - Option A: `V.compileObjective()` or `CompiledResiduals.compileObjective()`
   - Option B: Users wrap manually: `CompiledResiduals.compile(params, p => [obj(p)])`
   - Decision: Start with Option B, add helper if users request it

3. **Should line search compilation be separate?**
   - Currently compiles objective once, reuses in line search
   - Alternative: Compile line search function directly (more complex)
   - Decision: Current approach is fine - compilation amortized over iterations

---

## Risk Assessment

**Low Risk**:
- Reusing proven CompiledResiduals infrastructure
- Minimal code changes to L-BFGS
- Backwards compatible (objectiveFn function still works)

**Medium Risk**:
- Performance may not improve as much as expected (mitigated by benchmarking)
- API wrapping may confuse users (mitigated by documentation)

**High Risk**:
- None identified

---

## Timeline Estimate

**Phase 1** (Reuse CompiledResiduals):
- Implementation: 1-2 hours
- Testing: 1 hour
- Documentation: 30 minutes
- **Total: 2.5-3.5 hours**

**Phase 2** (If needed - CompiledObjective):
- Design: 30 minutes
- Implementation: 2-3 hours
- Testing: 1 hour
- Migration: 1 hour
- **Total: 4.5-5.5 hours**

---

## Next Steps

1. Implement Phase 1 (reuse CompiledResiduals)
2. Write tests and benchmarks
3. Profile on representative problems
4. Decide whether Phase 2 is warranted
5. Update documentation

---

## References

- `docs/KERNEL_REUSE_RESULTS.md` - CompiledResiduals architecture
- `src/CompiledResiduals.ts` - Implementation to reuse
- `src/LBFGS.ts` - Current gradient computation
- `src/NonlinearLeastSquares.ts` - Pattern for accepting compiled/uncompiled

---

**Status**: Plan complete, ready for implementation
**Recommended**: Start with Phase 1 (Option A)
**Author**: Planning session 2025-01-03
