# DifferentiablePlaneAlignment Compilation Bug - Test Results

## Summary

Tests have been created to verify that the compiled and uncompiled versions of `DifferentiablePlaneAlignment` produce identical results. **A significant bug has been found.**

## Test Files

1. **`planeAlignment.test.ts`** - Main test suite comparing compiled vs uncompiled
2. **`planeAlignment-diagnostic.test.ts`** - Detailed diagnostic output showing gradient differences

## Key Findings

### ✅ What Works

- **Residuals match perfectly**: Individual function values are identical between compiled and uncompiled versions
- **Total energy matches perfectly**: Sum of residuals is identical (verified to machine precision)

### ❌ What's Broken

- **Gradients DO NOT match**: Significant differences (up to 20-50% in some components!)
  - Example from diagnostic test:
    - Worst case: Vertex 8, y-component has **48.96% relative error**
    - Uncompiled: `-3.2461335944e-1`
    - Compiled: `-1.6568246498e-1`
  - Gradient norm difference: ~1.6% relative error
  - Some gradient components match perfectly (machine precision), others differ significantly

## What This Means

The compiled gradient computation has a bug. The compiled kernels are computing:
- ✅ Correct function values (energies)
- ❌ Incorrect gradients

This causes optimization with compiled gradients to follow the wrong descent direction, leading to different solutions than the uncompiled version.

## Test Data

### Small Sphere (12 vertices, level 0)
```
Energies:
  Uncompiled: 2.4951057062e+0
  Compiled:   2.4951057062e+0
  Difference: 0.000000e+0          ✅ Perfect match

Gradient norms:
  Uncompiled: 2.0578354082e+0
  Compiled:   2.0911829783e+0
  Difference: 3.334757e-2
  Relative:   1.620517e-2          ❌ 1.6% error
```

### Larger Sphere (42 vertices, level 1)
```
Energies:
  Uncompiled: 2.366761e+0
  Compiled:   2.366761e+0          ✅ Perfect match

Gradient norms:
  Uncompiled: 4.498645e+0
  Compiled:   4.496300e+0
  Relative:   2.108e-2             ❌ 2.1% error

Max gradient component error: 9.482e-2 (absolute)
```

## Investigation Results

### Comprehensive Operator Tests ✅

Created `test/compiled-operators.spec.ts` with 36 tests covering ALL operators:
- ✅ All arithmetic operations (add, sub, mul, div, pow, sqrt, abs, etc.)
- ✅ All transcendental functions (exp, log)
- ✅ All trig functions (sin, cos, tan, asin, acos, atan)
- ✅ All activation functions (relu, sigmoid, tanh, softplus)
- ✅ All comparison operations (min, max)
- ✅ Complex composite expressions
- ✅ Edge cases (tiny values, mixed scales, deep nesting)

**Result: ALL PASS!**

### Multi-Residual Tests ✅

Created `test/compiled-multi-residual.spec.ts` with 10 tests covering:
- ✅ Independent residuals
- ✅ Shared parameters across residuals
- ✅ Complex expressions with multiple residuals
- ✅ Shared subexpressions
- ✅ Distance constraints (IK-like)
- ✅ Many residuals (10+)
- ✅ Normalized vectors as residuals
- ✅ Cross products as residuals

**Result: ALL PASS!**

### Minimal Reproduction Tests ✅

Created `test/plane-alignment-minimal.spec.ts` testing:
- ✅ Vec3 dot product
- ✅ Vec3 cross product
- ✅ Multiple weighted cross products (exact logic from DifferentiablePlaneAlignment)
- ✅ Full vertex energy simulation (exact algorithm)

**Result: ALL PASS!**

## The Mystery

We have **comprehensive proof** that:
1. ✅ All individual operators work correctly
2. ✅ Multi-residual compilation works correctly
3. ✅ The exact algorithm from DifferentiablePlaneAlignment works correctly **in isolation**
4. ❌ But DifferentiablePlaneAlignment with actual TriangleMesh data **fails**

### Hypothesis

The bug is NOT in the compilation system itself. The bug must be in **how TriangleMesh data flows through the computation**. Possible causes:
1. Cached values in `mesh.getFaceNormal()` not being updated correctly during compilation
2. Mesh state being modified during compilation in a way that affects gradient computation
3. Parameter ordering or indexing issue specific to how mesh vertices are converted to parameters
4. Some Value objects being shared/reused incorrectly between mesh operations

## Critical Discovery: Kernel Caching is NOT the Issue

Added test that compiles the same residuals twice with fresh parameters:
- ✅ Both compilations produce **identical gradients** (diff = 0)
- ❌ But **both are wrong** compared to uncompiled (up to 11% error)

**Conclusion**: The bug is in the compiled kernel code itself, not in caching.

## Critical Discovery: Different Constant Objects!

Created `compilation-state-test.spec.ts` which reveals:

### Leaf Node Analysis
- **127 total leaf nodes** in the computation graph for one residual
- **Only 18/127 are the same Value objects** between compilation and uncompiled!
- **109/127 are different objects** (all constants like `V.C(0)`, `V.C(1)`, `V.C(1e-12)`)

### What This Means
When `DifferentiablePlaneAlignment.computeResiduals()` is called twice:
1. **During compilation**: Creates constant Value objects (`V.C(0)`, etc.)
2. **Uncompiled execution**: Creates **NEW** constant Value objects with same values

The compiled kernels have indices pointing to the FIRST set of constants in the registry.
The uncompiled graph uses the SECOND set of constants.

### Why This Causes Different Gradients
- Forward pass: ✅ Works correctly (values match perfectly)
- Backward pass: ❌ Compiled kernel accumulates gradients to wrong Value objects?

### Next Investigation
Need to understand why having different constant objects causes gradient mismatch when:
1. The constant VALUES are the same (both are 0, 1, 1e-12, etc.)
2. The forward pass produces identical results
3. Only gradients differ

Possible issues:
- Registry index mapping might be off
- Gradient accumulation might be using wrong indices
- Some constants might actually be reused across multiple operations, causing wrong grad accumulation

**Action**: Create minimal test without TriangleMesh to isolate the constant-reuse issue.

## Running the Tests

```bash
# Run all plane alignment tests
npm test -- demos/developable-sphere/test/planeAlignment

# Run diagnostic test only
npm test -- demos/developable-sphere/test/planeAlignment-diagnostic.test.ts
```
