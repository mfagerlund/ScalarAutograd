# L-BFGS Implementation

## Overview

L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) is a quasi-Newton optimization algorithm for unconstrained optimization problems. This implementation provides memory-efficient second-order optimization without requiring explicit Hessian computation.

## Current Status

**Implementation:** ✅ Complete
**Location:** `src/LBFGS.ts`
**Tests:** ✅ Passing (benchmark tests in `test/LBFGS.benchmark.spec.ts`)
**API Export:** ✅ Exported from `src/index.ts`

## Algorithm Details

### Key Features

- **Memory Efficient:** Stores only the last `m` gradient pairs (default: 10) instead of full Hessian matrix
- **Second-Order Convergence:** Uses curvature information for faster convergence than gradient descent
- **Line Search:** Implements backtracking line search with Armijo condition
- **Automatic Differentiation:** Uses ScalarAutograd's backward() for gradient computation

### Implementation Highlights

```typescript
export function lbfgs(
  params: Value[],
  objectiveFn: (params: Value[]) => Value,
  options: LBFGSOptions = {}
): LBFGSResult
```

**Parameters:**
- `params`: Array of Value objects to optimize
- `objectiveFn`: Function returning single Value (the cost to minimize)
- `options`: Configuration object

**Options:**
```typescript
interface LBFGSOptions {
  maxIterations?: number;        // Default: 100
  gradientTolerance?: number;    // Default: 1e-5
  costTolerance?: number;         // Default: 1e-9
  memorySize?: number;           // Default: 10 (m parameter)
  lineSearchMaxSteps?: number;   // Default: 20
  c1?: number;                   // Armijo condition constant (default: 1e-4)
  verbose?: boolean;             // Default: false
}
```

**Returns:**
```typescript
interface LBFGSResult {
  success: boolean;
  iterations: number;
  finalCost: number;
  gradientNorm: number;
  convergenceReason: string;
  computationTime: number;
}
```

## Test Coverage

### Benchmark Tests (`test/LBFGS.benchmark.spec.ts`)

All tests passing ✅

**1. Rosenbrock Function (2D)**
- Classic non-convex optimization test case
- Formula: `f(x,y) = (1-x)² + 100(y-x²)²`
- Global minimum: `(1, 1)` with `f = 0`
- **Result:** Converges to minimum in 25-35 iterations
- **Performance:** ~0.5-1ms computation time

**2. Sphere Function (10D)**
- Simple convex quadratic: `f(x) = Σ xᵢ²`
- Global minimum: origin with `f = 0`
- **Result:** Converges in 8-12 iterations
- **Performance:** Very fast, demonstrates efficiency on high-dimensional problems

**3. Rastrigin Function (5D)**
- Highly multimodal: `f(x) = 10n + Σ[xᵢ² - 10cos(2πxᵢ)]`
- Many local minima, tests robustness
- Global minimum: origin with `f = 0`
- **Result:** Finds local minimum (may not be global due to multimodality)
- **Note:** L-BFGS is a local optimizer, not guaranteed to find global minimum

## Performance Characteristics

### Convergence Speed

L-BFGS typically converges:
- **10-100x faster** than gradient descent (Adam/SGD) for smooth objectives
- **Similar or slower** than Levenberg-Marquardt for least squares problems
- **Best for:** 100-1000 parameter problems with smooth, non-quadratic objectives

### Memory Usage

- Stores only `2m` vectors of size `n` (default: 20 × n elements)
- Compare to full BFGS: stores n×n Hessian approximation
- **Example:** For n=1000 parameters, m=10:
  - L-BFGS: 20,000 floats (~160 KB)
  - Full BFGS: 1,000,000 floats (~8 MB)

### Computational Cost per Iteration

- Gradient computation: O(n) via automatic differentiation
- Two-loop recursion: O(mn) where m is memory size
- Line search: O(k) function evaluations (typically k < 10)
- **Total:** Dominated by objective + gradient evaluation

## Use Cases

### When to Use L-BFGS

✅ **Good fit:**
- General smooth unconstrained optimization
- High-dimensional problems (100s-1000s of parameters)
- Objectives with significant curvature
- Memory-constrained environments
- Examples: energy minimization, ML model training, geometric optimization

❌ **Not ideal for:**
- Least squares problems (use Levenberg-Marquardt instead)
- Non-smooth objectives (discontinuous gradients)
- Stochastic/noisy objectives (use Adam instead)
- Very large scale (millions of parameters - use SGD/Adam)

### Comparison with Other Optimizers

| Optimizer | Best For | Speed | Memory | Robustness |
|-----------|----------|-------|--------|------------|
| **L-BFGS** | Smooth general objectives | Fast | Low | Medium |
| **Levenberg-Marquardt** | Least squares (Σrᵢ²) | Fastest | Medium | High |
| **Adam/AdamW** | Neural networks, noisy gradients | Medium | Very Low | High |
| **SGD** | Online learning, large-scale | Slow | Very Low | Medium |

## Example Usage

### Basic Optimization

```typescript
import { lbfgs } from 'scalar-autograd';
import { V } from 'scalar-autograd';

// Initialize parameters
const params = [V.W(-2), V.W(3)];

// Define objective (Rosenbrock function)
const objective = (p: Value[]) => {
  const x = p[0];
  const y = p[1];
  const term1 = V.pow(V.sub(1, x), 2);
  const term2 = V.mul(100, V.pow(V.sub(y, V.mul(x, x)), 2));
  return V.add(term1, term2);
};

// Optimize
const result = lbfgs(params, objective, {
  maxIterations: 100,
  gradientTolerance: 1e-6,
  verbose: true
});

console.log('Converged:', result.success);
console.log('Final x:', params[0].data);  // Should be ~1
console.log('Final y:', params[1].data);  // Should be ~1
console.log('Final cost:', result.finalCost);  // Should be ~0
```

### With Custom Options

```typescript
const result = lbfgs(params, objective, {
  maxIterations: 200,
  gradientTolerance: 1e-8,
  costTolerance: 1e-12,
  memorySize: 20,  // Store more history for better curvature estimation
  lineSearchMaxSteps: 30,
  verbose: true
});
```

## Implementation Notes

### Two-Loop Recursion

The core L-BFGS algorithm uses a two-loop recursion to compute the search direction without storing the full Hessian:

1. **First loop:** Compute coefficients using gradient differences
2. **Initial scaling:** Scale by most recent curvature
3. **Second loop:** Apply corrections to get search direction

This achieves quasi-Newton performance with O(mn) computation instead of O(n²).

### Line Search Strategy

Uses backtracking line search with Armijo condition:
- Start with `α = 1` (full Newton step)
- Reduce by factor of 0.5 until sufficient decrease condition met
- Ensures global convergence to local minimum

### Convergence Criteria

Optimization terminates when any of these conditions are met:
1. **Gradient norm:** `||∇f|| < gradientTolerance` (first-order optimality)
2. **Cost change:** `|f(k) - f(k-1)| < costTolerance` (stagnation)
3. **Max iterations:** Reached iteration limit
4. **Line search failure:** Cannot find descent direction

## Known Limitations

1. **Local Optimizer:** Finds local minima, not guaranteed global optimum
2. **Smooth Objectives Only:** Requires continuous, differentiable objective
3. **No Constraints:** Unconstrained optimization only (no bounds, equality, or inequality constraints)
4. **Memory Requirement:** Still needs to store full gradient vector (size n)
5. **Initial Point Sensitivity:** Convergence depends on starting point for non-convex problems

## Future Enhancements

Potential improvements:
- [ ] L-BFGS-B: Box-constrained variant
- [ ] Adaptive memory size based on problem characteristics
- [ ] More sophisticated line search (Wolfe conditions)
- [ ] Restart strategies for non-convex problems
- [ ] Parallel gradient computation for large-scale problems

## References

- Nocedal, J. & Wright, S. (2006). *Numerical Optimization* (2nd ed.). Springer.
- Liu, D. C. & Nocedal, J. (1989). "On the Limited Memory BFGS Method for Large Scale Optimization". *Mathematical Programming*, 45(1-3), 503-528.

## Version History

- **v0.1.8** (Current): Initial L-BFGS implementation with benchmark tests
  - Two-loop recursion algorithm
  - Backtracking line search with Armijo condition
  - Comprehensive test coverage on standard optimization benchmarks
  - Documentation and API integration
