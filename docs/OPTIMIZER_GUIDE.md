# ScalarAutograd Optimizer Guide

Complete guide to choosing and using optimization algorithms in ScalarAutograd.

---

## Quick Reference

| Optimizer | Use Case | Speed | Memory | Convergence |
|-----------|----------|-------|--------|-------------|
| **SGD/Adam** | General training, neural nets | Fast | Low | Steady |
| **Least Squares** | Constraint solving, calibration | Very Fast | Medium | Excellent |
| **L-BFGS** | High-precision optimization | Medium | Low | Excellent |

---

## 1. Gradient Descent Optimizers (SGD, Adam, AdamW)

### How They Work

**Core Idea**: Iteratively move parameters in the direction opposite to the gradient.

```
θ_new = θ_old - learning_rate × ∇f(θ)
```

**Variants:**
- **SGD**: Pure gradient descent with momentum
- **Adam**: Adaptive learning rates per parameter + momentum
- **AdamW**: Adam with decoupled weight decay

### When to Use

✅ **Best for:**
- Training neural networks
- Large-scale optimization (thousands of parameters)
- Stochastic/mini-batch settings
- When you want incremental progress

❌ **Avoid when:**
- You need high precision (e.g., 1e-8 tolerance)
- You have a well-structured problem (use L-BFGS instead)
- You're solving constraints (use Least Squares)

### Usage Example

```typescript
import { V, Adam } from 'scalar-autograd';

// Setup
const params = [V.W(Math.random()), V.W(Math.random())];
const optimizer = new Adam(params, { learningRate: 0.01 });

// Training loop
for (let epoch = 0; epoch < 1000; epoch++) {
  // Compute loss
  const loss = computeLoss(params);

  // Backprop and update
  optimizer.zeroGrad();
  loss.backward();
  optimizer.step();
}
```

### Performance Characteristics

- **Iterations needed**: 100s to 10,000s
- **Per-iteration cost**: Very low (just gradient computation)
- **Convergence rate**: Linear (first-order method)
- **Tuning required**: Learning rate, momentum, decay

### Tips

1. **Start with Adam** - usually works well out of the box
2. **Learning rate**: Try 0.001, 0.01, or 0.1
3. **Use early stopping** to avoid overfitting
4. **Gradient clipping** for stability: `opt.clipGradients(1.0)`

---

## 2. Nonlinear Least Squares (Levenberg-Marquardt)

### How It Works

**Core Idea**: Minimize sum of squared residuals using local quadratic approximation.

```
minimize: ½ Σ rᵢ²
```

**Algorithm**: Levenberg-Marquardt interpolates between:
- **Gauss-Newton** (fast, when close to solution)
- **Gradient Descent** (stable, when far from solution)

**Key Step**: Builds Jacobian matrix J, solves (JᵀJ + λI)δ = -Jᵀr

### When to Use

✅ **Best for:**
- **Constraint solving** (sketch constraints, IK, physics)
- Curve fitting / calibration
- Small to medium problems (< 1000 parameters)
- When you have **multiple residuals** (not just one cost)
- High precision requirements

❌ **Avoid when:**
- Single scalar objective (use L-BFGS instead)
- Unbounded problems (residuals never reach zero)
- Very large scale (> 10,000 parameters)

### Usage Example

```typescript
import { V } from 'scalar-autograd';

// Define parameters and constraints
const x1 = V.W(0, 'x1');
const y1 = V.W(0, 'y1');
const x2 = V.W(3, 'x2');
const y2 = V.W(4, 'y2');
const params = [x1, y1, x2, y2];

// Define residuals (constraints)
const residualFn = (p: V.Value[]) => {
  const [x1, y1, x2, y2] = p;

  // Distance constraint: sqrt((x2-x1)² + (y2-y1)²) = 5
  const dx = V.sub(x2, x1);
  const dy = V.sub(y2, y1);
  const dist = V.sqrt(V.add(V.mul(dx, dx), V.mul(dy, dy)));
  const distResidual = V.sub(dist, V.C(5));

  // Horizontal constraint: y2 - y1 = 0
  const horizResidual = V.sub(y2, y1);

  return [distResidual, horizResidual];
};

// Solve
const result = V.nonlinearLeastSquares(params, residualFn, {
  maxIterations: 100,
  verbose: true
});
```

### Performance Optimization

**Use compiled residuals for 25x speedup:**

```typescript
import { CompiledResiduals } from 'scalar-autograd';

// Compile once
const compiled = CompiledResiduals.compile(params, residualFn);

// Solve many times (e.g., interactive sketching)
for (let i = 0; i < 100; i++) {
  params.forEach(p => p.data = /* new values */);
  V.nonlinearLeastSquares(params, compiled);
}
```

### Performance Characteristics

- **Iterations needed**: 5-50 (very fast convergence)
- **Per-iteration cost**: Moderate (Jacobian computation + linear solve)
- **Convergence rate**: Quadratic (near solution)
- **Tuning required**: Minimal (algorithm is robust)

### Real-World Applications

1. **Sketch Solvers**: 100 distance constraints → 1 kernel, 25x faster
2. **IK Systems**: Joint angle constraints
3. **Calibration**: Camera parameters, sensor fusion
4. **Physics**: Particle systems, collision resolution

---

## 3. L-BFGS (Limited-memory BFGS)

### How It Works

**Core Idea**: Quasi-Newton method that approximates the inverse Hessian using gradient history.

**Algorithm:**
1. Compute gradient: ∇f(x)
2. Estimate Hessian inverse: H ≈ H_BFGS (from stored gradient pairs)
3. Search direction: d = -H × ∇f
4. Line search: find optimal step size α
5. Update: x_new = x + α × d

**Memory**: Stores only last `m` (default 10) gradient/position pairs.

### When to Use

✅ **Best for:**
- **Single scalar objective** optimization
- High precision requirements (1e-8 or better)
- Smooth, well-behaved functions
- Medium-sized problems (10-1000 parameters)
- When second-order convergence matters

❌ **Avoid when:**
- You have multiple residuals (use Least Squares)
- Noisy/stochastic objectives
- Very large scale (> 10,000 parameters - memory grows)
- Non-smooth functions (L-BFGS assumes smoothness)

### Usage Example

```typescript
import { lbfgs, V } from 'scalar-autograd';

// Setup
const x = V.W(-1.2);
const y = V.W(1.0);
const params = [x, y];

// Objective function (scalar)
const objectiveFn = (p: V.Value[]) => {
  const [x, y] = p;
  const a = V.sub(V.C(1), x);
  const b = V.sub(y, V.pow(x, 2));
  return V.add(V.pow(a, 2), V.mul(V.C(100), V.pow(b, 2)));
};

// Optimize
const result = lbfgs(params, objectiveFn, {
  maxIterations: 100,
  verbose: true
});

console.log(`Solution: x=${x.data}, y=${y.data}`);
```

### Performance Optimization

**Use compiled objectives for 5-10x speedup:**

```typescript
// Compile once
const compiled = V.compileObjective(params, objectiveFn);

// Solve with compiled gradients (much faster)
const result = lbfgs(params, compiled, { verbose: true });
```

### Performance Characteristics

- **Iterations needed**: 10-100
- **Per-iteration cost**: Low (gradient + line search)
- **Convergence rate**: Superlinear (second-order approximation)
- **Tuning required**: Minimal (robust defaults)

### Classic Test Problems

**Rosenbrock function** (banana-shaped valley):
```typescript
f(x, y) = (1-x)² + 100(y-x²)²
// Global minimum at (1, 1)
```

**Beale function** (challenging):
```typescript
f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
// Global minimum at (3, 0.5)
```

---

## Decision Tree

```
START: What are you optimizing?

├─ Training a neural network?
│  └─> Use SGD/Adam
│
├─ Solving geometric constraints? (e.g., sketch, IK)
│  ├─ Multiple constraints/residuals?
│  │  └─> Use Least Squares (Levenberg-Marquardt)
│  │     └─> Compile with CompiledResiduals for 25x speedup
│  │
│  └─ Single objective function?
│      └─> Use L-BFGS
│         └─> Compile with V.compileObjective() for 5-10x speedup
│
├─ Need highest precision? (1e-8+)
│  └─> Use L-BFGS with compiled objective
│
├─ Very large scale? (> 10,000 params)
│  └─> Use Adam (memory efficient)
│
└─ Noisy/stochastic data?
   └─> Use Adam (robust to noise)
```

---

## Performance Comparison

### Problem: Minimize Rosenbrock function

| Method | Iterations | Time | Final Cost |
|--------|-----------|------|------------|
| SGD | 5000 | 150ms | 1e-4 |
| Adam | 1000 | 50ms | 1e-6 |
| L-BFGS (uncompiled) | 40 | 120ms | 1e-12 |
| L-BFGS (compiled) | 40 | 25ms | 1e-12 |

**Takeaway**: L-BFGS converges fastest to high precision, compilation gives 5x speedup.

### Problem: Solve 100 distance constraints

| Method | Iterations | Time | Note |
|--------|-----------|------|------|
| Adam | 500 | 200ms | Slow convergence |
| Least Squares (uncompiled) | 10 | 450ms | Fast convergence |
| Least Squares (compiled) | 10 | 18ms | **25x faster!** |

**Takeaway**: Least Squares + compilation is ideal for constraint solving.

---

## Compilation Guide

### What is Compilation?

Instead of using graph backward (slow), compile the gradient/Jacobian computation to optimized JavaScript code (fast).

**Speedups:**
- **Least Squares**: 7-25x faster
- **L-BFGS**: 5-10x faster
- **SGD/Adam**: Not yet supported (future work)

### When to Compile

✅ **Compile when:**
- Problem structure is **fixed** (same computation graph every iteration)
- You'll solve the same problem **multiple times**
- Performance matters

❌ **Don't compile when:**
- Graph structure changes (e.g., different number of residuals)
- You're just experimenting
- One-off optimization

### How to Compile

**Least Squares:**
```typescript
const compiled = CompiledResiduals.compile(params, residualFn);
V.nonlinearLeastSquares(params, compiled);
```

**L-BFGS:**
```typescript
const compiled = V.compileObjective(params, objectiveFn);
lbfgs(params, compiled);
```

**Break-even**: Compilation takes ~10-50ms. If you do >10 evaluations, it pays off.

---

## Advanced Topics

### Adaptive Damping (Least Squares)

Levenberg-Marquardt uses damping parameter λ:
- Large λ → acts like gradient descent (stable)
- Small λ → acts like Gauss-Newton (fast)

Algorithm automatically adjusts λ based on progress.

### Line Search (L-BFGS)

Uses **Wolfe conditions** to find good step size α:
1. **Armijo condition**: Sufficient decrease in objective
2. **Curvature condition**: Don't overshoot minimum

Ensures both stability and progress.

### Memory vs. Convergence (L-BFGS)

`historySize` parameter (default: 10):
- Larger → better Hessian approximation, more memory
- Smaller → less memory, slower convergence

Rule of thumb: 10 is good for most problems.

---

## Common Pitfalls

### 1. Wrong Optimizer for Problem Type

❌ **Bad**: Using L-BFGS for multiple residuals
```typescript
// DON'T: sum residuals into scalar for L-BFGS
const objective = V.add(residual1, V.add(residual2, residual3));
lbfgs(params, () => objective);
```

✅ **Good**: Use Least Squares directly
```typescript
V.nonlinearLeastSquares(params, () => [residual1, residual2, residual3]);
```

### 2. Forgetting to Compile

❌ **Bad**: Slow uncompiled path for production
```typescript
for (let i = 0; i < 1000; i++) {
  V.nonlinearLeastSquares(params, residualFn);  // Recomputes graph 1000x!
}
```

✅ **Good**: Compile once, reuse
```typescript
const compiled = CompiledResiduals.compile(params, residualFn);
for (let i = 0; i < 1000; i++) {
  V.nonlinearLeastSquares(params, compiled);  // 25x faster!
}
```

### 3. Wrong Learning Rate (Adam)

Too high → divergence, too low → slow convergence

**Strategy**: Try 0.01 first, then adjust by 10x up or down.

### 4. Not Using Kernel Reuse

For sketch solvers with 100 identical constraints:

❌ **Bad**: 100 unique kernels
✅ **Good**: 1 kernel reused 100x (automatic with CompiledResiduals)

---

## Summary

| Problem Type | Algorithm | Compile? | Expected Performance |
|--------------|-----------|----------|---------------------|
| Neural network training | Adam | No | 100-10k iterations |
| Constraint solving | Least Squares | **Yes** | 5-50 iterations, 25x faster |
| High-precision optimization | L-BFGS | **Yes** | 10-100 iterations, 5-10x faster |
| Large-scale (>10k params) | Adam | No | Many iterations, memory efficient |

**Golden Rules:**
1. Multiple residuals → Least Squares
2. Single objective → L-BFGS
3. Neural nets → Adam
4. Always compile for production performance

---

## Further Reading

- `docs/L-BFGS.md` - L-BFGS implementation details
- `docs/KERNEL_REUSE_RESULTS.md` - Compilation performance analysis
- `src/LBFGS.ts` - L-BFGS source code
- `src/NonlinearLeastSquares.ts` - Levenberg-Marquardt implementation
- `src/Optimizers.ts` - SGD, Adam, AdamW implementations
