# ScalarAutograd for TypeScript

A tiny scalar autograd engine for TypeScript/JavaScript.

Scautograd enables automatic differentiation for scalar operations, similar to what you'd find in PyTorch's `autograd`, but designed for TypeScript codebases. This makes it useful for building and training small neural networks, performing optimization, or experimenting with gradient-based techniques—entirely in the browser or Node.js.

## Features
- Scalar `Value` objects for tracking data, gradients, and computation graph.
- Backpropagation via `.backward()` to compute gradients automatically.
- Clean, TypeScript-first API.
- Does *NOT* handle matrices or tensors, just scalars.

## Installation

Simply copy the files in this folder into your project, or import as a local module if desired.

## Basic Usage

```typescript
import { V } from './V';

// Basic differentiation using static V API
const x = V.W(2.0); // differentiable variable
const y = V.W(3.0);
const z = V.add(V.mul(x, y), V.pow(x, 2)); // z = x*y + x^2
z.backward();
console.log('dz/dx:', x.grad); // Output: dz/dx = y + 2*x = 3 + 2*2 = 7
console.log('dz/dy:', y.grad); // Output: dz/dy = x = 2
```

## Example: Tiny Gradient Descent

```typescript
const a = V.W(5);
const b = V.W(-3);
const c = V.sin(V.mul(a, b)); // f = sin(a * b)
c.backward();
console.log(a.grad, b.grad); // Gradients w.r.t. a and b
```

## Example: Solving for Parameters via Backpropagation

Here's how you can use Scautograd's backpropagation and a simple optimizer to fit a linear regression model (y = 2x + 3):

```typescript
import { V } from './V';
import { SGD } from './Optimizers';

// Initialize parameters
let w = V.W(Math.random(), "w");
let b = V.W(Math.random(), "b");

// Example data: y = 2x + 3
const samples = [
  { x: 1, y: 5 },
  { x: 2, y: 7 },
  { x: 3, y: 9 },
];

const opt = new SGD([w, b], { learningRate: 0.1 });

for (let epoch = 0; epoch < 300; ++epoch) {
  let losses = [];
  for (const sample of samples) {
    const x = V.C(sample.x, "x");
    const pred = V.add(V.mul(w, x), b);
    const target = V.C(sample.y, "target");
    const loss = V.pow(V.sub(pred, target), 2);
    losses.push(loss);
  }
  const totalLoss = V.mean(losses);
  opt.zeroGrad();
  totalLoss.backward();
  opt.step();
  if (totalLoss.data < 1e-4) break;
}

console.log('Fitted w:', w.data); // ~2
console.log('Fitted b:', b.data); // ~3
```

This pattern—forward pass, backward for gradients, and calling `optimizer.step()`—applies to more complex optimization tasks and neural networks as well!

## Example: Nonlinear Least Squares Solver

For problems where you need to minimize the sum of squared residuals, the built-in Levenberg-Marquardt solver is much faster than gradient descent:

```typescript
import { V } from './V';

// Fit a circle to noisy points
const params = [V.W(0), V.W(0), V.W(5)]; // cx, cy, radius

// Generate noisy circle data
const points = Array.from({ length: 50 }, (_, i) => {
  const angle = (i / 50) * 2 * Math.PI;
  return {
    x: 10 * Math.cos(angle) + (Math.random() - 0.5) * 0.5,
    y: 10 * Math.sin(angle) + (Math.random() - 0.5) * 0.5,
  };
});

const result = V.nonlinearLeastSquares(
  params,
  ([cx, cy, r]) => {
    // Compute residual for each point (distance from circle)
    return points.map(p => {
      const dx = V.sub(p.x, cx);
      const dy = V.sub(p.y, cy);
      const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
      return V.sub(dist, r);
    });
  },
  {
    maxIterations: 100,
    costTolerance: 1e-6,
    verbose: true,
  }
);

console.log('Circle fitted in', result.iterations, 'iterations');
console.log('Center:', params[0].data, params[1].data);
console.log('Radius:', params[2].data);
```

The Levenberg-Marquardt algorithm typically converges 100-1000x faster than gradient descent for least squares problems.

## Choosing the Right Optimizer

ScalarAutograd provides three categories of optimizers, each suited for different problem types:

### 1. Gradient Descent Optimizers (SGD, Adam, AdamW)
**Best for:** Training neural networks, iterative refinement, online learning

```typescript
const opt = new Adam([w, b], { learningRate: 0.01 });
for (let epoch = 0; epoch < 1000; epoch++) {
  const loss = computeLoss();
  opt.zeroGrad();
  loss.backward();
  opt.step();
}
```

**Pros:** Simple, works on any differentiable objective, good for streaming data
**Cons:** Slow convergence, requires tuning learning rate and iterations

### 2. Levenberg-Marquardt (`V.nonlinearLeastSquares`)
**Best for:** Nonlinear least squares: minimizing Σ rᵢ(x)²

```typescript
const result = V.nonlinearLeastSquares(
  params,
  (p) => points.map(pt => residualFunction(p, pt)), // Returns array of residuals
  { maxIterations: 100 }
);
```

**Use when:**
- Problem is naturally formulated as sum of squared residuals
- You have overdetermined systems (more equations than unknowns)
- Examples: curve fitting, calibration, parameter estimation, circle/sphere fitting

**Pros:** 10-100x faster than gradient descent, exploits Jacobian structure
**Cons:** Only works for least squares problems, requires residual formulation

### 3. L-BFGS (`lbfgs`)
**Best for:** General unconstrained optimization

```typescript
import { lbfgs } from 'scalar-autograd';

const result = lbfgs(
  params,
  (p) => computeObjective(p), // Returns single Value (the cost)
  { maxIterations: 100 }
);
```

**Use when:**
- Objective has no special structure (not sum-of-squares)
- High-dimensional problems (100s-1000s of parameters)
- Memory constrained (stores only ~10 recent gradient pairs)
- Examples: energy minimization, ML losses, geometric optimization, developable surfaces

**Pros:** Memory efficient, handles non-quadratic objectives well, faster than gradient descent
**Cons:** Not as fast as LM for least squares, requires smooth objectives

### Quick Decision Guide

```
Can you write your objective as f(x) = Σ rᵢ(x)² ?
├─ YES → Use V.nonlinearLeastSquares() (Levenberg-Marquardt)
│         Fastest for curve fitting, calibration, parameter estimation
│
└─ NO → Is it a general smooth objective?
    ├─ YES → Use lbfgs() for large-scale or L-BFGS for efficiency
    │         Good for energy minimization, geometric optimization
    │
    └─ NO → Use Adam/AdamW for training neural networks
              Good for online learning, streaming data
```

## API Overview
- **Core Value construction:**
    - `V.C(data, label?)` — constant (non-differentiable), e.g. for data/inputs.
    - `V.W(data, label?)` — weight/parameter (differentiable).
- **Operators:**
    - Basic: `V.add(a, b)`, `V.sub(a, b)`, `V.mul(a, b)`, `V.div(a, b)`, `V.pow(a, n)`, `V.powValue(a, b)`.
    - Reductions: `V.sum(array)`, `V.mean(array)`
    - Trig: `V.sin(x)`, `V.cos(x)`, `V.tan(x)`, ...
    - Activations: `V.relu(x)`, `V.tanh(x)`, `V.sigmoid(x)`, etc.
    - Comparison: `V.eq(a, b)`, `V.gt(a, b)`, ... (outputs constant Values; never has grad)
- **Backward:**
    - `.backward()` — trigger automatic differentiation from this node.
    - `.grad` — access the computed gradient after backward pass.
- **Optimizers:**
    - `SGD`, `Adam`, `AdamW` - E.g. `const opt = new SGD([w, b], {learningRate: 0.01})`
- **Losses:**
    - `Losses.mse()`, `Losses.mae()`, `Losses.binaryCrossEntropy()`, `Losses.categoricalCrossEntropy()`, `Losses.huber()`, `Losses.tukey()`
- **Advanced Optimization:**
    - `V.nonlinearLeastSquares(params, residualFn, options)` — Levenberg-Marquardt solver for nonlinear least squares problems (minimizing Σ rᵢ²)
    - `lbfgs(params, objectiveFn, options)` — L-BFGS optimizer for general unconstrained optimization
- **Vector utilities:**
    - `Vec2`, `Vec3` — Differentiable 2D/3D vectors with dot, cross, normalize operations

All API operations work with both `Value` and raw number inputs (numbers are automatically wrapped as non-grad constants).


## Testing

To run the test suite and verify the correctness of ScalarAutograd, execute the following command in your project directory:

```shell
npm run test
```

## Deploy to npm

Start "Git Bash" terminal
Type ./release.sh

---

## License
MIT
