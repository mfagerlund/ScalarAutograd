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
    - E.g. `const opt = new SGD([w, b], {learningRate: 0.01})`
- **Losses:**
    - Import from `Losses.ts` (e.g. `import { mse } from './Losses'`)

All API operations work with both `Value` and raw number inputs (numbers are automatically wrapped as non-grad constants).

## Testing

To run the test suite and verify the correctness of ScalarAutograd, execute the following command in your project directory:

```shell
npm run test
```

## License
MIT
