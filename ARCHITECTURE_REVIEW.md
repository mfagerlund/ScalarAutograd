# ScalarAutograd Architecture Review

**Date**: October 2025
**Reviewer**: Architecture Assessment
**Project**: ScalarAutograd - Scalar-based Automatic Differentiation Library

## Executive Summary

ScalarAutograd is a well-designed TypeScript automatic differentiation library with a sophisticated JIT compilation system. The codebase demonstrates strong technical foundations but could benefit from improved organization, clearer API boundaries, and enhanced type safety. This review provides 10 actionable recommendations prioritized by impact and effort.

---

## 1. Directory Structure & Separation of Concerns

**Current State**: Flat structure with 24 files in `src/` mixing concerns (core engine, operations, compilation, optimization, utilities)

**Issues**:
- Difficult to navigate for new contributors
- Unclear module boundaries
- All files at same hierarchy level despite different responsibilities

**Recommended Structure**:

```
src/
├── core/              # Core autodiff engine
│   ├── Value.ts
│   ├── BackwardFn.ts
│   └── NoGrad.ts
├── operations/        # Operation implementations
│   ├── arithmetic.ts  (ValueArithmetic)
│   ├── trig.ts       (ValueTrig)
│   ├── activation.ts (ValueActivation)
│   └── comparison.ts (ValueComparison)
├── api/              # Public API surface
│   └── V.ts
├── compilation/      # JIT compilation system
│   ├── CompiledFunctions.ts
│   ├── KernelPool.ts
│   ├── ValueRegistry.ts
│   ├── GraphCanonicalizer*.ts
│   └── compileIndirectKernel.ts
├── optimizers/       # Optimization algorithms
│   ├── Optimizers.ts
│   ├── LBFGS.ts
│   └── NonlinearLeastSquares.ts
├── solvers/
│   └── LinearSolver.ts
├── losses/
│   └── Losses.ts
└── geometry/         # Vector utilities
    ├── Vec2.ts
    ├── Vec3.ts
    ├── Matrix3x3.ts
    └── Geometry.ts
```

**Benefits**:
- Clearer module boundaries
- Better navigation and discoverability
- Easier onboarding for new contributors
- Logical grouping for bundlers (tree-shaking potential)

**Migration Path**: Use `agent-refactor` for safe file moves with automatic import updates

---

## 2. API Design Consistency

**Current State**: Dual API surface creates confusion

Users have two ways to perform operations:
```typescript
// Instance methods
value.add(other)
value.mul(other)

// Static methods
V.add(a, b)
V.mul(a, b)
```

**Issues**:
- Unclear which style is "canonical"
- Inconsistent with `number` auto-conversion (instance methods auto-convert, static require explicit ensureValue)
- Duplicated documentation

**Recommendation**: Choose ONE primary API style

### Option A: Functional Style (Recommended)

Make `V` the primary API, keep instance methods as thin convenience wrappers:

```typescript
// Primary API - functional style
const result = V.add(a, b);
const loss = V.sum([V.pow(V.sub(pred, actual), 2)]);

// Instance methods delegate to V
class Value {
  add(other: Value | number): Value {
    return V.add(this, other);
  }

  mul(other: Value | number): Value {
    return V.mul(this, other);
  }
}
```

**Rationale**:
- Matches mathematical notation (functions operating on values)
- Consistent with compilation API (`V.compile`, `V.compileObjective`)
- Better for tree-shaking (unused operations can be eliminated)
- Easier to extend without modifying Value class

### Option B: Object-Oriented Style (Alternative)

Make Value instance methods primary, deprecate V operation methods (keep only factories):

```typescript
// Primary API - OOP style
const result = a.add(b).mul(c);

// V only for factories
const w = V.W(1.0);
const c = V.C(5.0);
const p = V.Param(2.0, "learning_rate");
```

**Impact**: Clearer API contract, reduced confusion, better IDE autocomplete

---

## 3. Type Safety Improvements

**Current State**: All Values are the same type, no compile-time distinction between constants, weights, and parameters

**Issues**:
- Can't enforce "only differentiable Values" at compile time
- Easy to accidentally pass constants where parameters expected
- Type system doesn't encode semantic meaning

**Recommended**: Branded types for different Value roles

```typescript
// Branded types using TypeScript's type system
type Constant = Value & { readonly __brand: 'constant' };
type Weight = Value & { readonly __brand: 'weight' };
type Parameter = Value & { readonly __brand: 'parameter' };

// Factory functions return branded types
class V {
  static C(value: number, label = ""): Constant {
    return new Value(value, label, false) as Constant;
  }

  static W(value: number, label = ""): Weight {
    return new Value(value, label, true) as Weight;
  }

  static Param(value: number, name: string): Parameter {
    const v = new Value(value, name, false);
    v.paramName = name;
    return v as Parameter;
  }
}

// Type-safe collection functions
function collectWeights(graph: Value): Weight[] {
  // Implementation
}

function collectParams(graph: Value): Parameter[] {
  // Implementation
}

// Optimizer functions can require specific types
function lbfgs(
  params: (Weight | Parameter)[],  // Only differentiable values
  objective: (params: Value[]) => Value,
  options?: LBFGSOptions
): LBFGSResult;
```

**Advanced**: Could use conditional types for operation results:
```typescript
// If both inputs are constants, result is constant
// If any input is a weight, result is a weight
type InferValueType<A, B> =
  A extends Weight ? Weight :
  B extends Weight ? Weight :
  Constant;

static add<A extends Value, B extends Value>(
  a: A,
  b: B
): InferValueType<A, B>;
```

**Benefits**:
- Catch errors at compile time
- Self-documenting code
- Better IDE support with specific type information
- No runtime overhead (types erased)

---

## 4. Consolidate Compilation API

**Current State**: Confusing naming and multiple entry points

Issues:
- `CompiledFunctions` vs `CompiledResiduals` (alias, but confusing)
- `V.compileObjective` wraps single output in array
- Users must understand when to use evaluateGradient vs evaluateSumWithGradient vs evaluateJacobian

**Recommended**: Single, unified compilation API

```typescript
class V {
  /**
   * Compile computation graph for efficient reuse.
   * Auto-detects single vs multi-output functions.
   */
  static compile(
    params: Value[],
    fn: (params: Value[]) => Value | Value[]
  ): CompiledGraph {
    const output = fn(params);
    const isScalar = !Array.isArray(output);

    return new CompiledGraph(
      params,
      isScalar ? [output] : output,
      { isScalar }
    );
  }
}

class CompiledGraph {
  private isScalar: boolean;

  // For single objective (L-BFGS, Adam, SGD)
  evaluateValue(params: Value[]): number {
    if (!this.isScalar) throw new Error("Use evaluateValues for multi-output");
    return this.evaluateGradient(params).value;
  }

  evaluateGradient(params: Value[]): { value: number; gradient: number[] } {
    if (!this.isScalar) throw new Error("Use evaluateJacobian for multi-output");
    // Implementation
  }

  // For multiple residuals (Levenberg-Marquardt)
  evaluateValues(params: Value[]): number[] {
    // Implementation
  }

  evaluateJacobian(params: Value[]): { values: number[]; jacobian: number[][] } {
    // Implementation
  }

  // Convenience for optimization
  evaluateSumWithGradient(params: Value[]): { value: number; gradient: number[] } {
    // Sum all outputs, accumulate gradients
  }

  // Metrics
  get kernelCount(): number;
  get kernelReuseFactor(): number;
}
```

**Usage Examples**:
```typescript
// Single objective optimization
const compiled = V.compile(params, (p) => lossFunction(p));
const result = lbfgs(params, compiled, { verbose: true });

// Least squares with residuals
const compiled = V.compile(params, (p) => residuals(p));
const result = nonlinearLeastSquares(params, compiled);
```

**Benefits**:
- Simpler mental model
- Auto-detection reduces user errors
- Clear method names indicate use case
- Single import point

---

## 5. Error Handling Strategy

**Current State**: Inline validation with generic `Error` class

```typescript
// Current pattern throughout codebase
if (a.data < 0) {
  throw new Error(`Cannot take sqrt of negative number: ${a.data}`);
}

if (Math.abs(b.data) < eps) {
  throw new Error(`Division by zero or near-zero encountered`);
}
```

**Issues**:
- Can't catch specific error types
- Hard to distinguish library errors from user errors
- No consistent error format
- Difficult to test error conditions

**Recommended**: Typed error hierarchy

```typescript
// src/core/errors.ts
export class AutogradError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AutogradError';
  }
}

export class DomainError extends AutogradError {
  constructor(operation: string, value: number, constraint: string) {
    super(`${operation} requires ${constraint}, got ${value}`);
    this.name = 'DomainError';
  }
}

export class GraphError extends AutogradError {
  constructor(message: string) {
    super(message);
    this.name = 'GraphError';
  }
}

export class CompilationError extends AutogradError {
  constructor(message: string) {
    super(message);
    this.name = 'CompilationError';
  }
}

export class ConvergenceError extends AutogradError {
  constructor(optimizer: string, reason: string) {
    super(`${optimizer} failed to converge: ${reason}`);
    this.name = 'ConvergenceError';
  }
}
```

**Usage**:
```typescript
// In ValueArithmetic.ts
static sqrt(a: Value): Value {
  if (a.data < 0) {
    throw new DomainError('sqrt', a.data, 'non-negative value');
  }
  // ...
}

// In Value.ts
backward(zeroGrad = false): void {
  if (this.prev.length === 0 && !this.requiresGrad) {
    throw new GraphError('Cannot call backward on constant with no dependencies');
  }
  // ...
}

// User code can catch specific errors
try {
  const result = V.sqrt(V.C(-1));
} catch (e) {
  if (e instanceof DomainError) {
    console.log('Invalid input to mathematical function');
  } else {
    throw e;
  }
}
```

**Benefits**:
- Better error filtering in user code
- Clearer error sources
- Easier to test error conditions
- Consistent error messages
- Can add error codes for i18n

---

## 6. Reduce Global State

**Current State**: Mutable global state in `Value.no_grad_mode`

```typescript
// Current implementation
class Value {
  static no_grad_mode = false;

  static withNoGrad<T>(fn: () => T): T {
    const prev = Value.no_grad_mode;
    Value.no_grad_mode = true;
    try {
      return fn();
    } finally {
      Value.no_grad_mode = prev;
    }
  }
}
```

**Issues**:
- Not thread-safe (if TS ever supports true threading)
- Difficult to test (need to ensure cleanup)
- Can't have multiple independent contexts
- Global mutation is error-prone

**Recommended**: Context-based approach

```typescript
// src/core/AutogradContext.ts
export class AutogradContext {
  private noGrad = false;

  withNoGrad<T>(fn: () => T): T {
    const prev = this.noGrad;
    this.noGrad = true;
    try {
      return fn();
    } finally {
      this.noGrad = prev;
    }
  }

  createValue(data: number, label = "", requiresGrad = false): Value {
    const actualRequiresGrad = this.noGrad ? false : requiresGrad;
    return new Value(data, label, actualRequiresGrad);
  }

  // Operations delegate to this context
  add(a: Value, b: Value): Value {
    const result = new Value(a.data + b.data);
    result.requiresGrad = !this.noGrad && (a.requiresGrad || b.requiresGrad);
    // ...
  }
}

// Default global context for convenience
export const defaultContext = new AutogradContext();

// V class uses default context
export class V {
  static context = defaultContext;

  static C(value: number, label = ""): Value {
    return V.context.createValue(value, label, false);
  }

  static withNoGrad<T>(fn: () => T): T {
    return V.context.withNoGrad(fn);
  }
}

// Advanced users can create isolated contexts
const ctx1 = new AutogradContext();
const ctx2 = new AutogradContext();

ctx1.withNoGrad(() => {
  // Only affects ctx1
});
```

**Benefits**:
- Thread-safe (future-proof)
- Testable (no global state)
- Composable (multiple contexts)
- Can add context-specific configuration (precision, device, etc.)

**Migration**: Keep static methods on Value for backward compatibility, delegate to default context

---

## 7. Operation Registration Pattern

**Current State**: Operations hardcoded in `Value.getForwardCode()` and `Value.getBackwardCode()`

```typescript
// Current: 600+ line switch statements
getForwardCode(childCodes: string[]): string {
  // ...
  switch (this._op) {
    case 'exp': return `Math.exp(${child})`;
    case 'log': return `Math.log(${child})`;
    case 'sqrt': return `Math.sqrt(${child})`;
    // ... 30+ cases
  }
}

getBackwardCode(...): string {
  // ... another 30+ cases
}
```

**Issues**:
- Can't add custom operations without modifying Value.ts
- Code generation logic mixed with Value class
- Hard to test individual operations
- No extensibility

**Recommended**: Registry pattern for operations

```typescript
// src/compilation/OperationRegistry.ts
export interface OperationDescriptor {
  opName: string;
  arity: number;  // 1 for unary, 2 for binary
  forwardCode: (childCodes: string[]) => string;
  backwardCode: (
    gradVar: string,
    childGrads: string[],
    childVars: string[]
  ) => string;
}

export class OperationRegistry {
  private ops = new Map<string, OperationDescriptor>();

  register(desc: OperationDescriptor): void {
    this.ops.set(desc.opName, desc);
  }

  getForward(op: string, childCodes: string[]): string | null {
    const desc = this.ops.get(op);
    return desc ? desc.forwardCode(childCodes) : null;
  }

  getBackward(
    op: string,
    gradVar: string,
    childGrads: string[],
    childVars: string[]
  ): string | null {
    const desc = this.ops.get(op);
    return desc ? desc.backwardCode(gradVar, childGrads, childVars) : null;
  }
}

// Global registry
export const defaultRegistry = new OperationRegistry();

// Register built-in operations
defaultRegistry.register({
  opName: 'exp',
  arity: 1,
  forwardCode: ([child]) => `Math.exp(${child})`,
  backwardCode: (gradVar, [childGrad], [child]) =>
    `${childGrad} += ${gradVar} * Math.exp(${child});`
});

defaultRegistry.register({
  opName: 'sin',
  arity: 1,
  forwardCode: ([child]) => `Math.sin(${child})`,
  backwardCode: (gradVar, [childGrad], [child]) =>
    `${childGrad} += ${gradVar} * Math.cos(${child});`
});

// ... register all operations

// Value class delegates to registry
class Value {
  static registry = defaultRegistry;

  getForwardCode(childCodes: string[]): string {
    if (this.paramName) return this.paramName;

    const code = Value.registry.getForward(this._op!, childCodes);
    return code ?? String(this.data);
  }

  getBackwardCode(gradVar: string, childGrads: string[], childVars: string[]): string {
    const code = Value.registry.getBackward(this._op!, gradVar, childGrads, childVars);
    return code ?? '';
  }
}
```

**Custom Operations**:
```typescript
// User can register custom operations
import { V, defaultRegistry } from 'scalar-autograd';

defaultRegistry.register({
  opName: 'swish',  // Custom activation: x * sigmoid(x)
  arity: 1,
  forwardCode: ([x]) => `${x} * (1 / (1 + Math.exp(-${x})))`,
  backwardCode: (grad, [childGrad], [x]) => {
    const sig = `(1 / (1 + Math.exp(-${x})))`;
    return `${childGrad} += ${grad} * (${sig} + ${x} * ${sig} * (1 - ${sig}));`;
  }
});

// Add to ValueActivation
export class ValueActivation {
  static swish(x: Value): Value {
    const sigmoid = 1 / (1 + Math.exp(-x.data));
    const data = x.data * sigmoid;
    return Value.make(
      data,
      x, null,
      (out) => () => {
        const sig = 1 / (1 + Math.exp(-x.data));
        x.grad += out.grad * (sig + x.data * sig * (1 - sig));
      },
      `swish(${x.label})`,
      'swish'
    );
  }
}
```

**Benefits**:
- Extensible without modifying core
- Operations self-contained
- Easier to test
- Plugin architecture
- Can load operations dynamically

---

## 8. Package Exports Organization

**Current State**: Single entry point exports everything

```json
// package.json
"exports": {
  ".": {
    "types": "./dist/index.d.ts",
    "default": "./dist/index.js"
  }
}
```

```typescript
// src/index.ts - exports everything
export { Value } from './Value';
export { V } from './V';
export { CompiledFunctions } from './CompiledFunctions';
export { Geometry } from './Geometry';
export { Matrix3x3 } from './Matrix3x3';
// ... 20+ exports
```

**Issues**:
- Large bundle size for simple use cases
- No clear separation between core API and advanced features
- Everything imported regardless of usage

**Recommended**: Layered exports with subpath exports

```json
// package.json
"exports": {
  ".": {
    "types": "./dist/index.d.ts",
    "default": "./dist/index.js"
  },
  "./core": {
    "types": "./dist/core/index.d.ts",
    "default": "./dist/core/index.js"
  },
  "./compilation": {
    "types": "./dist/compilation/index.d.ts",
    "default": "./dist/compilation/index.js"
  },
  "./optimizers": {
    "types": "./dist/optimizers/index.d.ts",
    "default": "./dist/optimizers/index.js"
  },
  "./geometry": {
    "types": "./dist/geometry/index.d.ts",
    "default": "./dist/geometry/index.js"
  }
}
```

```typescript
// dist/index.ts - Main API (most users only need this)
export { V } from './api/V';
export { Value } from './core/Value';
export type { BackwardFn } from './core/Value';

// dist/optimizers/index.ts
export { lbfgs, type LBFGSOptions, type LBFGSResult } from './LBFGS';
export { nonlinearLeastSquares, type NonlinearLeastSquaresOptions } from './NonlinearLeastSquares';
export { Optimizer, SGD, Adam, AdamW, type OptimizerOptions } from './Optimizers';

// dist/compilation/index.ts
export { CompiledFunctions } from './CompiledFunctions';
export { KernelPool } from './KernelPool';
export { ValueRegistry } from './ValueRegistry';

// dist/geometry/index.ts
export { Vec2 } from './Vec2';
export { Vec3 } from './Vec3';
export { Matrix3x3 } from './Matrix3x3';
export { Geometry } from './Geometry';
```

**Usage**:
```typescript
// Basic usage - small bundle
import { V, Value } from 'scalar-autograd';

const x = V.W(1.0);
const y = V.add(x, V.C(2.0));

// Optimization - load only what's needed
import { V } from 'scalar-autograd';
import { lbfgs } from 'scalar-autograd/optimizers';

const result = lbfgs(params, objective, { verbose: true });

// Advanced compilation features
import { V } from 'scalar-autograd';
import { CompiledFunctions } from 'scalar-autograd/compilation';

const compiled = CompiledFunctions.compile(params, residualFn);

// Geometry utilities
import { Vec2, Vec3 } from 'scalar-autograd/geometry';
```

**Benefits**:
- Smaller bundles (tree-shaking more effective)
- Clearer dependency graph
- Progressive disclosure (learn core first, then advanced)
- Easier to identify what features are actually used

---

## 9. Testing Infrastructure Improvements

**Current State**: Good test coverage but could be more systematic

**Recommended Additions**:

### Test Fixtures
```typescript
// test/fixtures/test-graphs.ts
export const TestGraphs = {
  rosenbrock: (x: Value, y: Value) => {
    const a = V.sub(V.C(1), x);
    const b = V.sub(y, V.square(x));
    return V.add(V.square(a), V.mul(V.C(100), V.square(b)));
  },

  rastrigin: (vars: Value[]) => {
    const A = 10;
    const n = vars.length;
    let sum = V.C(A * n);
    for (const x of vars) {
      const term = V.sub(
        V.square(x),
        V.mul(V.C(A), V.cos(V.mul(V.C(2 * Math.PI), x)))
      );
      sum = V.add(sum, term);
    }
    return sum;
  },

  quadratic: (x: Value, a: number, b: number, c: number) => {
    return V.add(
      V.add(V.mul(V.C(a), V.square(x)), V.mul(V.C(b), x)),
      V.C(c)
    );
  }
};
```

### Custom Test Assertions
```typescript
// test/helpers/assertions.ts
import { expect } from 'vitest';

export function expectGradient(
  value: Value,
  params: Value[],
  expected: number[],
  tolerance = 1e-6
) {
  Value.zeroGradTree(value);
  value.backward();

  const actual = params.map(p => p.grad);

  expect(actual).toHaveLength(expected.length);
  actual.forEach((grad, i) => {
    expect(Math.abs(grad - expected[i])).toBeLessThan(tolerance);
  });
}

export function expectValueClose(
  actual: number,
  expected: number,
  tolerance = 1e-6,
  message?: string
) {
  expect(Math.abs(actual - expected)).toBeLessThan(tolerance);
}

export function expectCompiles(graph: Value, params: Value[]) {
  expect(() => {
    CompiledFunctions.compile(params, () => [graph]);
  }).not.toThrow();
}
```

### Numerical Gradient Testing
```typescript
// test/helpers/numerical-gradient.ts
export function numericalGradient(
  fn: (params: Value[]) => Value,
  params: Value[],
  h = 1e-5
): number[] {
  const gradients: number[] = [];

  for (let i = 0; i < params.length; i++) {
    const original = params[i].data;

    params[i].data = original + h;
    const fPlus = fn(params).data;

    params[i].data = original - h;
    const fMinus = fn(params).data;

    params[i].data = original;

    gradients.push((fPlus - fMinus) / (2 * h));
  }

  return gradients;
}

// Usage in tests
it('should compute correct gradient', () => {
  const x = V.W(2.0);
  const y = V.W(3.0);
  const params = [x, y];

  const fn = (p: Value[]) => V.add(V.square(p[0]), V.square(p[1]));

  // Analytical gradient
  const output = fn(params);
  Value.zeroGradTree(output);
  output.backward();
  const analytical = params.map(p => p.grad);

  // Numerical gradient
  const numerical = numericalGradient(fn, params);

  expect(analytical[0]).toBeCloseTo(numerical[0], 5);
  expect(analytical[1]).toBeCloseTo(numerical[1], 5);
});
```

### Property-Based Testing
```typescript
// test/properties/gradient-properties.spec.ts
import { expect, it } from 'vitest';
import { V, Value } from '../src';

function randomValue(min = -10, max = 10): Value {
  return V.W(Math.random() * (max - min) + min);
}

it('gradient of constant should always be zero', () => {
  for (let i = 0; i < 100; i++) {
    const c = V.C(Math.random() * 100);
    const output = V.mul(c, V.C(2));

    Value.zeroGradTree(output);
    output.backward();

    expect(c.grad).toBe(0);
  }
});

it('gradient should be linear in output gradient', () => {
  for (let i = 0; i < 100; i++) {
    const x = randomValue();
    const original = x.data;

    const y = V.exp(x);
    Value.zeroGradTree(y);
    y.grad = 1;
    y.backward();
    const grad1 = x.grad;

    // Reset
    x.data = original;
    x.grad = 0;

    const y2 = V.exp(x);
    Value.zeroGradTree(y2);
    y2.grad = 2;
    y2.backward();
    const grad2 = x.grad;

    expect(grad2).toBeCloseTo(2 * grad1, 5);
  }
});
```

---

## 10. Documentation Architecture

**Current State**: API documentation via TSDoc, README.md, CLAUDE.md

**Recommended Structure**:

```
docs/
├── README.md                    # Overview, quick start
├── architecture/
│   ├── computation-graph.md     # How autodiff works
│   ├── jit-compilation.md       # Compilation system design
│   ├── kernel-reuse.md          # How kernel pooling works
│   └── operation-lifecycle.md   # How operations are created
├── guides/
│   ├── getting-started.md       # Tutorial
│   ├── optimization-workflows.md # L-BFGS, LM, etc.
│   ├── custom-operations.md     # Extending the library
│   ├── compilation-guide.md     # When and how to compile
│   └── debugging-gradients.md   # Common issues
├── examples/
│   ├── curve-fitting.md
│   ├── neural-networks.md
│   ├── robotics-ik.md
│   └── geometry-optimization.md
└── api/
    └── (generated from TSDoc via typedoc)
```

### Example Documentation Improvements

**docs/guides/getting-started.md**:
```markdown
# Getting Started with ScalarAutograd

## Installation
npm install scalar-autograd

## Your First Gradient

import { V } from 'scalar-autograd';

// Create a differentiable variable
const x = V.W(3.0);

// Build computation graph
const y = V.add(V.square(x), V.C(5));  // y = x² + 5

// Compute gradient
y.backward();
console.log(x.grad);  // 6.0 (dy/dx = 2x = 2*3)

## Optimization Example

const x = V.W(5.0);
const target = 10.0;

for (let i = 0; i < 100; i++) {
  const pred = V.mul(x, V.C(2));
  const loss = V.square(V.sub(pred, V.C(target)));

  Value.zeroGradTree(loss);
  loss.backward();

  x.data -= 0.1 * x.grad;
}

console.log(x.data);  // ≈ 5.0 (because 2*5 = 10)
```

**docs/architecture/kernel-reuse.md**:
```markdown
# Kernel Reuse Architecture

## Problem

When optimizing geometry with 1000s of residuals that share the same
structure (e.g., distance constraints), naïve compilation would create
1000 separate JavaScript functions, wasting memory and compilation time.

## Solution

ScalarAutograd uses graph canonicalization to detect structurally
identical computations and reuse compiled kernels:

1. **Graph Signature**: Compute canonical string from graph topology
2. **Kernel Pool**: Store kernels by signature hash
3. **Input Remapping**: Map each residual's inputs to shared kernel

[Diagrams and detailed explanation]
```

**Benefits**:
- Easier onboarding
- Better discoverability
- Reference for design decisions
- Examples for common use cases

---

## Priority Ranking

### High Impact, Low Effort
1. **#4 Consolidate Compilation API** - Simplify user-facing API
2. **#5 Error Handling Strategy** - Better error messages
3. **#2 API Design Consistency** - Choose primary style

### High Impact, Medium Effort
4. **#1 Directory Structure** - Organize files logically
5. **#7 Operation Registration** - Extensibility pattern
6. **#8 Package Exports** - Better tree-shaking

### Medium Impact, Higher Effort
7. **#3 Type Safety Improvements** - Branded types
8. **#6 Reduce Global State** - Context-based approach
9. **#9 Testing Infrastructure** - Systematic testing
10. **#10 Documentation Architecture** - Comprehensive docs

---

## Migration Strategy

### Phase 1: Internal Refactoring (Non-Breaking)
**Timeline**: 1-2 weeks

- Implement error hierarchy (#5)
- Reorganize directory structure (#1)
- Add operation registry (#7) alongside existing code
- Improve test infrastructure (#9)

**Release**: Patch version (0.1.9)

### Phase 2: API Consolidation (Deprecations)
**Timeline**: 2-3 weeks

- Add `CompiledGraph` class alongside existing (#4)
- Add branded types as optional (#3)
- Implement subpath exports (#8)
- Mark old APIs as `@deprecated` with migration guides
- Add context-based approach (#6) with backward compatibility

**Release**: Minor version (0.2.0)

### Phase 3: Breaking Changes (Major Version)
**Timeline**: 1-2 weeks

- Remove deprecated APIs
- Choose primary API style (#2)
- Remove global state (#6)
- Clean up legacy code

**Release**: Major version (1.0.0)

### Phase 4: Advanced Features
**Timeline**: Ongoing

- Custom operations via registry (#7)
- Enhanced type inference (#3)
- Advanced documentation (#10)

**Release**: Minor versions (1.1.0, 1.2.0, etc.)

---

## Conclusion

ScalarAutograd has a solid foundation with sophisticated JIT compilation and kernel reuse. These recommendations focus on:

1. **Developer Experience**: Clearer APIs, better errors, organized structure
2. **Type Safety**: Catch errors at compile time
3. **Extensibility**: Allow users to add custom operations
4. **Performance**: Better tree-shaking, smaller bundles

The phased approach allows gradual improvement without breaking existing users, culminating in a clean 1.0.0 release.
