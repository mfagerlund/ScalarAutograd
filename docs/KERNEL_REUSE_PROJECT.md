# KernelReuse Project

## Problem Statement

Currently, when compiling residual functions for Jacobian computation, **each residual gets its own unique compiled kernel**. This is wasteful because many residuals have identical computational structure (topology) but operate on different input values.

### Example

In a sketch solver with 100 distance constraints:
- Each constraint computes: `sqrt((x1-x2)² + (y1-y2)²) - target`
- Today: 100 unique compiled kernels (one per residual)
- Optimal: 1 shared kernel called 100 times with different indices

Current code in `CompiledResiduals.compile()`:
```typescript
const compiledFunctions = residualValues.map((r, i) =>
  compileResidualJacobian(r, params, i)  // Creates unique kernel per residual
);
```

## Current Architecture

### Compilation Flow

1. **CompiledResiduals.compile()**: Takes params and residualFn
2. Calls residualFn(params) to build computation graph
3. For each residual Value:
   - Traverses graph topologically
   - Generates unique forward/backward code
   - Compiles to JavaScript function via `new Function(...)`
   - Each kernel is: `(paramValues: number[], row: number[]) => number`

### Graph Structure

Each Value node contains:
- `data: number` - current value
- `prev: Value[]` - parent nodes in computation graph
- `_op: string` - operation type ('+', '*', 'exp', 'sin', etc.)
- `paramName?: string` - identifies input parameters
- `requiresGrad: boolean` - whether to track gradients

### Code Generation

`compileResidualJacobian()` in `jit-compile-value.ts`:
- Assigns variable names to nodes (`_v0`, `_v1`, etc.)
- Generates forward pass code using `getForwardCode()`
- Generates backward pass code using `getBackwardCode()`
- Produces JavaScript function string

Example compiled output:
```javascript
function(paramValues, row) {
  const p0 = paramValues[0];
  const p1 = paramValues[1];
  const _v0 = (p0 + p1);
  const _v1 = (_v0 * 2.0);

  let grad__v1 = 0;
  let grad__v0 = 0;
  let grad_p0 = 0;
  let grad_p1 = 0;

  grad__v1 = 1;
  grad__v0 += grad__v1 * 2.0;
  grad_p0 += grad__v0;
  grad_p1 += grad__v0;

  row[0] = grad_p0;
  row[1] = grad_p1;
  return _v1;
}
```

## Proposed Solution

### Core Idea: Graph Canonicalization + Value Registry

Instead of compiling unique kernels per residual, we:

1. **Extract all unique Values** from all residual graphs into a global registry
2. **Assign unique IDs** to each Value (stored on Value object)
3. **Canonicalize graph structure** to identify topologically identical graphs
4. **Share kernels** between graphs with identical structure
5. **Pass input indices** to kernels instead of hardcoding parameter positions

### Architecture Changes

#### 1. Value Registry (New)

```typescript
class ValueRegistry {
  private values: Value[] = [];
  private valueToId = new Map<Value, number>();

  register(value: Value): number {
    // Constants: reuse if same value
    if (!value.requiresGrad && value.prev.length === 0) {
      const existing = this.values.find(v =>
        !v.requiresGrad && v.prev.length === 0 && v.data === value.data
      );
      if (existing) return this.valueToId.get(existing)!;
    }

    // Variables/computed: always unique
    const id = this.values.length;
    this.values.push(value);
    this.valueToId.set(value, id);
    value._registryId = id;  // Store ID on Value
    return id;
  }

  getValues(): number[] {
    return this.values.map(v => v.data);
  }
}
```

#### 2. Graph Canonicalization (New)

```typescript
interface GraphSignature {
  operations: string[];    // Ordered list: ['add', 'mul', 'sub']
  topology: number[][];    // Parent indices: [[0,1], [2,3], ...]
  hash: string;           // Fast lookup key
}

function canonicalizeGraph(output: Value, registry: ValueRegistry): GraphSignature {
  const visited = new Set<Value>();
  const topoOrder: Value[] = [];

  function traverse(node: Value) {
    if (visited.has(node)) return;
    visited.add(node);
    for (const child of node.prev) traverse(child);
    topoOrder.push(node);
  }

  traverse(output);

  // Build canonical representation
  const operations = topoOrder.map(n => n._op || 'const');
  const topology = topoOrder.map(n =>
    n.prev.map(p => topoOrder.indexOf(p))
  );

  // Create hash for fast lookup
  const hash = JSON.stringify({ operations, topology });

  return { operations, topology, hash };
}
```

#### 3. Kernel Pool (New)

```typescript
interface KernelDescriptor {
  signature: GraphSignature;
  kernel: (values: number[], indices: number[], row: number[]) => number;
  outputIndex: number;
  paramIndices: number[];
}

class KernelPool {
  private kernels = new Map<string, KernelDescriptor>();

  getOrCompile(
    output: Value,
    params: Value[],
    registry: ValueRegistry
  ): KernelDescriptor {
    const signature = canonicalizeGraph(output, registry);

    if (this.kernels.has(signature.hash)) {
      return this.kernels.get(signature.hash)!;
    }

    // Compile new kernel with indirect indexing
    const kernel = compileIndirectKernel(output, params, registry, signature);
    const descriptor: KernelDescriptor = {
      signature,
      kernel,
      outputIndex: registry.valueToId.get(output)!,
      paramIndices: params.map(p => registry.valueToId.get(p)!)
    };

    this.kernels.set(signature.hash, descriptor);
    return descriptor;
  }
}
```

#### 4. Modified Kernel Signature

Old:
```typescript
(paramValues: number[], row: number[]) => number
```

New:
```typescript
(
  allValues: number[],      // Global value array
  indices: number[],        // Which values this graph uses
  row: number[]            // Jacobian row output
) => number
```

#### 5. Modified CompiledResiduals

```typescript
class CompiledResiduals {
  private registry: ValueRegistry;
  private kernelPool: KernelPool;
  private residualDescriptors: Array<{
    kernelHash: string;
    inputIndices: number[];
    outputIndex: number;
  }>;

  static compile(params: Value[], residualFn: (params: Value[]) => Value[]): CompiledResiduals {
    const registry = new ValueRegistry();
    const kernelPool = new KernelPool();

    // Register all params first
    params.forEach((p, i) => {
      if (!p.paramName) p.paramName = `p${i}`;
      registry.register(p);
    });

    // Build residuals
    const residualValues = residualFn(params);

    // Register all Values and compile kernels
    const residualDescriptors = residualValues.map(r => {
      // Register all nodes in this graph
      registerGraphNodes(r, registry);

      // Get or compile kernel
      const descriptor = kernelPool.getOrCompile(r, params, registry);

      // Extract input indices for this specific residual
      const inputIndices = extractInputIndices(r, registry);

      return {
        kernelHash: descriptor.signature.hash,
        inputIndices,
        outputIndex: registry.valueToId.get(r)!
      };
    });

    return new CompiledResiduals(registry, kernelPool, residualDescriptors, params.length);
  }

  evaluate(params: Value[]): { residuals: number[]; J: number[][]; cost: number } {
    // Update value array with current param values
    const values = this.registry.getValues();
    params.forEach((p, i) => {
      values[this.registry.valueToId.get(p)!] = p.data;
    });

    const numResiduals = this.residualDescriptors.length;
    const residuals: number[] = new Array(numResiduals);
    const J: number[][] = Array(numResiduals).fill(0).map(() =>
      new Array(params.length).fill(0)
    );
    let cost = 0;

    for (let i = 0; i < numResiduals; i++) {
      const desc = this.residualDescriptors[i];
      const kernel = this.kernelPool.kernels.get(desc.kernelHash)!.kernel;

      const value = kernel(values, desc.inputIndices, J[i]);
      cost += value * value;
      residuals[i] = value;
    }

    return { residuals, J, cost };
  }
}
```

## Implementation Strategy

### Phase 1: Registry + Indirect Kernel Compilation

**Goal**: Create value registry and compile kernels with indirect indexing (still unique per residual)

1. Add `_registryId?: number` to Value class
2. Implement ValueRegistry class with deduplication rules:
   - Constants: dedupe by value only (ignore labels)
   - Variables with same paramName: investigate if they should be same
   - Weights: always unique
3. Implement `compileIndirectKernel()` - new function that:
   - Takes registry and builds index mappings
   - Generates code using `allValues[indices[i]]` instead of direct params
   - Returns: `(allValues: number[], indices: number[], row: number[]) => number`
4. Create new test suite in `test/kernel-reuse/` - ignore existing tests
5. **Test Strategy**: Validate compiled kernels match graph backward pass
   - Run compiled kernel → get Jacobian row
   - Run graph.backward() → get gradients
   - Compare: Jacobian values must match gradient values

### Phase 2: Graph Canonicalization

**Goal**: Identify identical graph structures

1. Implement `canonicalizeGraph()` function
2. Add hash generation for graph signatures
3. Add logging to show kernel reuse opportunities
4. Test with known identical/different graph pairs
5. **Test**: Verify correct graph matching with edge cases

### Phase 3: Kernel Pool + Reuse

**Goal**: Share kernels across residuals with identical topology

1. Implement KernelPool class
2. Modify CompiledResiduals to use registry + pool
3. Add metrics: kernels compiled vs residuals
4. **Test**: Verify kernel reuse works correctly with complex examples

### Phase 4: Optimization & Refinement

**Goal**: Performance tuning and production readiness

1. Optimize hash generation (consider faster hashing)
2. Add compilation cache statistics
3. Investigate variable deduplication edge cases
4. Consider partial canonicalization (balance reuse vs lookup cost)
5. **Benchmark**: Measure speedup in IK and sketch scenarios

## Test Plan

### Unit Tests

**test/value-registry.spec.ts**
```typescript
describe('ValueRegistry', () => {
  it('should assign unique IDs to values', () => {
    const reg = new ValueRegistry();
    const v1 = V.W(1.0);
    const v2 = V.W(2.0);

    expect(reg.register(v1)).toBe(0);
    expect(reg.register(v2)).toBe(1);
    expect(v1._registryId).toBe(0);
    expect(v2._registryId).toBe(1);
  });

  it('should reuse constant values', () => {
    const reg = new ValueRegistry();
    const c1 = V.C(5.0);
    const c2 = V.C(5.0);
    const c3 = V.C(3.0);

    const id1 = reg.register(c1);
    const id2 = reg.register(c2);
    const id3 = reg.register(c3);

    expect(id1).toBe(id2); // Same constant reused
    expect(id1).not.toBe(id3); // Different constant
  });

  it('should not reuse variables/weights', () => {
    const reg = new ValueRegistry();
    const w1 = V.W(1.0);
    const w2 = V.W(1.0);

    expect(reg.register(w1)).not.toBe(reg.register(w2));
  });
});
```

**test/graph-canonicalization.spec.ts**
```typescript
describe('Graph Canonicalization', () => {
  it('should match identical graphs', () => {
    const reg = new ValueRegistry();
    const a0 = V.W(1, 'a0');
    const b0 = V.W(2, 'b0');
    const a1 = V.W(3, 'a1');
    const b1 = V.W(4, 'b1');

    reg.register(a0);
    reg.register(b0);
    reg.register(a1);
    reg.register(b1);

    const graph1 = V.add(a0, b0);
    const graph2 = V.add(a1, b1);

    const sig1 = canonicalizeGraph(graph1, reg);
    const sig2 = canonicalizeGraph(graph2, reg);

    expect(sig1.hash).toBe(sig2.hash);
  });

  it('should distinguish different operations', () => {
    const reg = new ValueRegistry();
    const a = V.W(1);
    const b = V.W(2);

    const add = V.add(a, b);
    const mul = V.mul(a, b);

    const sig1 = canonicalizeGraph(add, reg);
    const sig2 = canonicalizeGraph(mul, reg);

    expect(sig1.hash).not.toBe(sig2.hash);
  });

  it('should distinguish different topologies', () => {
    const reg = new ValueRegistry();
    const a = V.W(1);
    const b = V.W(2);
    const c = V.W(3);

    const graph1 = V.mul(V.add(a, b), c);  // (a+b)*c
    const graph2 = V.mul(a, V.add(b, c));  // a*(b+c)

    const sig1 = canonicalizeGraph(graph1, reg);
    const sig2 = canonicalizeGraph(graph2, reg);

    expect(sig1.hash).not.toBe(sig2.hash);
  });

  it('should match same topology with different inputs', () => {
    const reg = new ValueRegistry();
    const a0 = V.W(1);
    const b0 = V.W(2);
    const c0 = V.W(3);
    const a1 = V.W(4);
    const b1 = V.W(5);
    const c1 = V.W(6);

    const graph1 = V.mul(V.add(a0, b0), c0);
    const graph2 = V.mul(V.add(a1, b1), c1);

    const sig1 = canonicalizeGraph(graph1, reg);
    const sig2 = canonicalizeGraph(graph2, reg);

    expect(sig1.hash).toBe(sig2.hash);
  });
});
```

**test/kernel-pool.spec.ts**
```typescript
describe('KernelPool', () => {
  it('should reuse kernels for identical graphs', () => {
    const pool = new KernelPool();
    const reg = new ValueRegistry();

    const a0 = V.W(1, 'a0');
    const b0 = V.W(2, 'b0');
    const a1 = V.W(3, 'a1');
    const b1 = V.W(4, 'b1');

    [a0, b0, a1, b1].forEach(v => reg.register(v));

    const graph1 = V.add(a0, b0);
    const graph2 = V.add(a1, b1);

    const desc1 = pool.getOrCompile(graph1, [a0, b0], reg);
    const desc2 = pool.getOrCompile(graph2, [a1, b1], reg);

    expect(pool.size).toBe(1); // Only one kernel compiled
    expect(desc1.kernel).toBe(desc2.kernel); // Same kernel function
  });

  it('should compile different kernels for different graphs', () => {
    const pool = new KernelPool();
    const reg = new ValueRegistry();

    const a = V.W(1);
    const b = V.W(2);

    const add = V.add(a, b);
    const mul = V.mul(a, b);

    pool.getOrCompile(add, [a, b], reg);
    pool.getOrCompile(mul, [a, b], reg);

    expect(pool.size).toBe(2);
  });
});
```

**test/kernel-reuse-integration.spec.ts**
```typescript
describe('Kernel Reuse Integration', () => {
  it('should reuse kernels in distance constraints', () => {
    const points: Value[] = [];
    for (let i = 0; i < 20; i++) {
      points.push(V.W(i, `x${i}`));
      points.push(V.W(i + 0.5, `y${i}`));
    }

    // 10 distance constraints (all identical structure)
    const residuals = (params: Value[]) => {
      const res: Value[] = [];
      for (let i = 0; i < 10; i++) {
        const idx = i * 2;
        const x1 = params[idx];
        const y1 = params[idx + 1];
        const x2 = params[idx + 2];
        const y2 = params[idx + 3];

        const dx = V.sub(x1, x2);
        const dy = V.sub(y1, y2);
        const distSq = V.add(V.square(dx), V.square(dy));
        const dist = V.sqrt(distSq);

        res.push(V.sub(dist, V.C(1.0))); // Target distance = 1
      }
      return res;
    };

    const compiled = CompiledResiduals.compile(points, residuals);

    // Verify kernel reuse
    expect(compiled.kernelCount).toBe(1); // All use same kernel
    expect(compiled.numResiduals).toBe(10);

    // Verify correct evaluation
    const result = compiled.evaluate(points);
    expect(result.residuals).toHaveLength(10);
    expect(result.J).toHaveLength(10);
    expect(result.J[0]).toHaveLength(20);
  });

  it('should handle mixed kernel types', () => {
    const a = V.W(1);
    const b = V.W(2);
    const c = V.W(3);
    const params = [a, b, c];

    const residuals = (p: Value[]) => [
      V.add(p[0], p[1]),      // Type 1: addition
      V.add(p[1], p[2]),      // Type 1: addition (reuse kernel)
      V.mul(p[0], p[2]),      // Type 2: multiplication
      V.add(p[0], p[2]),      // Type 1: addition (reuse kernel)
    ];

    const compiled = CompiledResiduals.compile(params, residuals);

    expect(compiled.kernelCount).toBe(2); // 2 unique kernels
    expect(compiled.numResiduals).toBe(4);
  });
});
```

### Benchmark Tests

**test/kernel-reuse.benchmark.spec.ts**
```typescript
describe('Kernel Reuse Benchmarks', () => {
  it('should benchmark sketch solver with kernel reuse', () => {
    // 100 parallel line constraints (identical structure)
    const numLines = 100;
    const points: Value[] = [];

    for (let i = 0; i < numLines * 4; i++) {
      points.push(V.W(Math.random() * 100, `p${i}`));
    }

    const residuals = (params: Value[]) => {
      const res: Value[] = [];
      for (let i = 0; i < numLines; i++) {
        const idx = i * 4;
        const l1Start = new Vec2(params[idx], params[idx + 1]);
        const l1End = new Vec2(params[idx + 2], params[idx + 3]);
        const l2Start = new Vec2(params[idx + 4], params[idx + 5]);
        const l2End = new Vec2(params[idx + 6], params[idx + 7]);

        const dir1 = l1End.sub(l1Start);
        const dir2 = l2End.sub(l2Start);
        const cross = Vec2.cross(dir1, dir2);

        res.push(cross);
      }
      return res;
    };

    const compiled = CompiledResiduals.compile(points, residuals);

    console.log(`Kernels compiled: ${compiled.kernelCount}`);
    console.log(`Residuals: ${compiled.numResiduals}`);
    console.log(`Kernel reuse factor: ${compiled.numResiduals / compiled.kernelCount}x`);

    // Expected: ~1 kernel for 100 residuals
    expect(compiled.kernelCount).toBeLessThan(5);
  });
});
```

## Tradeoffs & Considerations

### Canonicalization Cost vs Reuse Benefit

**Tradeoff**: Time spent finding matching kernels vs. kernel reuse savings

**Analysis**:
- **IK problems**: Few residuals, likely unique topologies → kernel reuse minimal
- **Mesh/sketch**: Many residuals, repetitive patterns → high reuse potential

**Strategy**:
- Simple hash-based lookup (very fast)
- Don't optimize for partial matches initially
- Accept some redundant kernels if it avoids expensive matching

### Hash Collisions

**Risk**: Different graphs produce same hash

**Mitigation**:
- Use JSON.stringify for initial implementation (slow but correct)
- Later: Custom hash with collision detection
- Verify kernel correctness in tests

### Memory vs Computation

**Current**: N residuals × M bytes per kernel function
**Proposed**: K unique kernels × M bytes + N × small descriptor

**Expected savings**:
- Sketch with 1000 distance constraints: 1000 kernels → ~10 unique
- Memory reduction: ~99%

### Value Deduplication Rules (UPDATED)

**Constants** (requiresGrad=false, no prev):
- Dedupe by value only
- Labels/paramNames ignored (truly constant = interchangeable)
- Example: `V.C(5.0, "width")` and `V.C(5.0, "height")` → same registry entry

**Variables** (requiresGrad=true, has paramName):
- **TBD**: If same paramName, are they the same Value?
- Need to investigate with test variants
- Hypothesis: Same paramName → same variable (e.g., reused parameter)
- Counter-hypothesis: Always unique even with same name

**Weights** (requiresGrad=true, no paramName or explicit weight):
- Always unique
- Even if same data value, different instances

**Implementation**:
```typescript
register(value: Value): number {
  // Constants: dedupe by value only
  if (!value.requiresGrad && value.prev.length === 0) {
    const existing = this.values.find(v =>
      !v.requiresGrad && v.prev.length === 0 && v.data === value.data
    );
    if (existing) return this.valueToId.get(existing)!;
  }

  // Variables: investigate if paramName match means same value
  // TODO: Test this assumption
  if (value.requiresGrad && value.paramName) {
    const existing = this.values.find(v =>
      v.requiresGrad && v.paramName === value.paramName
    );
    if (existing) return this.valueToId.get(existing)!;
  }

  // Weights & computed values: always unique
  const id = this.values.length;
  this.values.push(value);
  this.valueToId.set(value, id);
  value._registryId = id;
  return id;
}
```

## Success Metrics

### Kernel Reuse Rate
- **Target**: >90% reuse in sketch solver (1000 residuals → <100 kernels)
- **Measurement**: `kernelCount / numResiduals`

### Performance
- **Target**: <5% compilation overhead
- **Target**: Same or better evaluation speed (due to better cache locality)

### Correctness
- **Target**: 100% test pass rate
- **Target**: Bit-identical results vs. current implementation

## Future Enhancements

### Partial Graph Matching

Instead of requiring exact topology match, match sub-graphs:
- `(a+b)*c` could partially reuse addition from `a+b`
- Requires more complex matching algorithm
- Benefit unclear - likely not worth complexity

### Template-Based Kernels

Pre-compile common patterns:
- Distance: `sqrt(dx² + dy²)`
- Parallel: `cross(dir1, dir2)`
- Dot product: `a.x*b.x + a.y*b.y`

Register templates and instantiate with indices.

### Multi-threaded Evaluation

With shared kernels, easier to parallelize:
```typescript
// Worker thread receives:
{ kernel: Function, valueArray: Float64Array, indices: number[] }
```

Evaluate multiple residuals in parallel.

## Migration Path

### Backward Compatibility

Keep existing `compileResidualJacobian()` working:
```typescript
// Old API (deprecated)
export function compileResidualJacobian(
  residual: Value,
  params: Value[],
  rowIdx: number
): (paramValues: number[], row: number[]) => number {
  // Old implementation stays
}

// New API (internal)
function compileIndirectKernel(...) {
  // New implementation
}
```

### Gradual Rollout

1. Phase 1-3: Internal only, no API changes
2. Phase 4: Enable by default in CompiledResiduals
3. Phase 5: Deprecate old direct compilation
4. Phase 6: Remove old implementation

### Feature Flag

```typescript
class CompiledResiduals {
  static compile(
    params: Value[],
    residualFn: (params: Value[]) => Value[],
    options: { useKernelReuse?: boolean } = {}
  ): CompiledResiduals {
    if (options.useKernelReuse ?? true) {
      return compileWithKernelReuse(params, residualFn);
    } else {
      return compileWithUniqueKernels(params, residualFn);
    }
  }
}
```

## Open Questions & Decisions

1. **Hash algorithm**: JSON.stringify vs custom hash?
   - JSON: Slow but correct
   - Custom: Fast but needs collision handling
   - **Decision**: Start with JSON, optimize if needed

2. **Canonicalization scope**: Exact match only, or allow commutativity?
   - `a+b` matches `b+a`?
   - Likely not worth complexity
   - **Decision**: Exact topology match only

3. **Value array updates**: Full array vs delta updates?
   - Full: Simple, potentially wasteful
   - Delta: Complex, cache-friendly
   - **Decision**: Start with full array updates

4. **Kernel warmup**: Pre-JIT compile hot kernels?
   - V8 will JIT automatically
   - Explicit warmup unlikely to help
   - **Decision**: No explicit warmup

5. **Variable deduplication** (UPDATED):
   - Should variables with same paramName be deduplicated?
   - **Decision**: Start with deduplication, test with variants to verify correctness
   - If issues arise, make them always unique

6. **Registry scope** (UPDATED):
   - Per-instance for now
   - **Future**: Cross-compilation reuse (e.g., Fusion 360 sketch constraints)
   - **Benefit**: Avoid recomputing entire graph on each constraint add/remove
   - **Challenge**: Kernel lifecycle management (unused kernels accumulate)
   - **Note for future**: Implement kernel GC or reference counting

## Summary

This project will significantly reduce memory usage and potentially improve performance by sharing compiled kernels across topologically identical residual graphs. The implementation follows a phased approach with comprehensive testing at each stage. Success is measured by kernel reuse rate (>90% target) and correctness (bit-identical results).

Key innovation: Instead of compiling unique kernels per residual, we:
1. Register all Values globally with unique IDs
2. Canonicalize graph structure into hash signatures
3. Share kernels for identical topologies
4. Pass input indices at runtime instead of hardcoding
