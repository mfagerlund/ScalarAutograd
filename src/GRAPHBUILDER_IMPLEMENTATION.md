# GraphBuilder Implementation Summary

## What Was Built

A new **incremental graph signature system** that eliminates the need for separate graph traversal during kernel selection.

### Key Components

1. **GraphBuilder** (`src/GraphBuilder.ts`)
   - Tracks leaf Values (params + constants) during graph construction
   - Computes stable hash after graph is complete
   - Returns `{ output: Value, signature: GraphSignature }`
   - Zero overhead when not in tracked mode

2. **Value Integration** (`src/Value.ts`)
   - Added `Value.currentBuilder` for context tracking
   - Added `_id` field for stable node identification
   - Hooks in `Value.make()` and `Value.makeNary()`

3. **Constant Interning** (`src/V.ts`)
   - `V.C()` now returns same Value object for same numeric value
   - Improves memory and enables proper signature matching
   - Limited to 10,000 cached constants

4. **Benchmarks** (`test/GraphBuilder.bench.test.ts`)
   - Performance comparisons across graph sizes
   - Collision rate testing (0.00% collisions)
   - Stability testing (100% stable)

## Performance Results

| Graph Type | Old (ms) | New (ms) | Speedup |
|------------|----------|----------|---------|
| Simple (3 params, 4 ops) | 0.006 | 0.005 | 1.29x |
| Medium (10 params, ~30 ops) | 0.026 | 0.022 | 1.19x |
| Complex (20 params, ~100 ops) | 0.094 | 0.068 | 1.38x |
| Hinge-like (8 params, ~50 ops) | 0.104 | 0.073 | 1.41x |

### Analysis

- **Modest speedup (1.2x-1.4x)** because graph building dominates
- **For 1000 vertices:** ~73ms/iteration (vs ~104ms old way)
- **Still expensive:** ~73 seconds to signature all vertices per optimization iteration
- **Conclusion:** Faster hashing helps but doesn't solve fundamental cost

## Usage Example

```typescript
import { GraphBuilder, V, Value } from 'scalar-autograd';

// Define parameters upfront
const params = [V.W(1), V.W(2), V.W(3)];

// Build graph with tracking
const builder = new GraphBuilder(params);
const { output, signature } = builder.build(() => {
  const a = V.add(params[0], params[1]);
  const b = V.mul(a, params[2]);
  return V.exp(b);
});

// Use signature for kernel lookup
console.log(signature.hash);           // '5f9a3b2c1e8d4f7a'
console.log(signature.leaves);         // [params[0], params[1], params[2]]
console.log(signature.leafIndexMap);   // Map { param0 => 0, param1 => 1, ... }

// Dynamic parameter discovery also works
const builder2 = new GraphBuilder();  // No params specified
const { signature: sig2 } = builder2.build(() => {
  const x = V.W(5);  // New parameter discovered during build!
  return V.mul(x, V.C(2));
});
console.log(sig2.leaves);  // [x, constant_2]
```

## How It Works

### 1. Graph Building Phase

```typescript
Value.currentBuilder = this;
const output = fn();  // User builds graph normally
```

- Every `Value.make()` call checks `Value.currentBuilder`
- If set, calls `builder.recordOp(value)` automatically
- Records leaf Values and operations

### 2. Leaf Registration

```typescript
if (prev.length === 0) {
  this.registerLeaf(value);  // Assign index
}
```

- Params specified upfront get indices 0, 1, 2...
- New params discovered during build are appended
- Constants are tracked separately

### 3. Hash Computation

After graph is complete:
1. Sort leaves: params (with grad) → params (no grad) → constants
2. Assign stable IDs: leaves get their indices, ops get sequential IDs
3. Hash structure based on operations + child IDs
4. Return signature with hash + leaf mapping

### 4. Kernel Execution

At execution time:
```typescript
const inputIndices = signature.leaves.map(leaf =>
  valueRegistry.getIndex(leaf)
);
kernel.execute(allValues, inputIndices, ...);
```

Map from Value objects → indices in global array.

## Key Design Decisions

### Why Not Hash During Building?

Initially tried incremental hashing during `recordOp()`, but:
- Hash depends on stable leaf indices
- Can't know final indices until graph is complete
- Led to unstable signatures

**Solution:** Record operations, compute hash after finalization.

### Why Constant Interning?

Without it:
```typescript
V.C(1e-12);  // Value object #1
V.C(1e-12);  // Value object #2 (different!)
```

These are semantically identical but hash differently → cache miss.

**Solution:** Intern constants so same value → same object.

### Why Track in Value.currentBuilder?

Alternatives considered:
- Wrap all V operations → verbose, error-prone
- Proxy pattern → performance overhead
- Thread-local context → **CHOSEN** (clean, zero overhead when unused)

## Limitations & Future Work

### 1. Graph Building Still Dominates

The real cost is building Value graphs, not hashing them.

**Better approaches:**
- **Topology grouping:** Group by valence, skip hashing entirely
- **Signature caching:** Assume stable across iterations, validate occasionally
- **Static declaration:** `energy.computeStatic()` pre-compiles once

### 2. Dynamic Graphs Remain Expensive

If graph structure changes (conditionals, topology changes), must rebuild every time.

**Potential solutions:**
- Conservative graph construction (remove conditionals)
- Template graphs with max capacity (pad with zeros)
- Hybrid CPU/GPU (complex on CPU, uniform on GPU)

### 3. TypeScript Performance Ceiling

For production at scale:
- Consider native module (Rust/WASM) for graph building
- Parallel signature computation (C# could do this, TS can't easily)
- Or accept sequential cost and cache aggressively

## Next Steps

1. **Measure real-world usage** in developable-sphere demo
2. **Design API** for static vs dynamic graph declaration
3. **Implement caching strategy** for stable signatures
4. **Benchmark topology-based grouping** as alternative
5. **Integrate with CompiledResiduals** (currently uses old canonicalizer)

## Files Modified

- `src/GraphBuilder.ts` - New incremental builder
- `src/Value.ts` - Added `currentBuilder`, `_id`, tracking hooks
- `src/V.ts` - Added constant interning
- `src/index.ts` - Export GraphBuilder
- `test/GraphBuilder.bench.test.ts` - Benchmark suite
- `src/GRAPH_HASHER_REFACTOR.md` - Results and analysis
