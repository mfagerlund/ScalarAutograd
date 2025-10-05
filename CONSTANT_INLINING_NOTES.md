# Constant Inlining Optimization

## Current Status

### ✅ GPU Compiler (WGSL)
**Already optimized!** Constants are inlined as f32 literals in generated WGSL code.

```wgsl
// Before (conceptual):
let v0 = inputs[thread_idx * 3 + 0];  // parameter a
let v1 = inputs[thread_idx * 3 + 1];  // constant 2.0 (wasteful!)
let v2 = inputs[thread_idx * 3 + 2];  // parameter b

// After (current implementation):
let v0 = inputs[thread_idx * 2 + 0];  // parameter a
let v1 = 2.0;                          // constant inlined!
let v2 = inputs[thread_idx * 2 + 1];  // parameter b
```

**Implementation**: `src/gpu/compileToWGSL.ts` lines 92-103

```typescript
if (node.requiresGrad) {
  // Parameter - load from buffer
  const inputIdx = graphInputs.indexOf(node);
  declarations.push(`let ${varName} = inputs[thread_idx * ${graphInputs.length} + ${inputIdx}];`);
} else {
  // Constant - inline as f32 literal
  const value = node.data;
  const wgslLiteral = Number.isInteger(value) ? `${value}.0` : String(value);
  declarations.push(`let ${varName} = ${wgslLiteral};`);
}
```

**Benefits**:
- ✅ Smaller input buffers (only parameters, not constants)
- ✅ Better memory locality
- ✅ No wasted GPU memory reads for constants
- ✅ Type-safe (integers auto-converted to `f32`)

### ⏸️ CPU Compiler (JavaScript)
**Not yet optimized** - constants currently loaded from `allValues` array.

```javascript
// Current:
const _v0 = allValues[idx_0];  // parameter
const _v1 = allValues[idx_1];  // constant 2.0 (via registry)
const _v2 = allValues[idx_2];  // parameter

// Proposed:
const _v0 = allValues[idx_0];  // parameter
const _v1 = 2.0;                // constant inlined!
const _v2 = allValues[idx_1];  // parameter (note: idx_1 now!)
```

## Why CPU Optimization is Complex

The CPU compiler is deeply integrated with the `ValueRegistry` system. Making constants inline-only requires coordinated changes:

### 1. Registry Registration
**Current**: All leaf nodes (parameters + constants) registered
```typescript
if ((node as any).prev.length === 0) {
  registry.register(node);  // Both params AND constants
}
```

**Proposed**: Only register parameters
```typescript
if ((node as any).prev.length === 0 && node.requiresGrad) {
  registry.register(node);  // Only parameters
}
```

### 2. Index Extraction
**Current**: `extractInputIndices()` returns IDs for all leaves
```typescript
// Returns [id_param1, id_constant_2.0, id_param2, ...]
```

**Proposed**: Only return parameter IDs
```typescript
// Returns [id_param1, id_param2, ...]
```

### 3. Code Generation
**Current**: All leaves load from `allValues`
```typescript
if (prev.length === 0) {
  const indexVar = nodeToIndexVar.get(node)!;
  forwardCode.push(`const ${getVarName(node)} = allValues[${indexVar}];`);
}
```

**Proposed**: Parameters load, constants inline
```typescript
if (prev.length === 0) {
  if (node.requiresGrad) {
    const indexVar = nodeToIndexVar.get(node)!;
    forwardCode.push(`const ${getVarName(node)} = allValues[${indexVar}];`);
  } else {
    forwardCode.push(`const ${getVarName(node)} = ${node.data};`);
  }
}
```

### 4. CompiledFunctions Integration
**Current**: Builds `allValues` array with all leaves
```typescript
const allValues = registry.getValues();  // [param1, constant_2.0, param2, ...]
```

**Proposed**: Only pass parameters
```typescript
const allValues = registry.getParameterValues();  // [param1, param2, ...]
```

### 5. Call Sites
Many places assume `indices.length === graphInputs.length` (including constants).

**Examples needing updates**:
- `test/same-graph-test.spec.ts` (manually collects leaves)
- Any code that calls `registry.getId()` expecting constants to be registered
- Kernel signature matching logic

## Attempted Fix & Why It Failed

**Attempt**: Inline constants in code generation only, keep them in registry

**Result**: Gradients incorrect!

**Why**: The `indices` array still had entries for constants, but the code was reading from wrong positions:

```javascript
// indices = [0, 1, 2]  // param, constant, param
// allValues at runtime = [5, 2.0, 7]

// Generated code (buggy):
const _v0 = allValues[idx_0];  // allValues[0] = 5 ✅
const _v1 = 2.0;                // inlined ✅
const _v2 = allValues[idx_2];  // allValues[2] = 7, but should read position 1!
```

The indexing was off because constants weren't actually in `allValues` but `indices` thought they were.

## Correct Implementation Path

**Phase 1: Registry Changes**
1. Add `registry.getParameterValues()` - returns only `requiresGrad=true` values
2. Keep `registry.getValues()` for backward compatibility
3. Update registry to track parameters separately

**Phase 2: Compiler Changes**
1. Modify `compileIndirectKernel()` to inline constants
2. Modify `extractInputIndices()` to exclude constants
3. Update generated code to use filtered indices

**Phase 3: Integration**
1. Update `CompiledFunctions` to use parameter-only values
2. Update kernel signature generation to exclude constants
3. Fix all call sites that manually build indices

**Phase 4: Testing**
1. All gradient tests must pass
2. Kernel reuse tests must pass
3. Performance benchmarks to verify optimization helps

## Benefits When Complete

### Memory Savings
For a graph like `sqrt(x^2 + y^2 + 2.0^2)`:
- **Current**: 3 entries in `allValues` (x, y, 2.0)
- **Optimized**: 2 entries (x, y)

With 1000 residuals: **33% reduction** in indices array size!

### Cache Locality
Parameters packed tightly in `allValues` array = better CPU cache utilization.

### Kernel Reuse
Two graphs with same topology but different constant values:
- `sqrt(x^2 + y^2 + 2.0^2)`
- `sqrt(x^2 + y^2 + 5.0^2)`

**Could share more code** since constants are baked into kernel, not looked up.

## Current Recommendation

**GPU**: ✅ Already optimized, no changes needed.

**CPU**: ⏸️ Defer optimization until:
1. Core functionality is stable
2. Performance profiling shows this is a bottleneck
3. Test suite is comprehensive

The optimization is **correct in principle** but requires significant refactoring across the codebase. The GPU already has the benefit, so focus on higher-value work first.

## References

- GPU implementation: `src/gpu/compileToWGSL.ts`
- CPU compiler: `src/compileIndirectKernel.ts`
- Registry: `src/ValueRegistry.ts`
- Integration: `src/CompiledFunctions.ts`
