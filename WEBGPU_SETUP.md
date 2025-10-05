# WebGPU Setup Guide

## Prerequisites

### Node.js Version
- **Minimum**: Node.js 18 or higher
- **Recommended**: Node.js 20+

Check your version:
```bash
node --version
```

### Platform Requirements

WebGPU runs on all major platforms but may require different setup:

#### Windows
- Works out of the box with prebuilt binaries
- Uses D3D12 backend by default

#### macOS
- Works out of the box with prebuilt binaries (x64 and ARM)
- Uses Metal backend

#### Linux
- Works out of the box with prebuilt binaries (x64)
- Uses Vulkan backend (requires Vulkan drivers)

## Installation

Add WebGPU dependencies to your project:

```bash
npm install webgpu @webgpu/types
```

Or add to `package.json`:

```json
{
  "dependencies": {
    "@webgpu/types": "^0.1.65",
    "webgpu": "^0.3.8"
  }
}
```

## Verification

Run the test suite to verify WebGPU is working:

```bash
npm test -- test/gpu/hello-gpu.spec.ts
```

Expected output:
```
✓ test/gpu/hello-gpu.spec.ts (2 tests)
  WebGPU Hello World
    ✓ should square an array of numbers on GPU
    ✓ should handle larger arrays efficiently
```

## Usage

### Basic Initialization

```typescript
import { WebGPUContext } from './src/gpu/WebGPUContext';

// Get singleton instance
const ctx = WebGPUContext.getInstance();

// Initialize (call once)
await ctx.initialize();

// Access device and queue
const device = ctx.device;
const queue = ctx.queue;
```

### Check Availability

```typescript
if (WebGPUContext.isAvailable()) {
  // WebGPU is available
  const ctx = WebGPUContext.getInstance();
  await ctx.initialize();
} else {
  // Fall back to CPU
  console.warn('WebGPU not available');
}
```

## Troubleshooting

### "Cannot find package 'webgpu'"

**Cause**: Package not installed or in wrong dependencies section

**Solution**:
```bash
npm install webgpu @webgpu/types
```

Make sure `webgpu` is in `dependencies` (not `devDependencies`) in `package.json`.

### "WebGPU initialization failed"

**Cause**: No GPU adapter found

**Possible reasons**:
- Running in environment without GPU access
- Graphics drivers outdated
- Vulkan not installed (Linux)

**Solution**:
- Update graphics drivers
- On Linux: Install Vulkan runtime (`sudo apt install vulkan-tools` on Ubuntu)
- Use fallback to CPU mode if GPU unavailable

### Test timeouts

**Cause**: GPU operations taking too long

**Solution**: Increase test timeout in vitest config or test file:
```typescript
it('test name', async () => {
  // test code
}, { timeout: 10000 }); // 10 second timeout
```

## Current Status

### ✅ Milestone 1: WebGPU Foundation - COMPLETE

- [x] WebGPU API research (using `webgpu` package v0.3.8)
- [x] Created `WebGPUContext.ts` singleton for device management
- [x] "Hello GPU" test: squares array of numbers
- [x] Platform compatibility: Windows/macOS/Linux
- [x] Documentation

**Test Results**:
- Basic computation: ✅ `[1,2,3,4]` squared correctly
- Large arrays: ✅ 1000 elements processed in ~900ms
- GPU initialization: ✅ Automatic adapter/device setup

### Next Steps

Proceed to **Milestone 2: Simple Kernel Compilation**
- Compile Value graph to WGSL
- Implement forward pass code generation
- Test: `V.add(V.mul(a, 2), b)` runs on GPU

## References

- [WebGPU npm package](https://www.npmjs.com/package/webgpu)
- [WebGPU types](https://www.npmjs.com/package/@webgpu/types)
- [dawn-gpu/node-webgpu GitHub](https://github.com/dawn-gpu/node-webgpu)
- [WebGPU Spec](https://www.w3.org/TR/webgpu/)
- [WGSL Spec](https://www.w3.org/TR/WGSL/)

## Performance Notes

Current GPU overhead observed:
- Initialization: ~200ms (one-time cost)
- Simple kernel (4 elements): ~905ms total test time
- Large kernel (1000 elements): Same ~905ms (most is setup)

**Implication**: GPU only wins when:
1. Kernel can be reused many times
2. Batch size is large (>100 elements)
3. Computation is complex enough to offset buffer transfer

This validates the plan to focus on **batched residual execution** where same kernel runs on many instances.
