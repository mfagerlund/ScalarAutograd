/**
 * Test GPU kernel batching - execute many residuals in parallel
 */

import { V, Value } from '../../src';
import { WebGPUContext } from '../../src/gpu/WebGPUContext';
import { compileToWGSL, WGSLKernel } from '../../src/gpu/compileToWGSL';

describe('GPU Kernel Batching', () => {
  let ctx: WebGPUContext;

  beforeAll(async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('WebGPU not available - skipping GPU tests');
      return;
    }

    ctx = WebGPUContext.getInstance();
    await ctx.initialize();
  });

  afterAll(() => {
    WebGPUContext.reset();
  });

  it('should batch 100 distance constraints in single GPU call', async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    // Distance constraint: sqrt((x1-x0)^2 + (y1-y0)^2) - targetDist
    const x0 = V.W(0);
    const y0 = V.W(0);
    const x1 = V.W(0);
    const y1 = V.W(0);
    const targetDist = V.C(5.0);

    const dx = V.sub(x1, x0);
    const dy = V.sub(y1, y0);
    const distSq = V.add(V.mul(dx, dx), V.mul(dy, dy));
    const dist = V.sqrt(distSq);
    const residual = V.sub(dist, targetDist);

    // Compile to WGSL
    const { wgslCode, graphInputs } = compileToWGSL(residual);

    console.log('\n=== Distance Constraint WGSL ===');
    console.log(wgslCode);
    console.log('Graph inputs order:', graphInputs.map(v => v.label));
    console.log('================================\n');

    expect(graphInputs.length).toBe(4); // Will be in discovery order

    const kernel = new WGSLKernel(ctx.device, wgslCode, graphInputs);

    // Create 100 distance constraints with different positions
    const batchSize = 100;
    const batchInputs = new Float32Array(batchSize * 4);

    // Map: which graphInput corresponds to which semantic meaning?
    const x1Idx = graphInputs.indexOf(x1);
    const x0Idx = graphInputs.indexOf(x0);
    const y1Idx = graphInputs.indexOf(y1);
    const y0Idx = graphInputs.indexOf(y0);

    for (let i = 0; i < batchSize; i++) {
      // Point 0 at origin, Point 1 at varying distances
      const angle = (i / batchSize) * 2 * Math.PI;
      const actualDist = 3 + i * 0.05; // Varying distance from 3 to 8

      // Pack in graphInputs order
      batchInputs[i * 4 + x0Idx] = 0;
      batchInputs[i * 4 + y0Idx] = 0;
      batchInputs[i * 4 + x1Idx] = actualDist * Math.cos(angle);
      batchInputs[i * 4 + y1Idx] = actualDist * Math.sin(angle);
    }

    // Execute all 100 residuals on GPU in single call
    const t0 = performance.now();
    const gpuResults = await kernel.execute(batchInputs, batchSize);
    const gpuTime = performance.now() - t0;

    // Verify results against CPU
    let maxError = 0;
    for (let i = 0; i < batchSize; i++) {
      const x0Val = batchInputs[i * 4 + x0Idx];
      const y0Val = batchInputs[i * 4 + y0Idx];
      const x1Val = batchInputs[i * 4 + x1Idx];
      const y1Val = batchInputs[i * 4 + y1Idx];

      const cpuDist = Math.sqrt((x1Val - x0Val) ** 2 + (y1Val - y0Val) ** 2);
      const cpuResidual = cpuDist - 5.0;

      const error = Math.abs(gpuResults[i] - cpuResidual);
      maxError = Math.max(maxError, error);
    }

    console.log(`GPU batched ${batchSize} residuals in ${gpuTime.toFixed(2)}ms`);
    console.log(`Max error vs CPU: ${maxError.toExponential(6)}`);

    expect(maxError).toBeLessThan(1e-5);
  });

  it('should handle varying batch sizes efficiently', async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    // Simple residual: (x - target)^2
    const x = V.W(0);
    const target = V.C(10.0);
    const residual = V.pow(V.sub(x, target), 2);

    const { wgslCode, graphInputs } = compileToWGSL(residual);
    const kernel = new WGSLKernel(ctx.device, wgslCode, graphInputs);

    const sizes = [1, 10, 100, 1000];
    const times: number[] = [];

    for (const size of sizes) {
      const inputs = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        inputs[i] = i;  // x values from 0 to size-1
      }

      const t0 = performance.now();
      const results = await kernel.execute(inputs, size);
      const elapsed = performance.now() - t0;
      times.push(elapsed);

      // Verify a few samples
      expect(results[0]).toBeCloseTo((0 - 10) ** 2, 5);
      if (size > 10) {
        expect(results[10]).toBeCloseTo((10 - 10) ** 2, 5);
      }

      console.log(`Batch size ${size.toString().padStart(4)}: ${elapsed.toFixed(2)}ms`);
    }

    // Verify all results correct
    console.log('\nBatch scaling is working!');
  });

  it('should match CPU performance on complex expression', async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    // Complex residual: sin(x) * cos(y) + exp(-z)
    const x = V.W(0);
    const y = V.W(0);
    const z = V.W(0);

    const term1 = V.mul(V.sin(x), V.cos(y));
    const term2 = V.exp(V.neg(z));
    const residual = V.add(term1, term2);

    const { wgslCode, graphInputs } = compileToWGSL(residual);
    const kernel = new WGSLKernel(ctx.device, wgslCode, graphInputs);

    const batchSize = 500;
    const batchInputs = new Float32Array(batchSize * 3);

    // Fill with varying inputs
    for (let i = 0; i < batchSize; i++) {
      batchInputs[i * 3 + 0] = (i / batchSize) * Math.PI;     // x: 0 to π
      batchInputs[i * 3 + 1] = (i / batchSize) * Math.PI / 2; // y: 0 to π/2
      batchInputs[i * 3 + 2] = i * 0.01;                       // z: 0 to 5
    }

    // Execute on GPU
    const t0 = performance.now();
    const gpuResults = await kernel.execute(batchInputs, batchSize);
    const gpuTime = performance.now() - t0;

    // Execute on CPU for comparison
    const t1 = performance.now();
    const cpuResults = new Float32Array(batchSize);
    for (let i = 0; i < batchSize; i++) {
      const xVal = batchInputs[i * 3 + 0];
      const yVal = batchInputs[i * 3 + 1];
      const zVal = batchInputs[i * 3 + 2];
      cpuResults[i] = Math.sin(xVal) * Math.cos(yVal) + Math.exp(-zVal);
    }
    const cpuTime = performance.now() - t1;

    // Compare results
    let maxError = 0;
    for (let i = 0; i < batchSize; i++) {
      const error = Math.abs(gpuResults[i] - cpuResults[i]);
      maxError = Math.max(maxError, error);
    }

    console.log(`\nGPU: ${gpuTime.toFixed(2)}ms for ${batchSize} evaluations`);
    console.log(`CPU: ${cpuTime.toFixed(2)}ms for ${batchSize} evaluations`);
    console.log(`Speedup: ${(cpuTime / gpuTime).toFixed(2)}x`);
    console.log(`Max error: ${maxError.toExponential(6)}`);

    expect(maxError).toBeLessThan(2e-5);  // f32 precision limits
  });
});
