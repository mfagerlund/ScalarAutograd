/**
 * Optimized Mesh Benchmark - Build ONE residual template, not 5000 copies
 */

import { V, Value, CompiledFunctions } from '../../src';
import { WebGPUContext } from '../../src/gpu/WebGPUContext';
import { compileToWGSL, WGSLKernel } from '../../src/gpu/compileToWGSL';

describe('Mesh Optimization (Optimized)', () => {
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

  it('should benchmark 1k vertices × 200 iterations (fast)', { timeout: 30000 }, async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    await runOptimizedBenchmark(1000, 200);
  });

  it('should benchmark 5k vertices × 200 iterations (fast)', { timeout: 60000 }, async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    await runOptimizedBenchmark(5000, 200);
  });

  it('should benchmark 10k vertices × 200 iterations (fast)', { timeout: 90000 }, async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    await runOptimizedBenchmark(10000, 200);
  });
});

async function runOptimizedBenchmark(numVertices: number, iterations: number) {
  const ctx = WebGPUContext.getInstance();
  console.log(`\n${'='.repeat(60)}`);
  console.log(`MESH OPTIMIZATION (OPTIMIZED): ${numVertices} vertices × ${iterations} iterations`);
  console.log('='.repeat(60));

  // KEY INSIGHT: Build ONE residual template and reuse it for all vertices
  // Instead of creating 5000 identical Value graphs, create 1 template

  // Create parameter array for ALL vertices
  const params: Value[] = [];
  for (let i = 0; i < numVertices * 6; i++) {
    params.push(V.W(0));
  }

  console.log(`\nSetup: ${numVertices} vertices, ${params.length} parameters`);

  // Build residuals that reference different slices of params
  const residuals: Value[] = [];
  for (let i = 0; i < numVertices; i++) {
    const offset = i * 6;
    const x0 = params[offset + 0];
    const y0 = params[offset + 1];
    const z0 = params[offset + 2];
    const x1 = params[offset + 3];
    const y1 = params[offset + 4];
    const z1 = params[offset + 5];

    const dx = V.sub(x1, x0);
    const dy = V.sub(y1, y0);
    const dz = V.sub(z1, z0);
    const residual = V.add(V.add(V.square(dx), V.square(dy)), V.square(dz));

    residuals.push(residual);
  }

  // ===== COMPILED CPU =====
  console.log('\n--- Compiling CPU kernels ---');
  const compileStart = performance.now();
  const compiled = CompiledFunctions.compile(params, (p) => residuals);
  const compileTime = performance.now() - compileStart;
  console.log(`Compilation: ${compileTime.toFixed(0)}ms`);

  // Warm-up
  compiled.evaluateSumWithGradient(params);

  // Benchmark
  const sampleIters = 5;
  console.log(`\nRunning ${sampleIters} sample iterations (Compiled CPU)...`);
  const compiledStart = performance.now();
  for (let iter = 0; iter < sampleIters; iter++) {
    compiled.evaluateSumWithGradient(params);
  }
  const compiledSampleTime = performance.now() - compiledStart;
  const compiledPerIter = compiledSampleTime / sampleIters;
  const compiledTime = compiledPerIter * iterations;

  console.log(`\n[COMPILED CPU] (extrapolated from ${sampleIters} samples)`);
  console.log(`  Compile time: ${compileTime.toFixed(0)}ms`);
  console.log(`  Total runtime (${iterations} iters): ${compiledTime.toFixed(0)}ms`);
  console.log(`  Per iteration: ${compiledPerIter.toFixed(2)}ms`);
  console.log(`  Residual evals/sec: ${(numVertices * iterations / compiledTime * 1000).toFixed(0)}`);

  // ===== GPU =====
  console.log('\n--- Compiling GPU kernel ---');
  const gpuCompileStart = performance.now();
  const { wgslCode, graphInputs } = compileToWGSL(residuals[0]);
  const kernel = new WGSLKernel(ctx.device, wgslCode, graphInputs);
  const gpuCompileTime = performance.now() - gpuCompileStart;
  console.log(`GPU compilation: ${gpuCompileTime.toFixed(0)}ms`);

  // Pack batch data
  const inputsPerResidual = graphInputs.length;
  const batchInputs = new Float32Array(numVertices * inputsPerResidual);

  // Initialize with random positions
  for (let i = 0; i < numVertices; i++) {
    const theta = (i / numVertices) * 2 * Math.PI;
    const phi = Math.acos(2 * (i / numVertices) - 1);

    // Get the 6 params for this vertex
    const offset = i * 6;
    const vertexParams = [
      params[offset + 0],
      params[offset + 1],
      params[offset + 2],
      params[offset + 3],
      params[offset + 4],
      params[offset + 5]
    ];

    // Map to graphInputs order
    for (let j = 0; j < 6; j++) {
      const param = vertexParams[j];
      const idx = graphInputs.indexOf(param);

      if (j === 0) batchInputs[i * inputsPerResidual + idx] = Math.sin(phi) * Math.cos(theta); // x0
      else if (j === 1) batchInputs[i * inputsPerResidual + idx] = Math.sin(phi) * Math.sin(theta); // y0
      else if (j === 2) batchInputs[i * inputsPerResidual + idx] = Math.cos(phi); // z0
      else if (j === 3) batchInputs[i * inputsPerResidual + idx] = Math.sin(phi) * Math.cos(theta + 0.1); // x1
      else if (j === 4) batchInputs[i * inputsPerResidual + idx] = Math.sin(phi) * Math.sin(theta + 0.1); // y1
      else if (j === 5) batchInputs[i * inputsPerResidual + idx] = Math.cos(phi) + 0.1; // z1
    }
  }

  // Warm-up
  await kernel.execute(batchInputs, numVertices);

  // Benchmark
  console.log(`\nRunning ${sampleIters} sample iterations (GPU batched)...`);
  const gpuStart = performance.now();
  for (let iter = 0; iter < sampleIters; iter++) {
    await kernel.execute(batchInputs, numVertices);
  }
  const gpuSampleTime = performance.now() - gpuStart;
  const gpuPerIter = gpuSampleTime / sampleIters;
  const gpuTime = gpuPerIter * iterations;

  console.log(`\n[GPU BATCHED] (extrapolated from ${sampleIters} samples)`);
  console.log(`  Compile time: ${gpuCompileTime.toFixed(0)}ms`);
  console.log(`  Total runtime (${iterations} iters): ${gpuTime.toFixed(0)}ms`);
  console.log(`  Per iteration: ${gpuPerIter.toFixed(2)}ms`);
  console.log(`  Residual evals/sec: ${(numVertices * iterations / gpuTime * 1000).toFixed(0)}`);

  // ===== COMPARISON =====
  const speedup = compiledTime / gpuTime;
  const timeSaved = compiledTime - gpuTime;

  console.log(`\n${'='.repeat(60)}`);
  console.log('RESULTS');
  console.log('='.repeat(60));
  console.log(`Compiled CPU: ${compiledTime.toFixed(0)}ms (+ ${compileTime.toFixed(0)}ms compile)`);
  console.log(`GPU Batched:  ${gpuTime.toFixed(0)}ms (+ ${gpuCompileTime.toFixed(0)}ms compile)`);
  console.log(`Speedup:      ${speedup.toFixed(2)}x`);
  console.log(`Time saved:   ${timeSaved.toFixed(0)}ms (${(timeSaved / 1000).toFixed(1)}s)`);
  console.log('='.repeat(60));

  if (speedup > 1.0) {
    console.log(`\n✅ GPU wins by ${speedup.toFixed(1)}x!`);
  } else {
    console.log(`\n⚠️  Compiled CPU is faster (GPU: ${(1/speedup).toFixed(2)}x slower)`);
  }

  console.log(`\nPer-iteration breakdown:`);
  console.log(`  Compiled CPU: ${compiledPerIter.toFixed(2)}ms/iter`);
  console.log(`  GPU:          ${gpuPerIter.toFixed(2)}ms/iter`);
  console.log(`  Difference:   ${Math.abs(compiledPerIter - gpuPerIter).toFixed(2)}ms/iter\n`);

  // For realistic use cases, GPU should win at 5k+
  if (numVertices >= 5000) {
    expect(speedup).toBeGreaterThan(1.0);
  }
}
