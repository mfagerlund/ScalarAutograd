/**
 * Realistic Mesh Optimization Benchmark
 *
 * Simulates typical mesh editing workflow:
 * - 1k-10k vertices
 * - 200 optimization iterations
 * - Distance/angle constraints per vertex
 */

import { V, Value, CompiledFunctions } from '../../src';
import { WebGPUContext } from '../../src/gpu/WebGPUContext';
import { compileToWGSL, WGSLKernel } from '../../src/gpu/compileToWGSL';

describe('Mesh Optimization Benchmark (Realistic)', () => {
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

  it('should benchmark 1k vertices × 200 iterations', { timeout: 120000 }, async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    await runMeshBenchmark(1000, 200);
  });

  it('should benchmark 5k vertices × 200 iterations', { timeout: 180000 }, async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    await runMeshBenchmark(5000, 200);
  });

  it('should benchmark 10k vertices × 200 iterations', { timeout: 180000 }, async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    await runMeshBenchmark(10000, 200);
  });
});

async function runMeshBenchmark(numVertices: number, iterations: number) {
  const ctx = WebGPUContext.getInstance();
  console.log(`\n${'='.repeat(60)}`);
  console.log(`MESH OPTIMIZATION: ${numVertices} vertices × ${iterations} iterations`);
  console.log('='.repeat(60));

  // Create simple flattening constraint: each vertex wants to align with neighbors
  // This is typical for developable surface optimization
  const createResidual = () => {
    const x0 = V.W(0);
    const y0 = V.W(0);
    const z0 = V.W(0);
    const x1 = V.W(0);
    const y1 = V.W(0);
    const z1 = V.W(0);

    // Angle between vertex normal and neighbor
    // Simplified: just distance for now
    const dx = V.sub(x1, x0);
    const dy = V.sub(y1, y0);
    const dz = V.sub(z1, z0);
    return V.add(V.add(V.square(dx), V.square(dy)), V.square(dz));
  };

  // Build residuals and params
  const residuals: Value[] = [];
  const allParams: Value[] = [];

  for (let i = 0; i < numVertices; i++) {
    const residual = createResidual();
    residuals.push(residual);

    // Collect params (6 per residual: x0,y0,z0, x1,y1,z1)
    const visited = new Set<Value>();
    const leaves: Value[] = [];

    function collectLeaves(node: Value) {
      if (visited.has(node)) return;
      visited.add(node);

      const prev = (node as any).prev as Value[];
      if (prev.length === 0 && node.requiresGrad) {
        leaves.push(node);
      } else {
        for (const child of prev) {
          collectLeaves(child);
        }
      }
    }

    collectLeaves(residual);
    allParams.push(...leaves);
  }

  console.log(`\nSetup: ${numVertices} residuals, ${allParams.length} parameters`);

  // ===== COMPILED CPU =====
  console.log('\n--- Compiling CPU kernels ---');
  const compileStart = performance.now();
  const compiled = CompiledFunctions.compile(allParams, (p) => residuals);
  const compileTime = performance.now() - compileStart;
  console.log(`Compilation: ${compileTime.toFixed(0)}ms`);

  // Warm-up
  compiled.evaluateSumWithGradient(allParams);

  // Benchmark with small sample, extrapolate
  const sampleIters = 5;
  console.log(`\nRunning ${sampleIters} sample iterations (Compiled CPU)...`);
  const compiledStart = performance.now();
  for (let iter = 0; iter < sampleIters; iter++) {
    compiled.evaluateSumWithGradient(allParams);
  }
  const compiledSampleTime = performance.now() - compiledStart;
  const compiledPerIter = compiledSampleTime / sampleIters;
  const compiledTime = compiledPerIter * iterations; // Extrapolate

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

    // Find param order
    const residual = residuals[i];
    const visited = new Set<Value>();
    const leaves: Value[] = [];

    function collectLeaves(node: Value) {
      if (visited.has(node)) return;
      visited.add(node);

      const prev = (node as any).prev as Value[];
      if (prev.length === 0 && node.requiresGrad) {
        leaves.push(node);
      } else {
        for (const child of prev) {
          collectLeaves(child);
        }
      }
    }

    collectLeaves(residual);

    // Map to graphInputs order
    for (let j = 0; j < inputsPerResidual; j++) {
      const param = leaves[j];
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

  // Benchmark with small sample, extrapolate
  console.log(`\nRunning ${sampleIters} sample iterations (GPU batched)...`);
  const gpuStart = performance.now();
  for (let iter = 0; iter < sampleIters; iter++) {
    await kernel.execute(batchInputs, numVertices);
  }
  const gpuSampleTime = performance.now() - gpuStart;
  const gpuPerIter = gpuSampleTime / sampleIters;
  const gpuTime = gpuPerIter * iterations; // Extrapolate

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
  console.log(`Compiled CPU: ${compiledTime.toFixed(0)}ms`);
  console.log(`GPU Batched:  ${gpuTime.toFixed(0)}ms`);
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

  // Verify GPU is actually faster for this use case
  expect(speedup).toBeGreaterThan(1.0);
}
