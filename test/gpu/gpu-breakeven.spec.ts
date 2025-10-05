/**
 * Find the break-even point where GPU beats Compiled CPU
 */

import { V, Value, CompiledFunctions } from '../../src';
import { WebGPUContext } from '../../src/gpu/WebGPUContext';
import { compileToWGSL, WGSLKernel } from '../../src/gpu/compileToWGSL';

describe('GPU Break-Even Analysis', () => {
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

  it('should find break-even point for simple residuals', { timeout: 60000 }, async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    // Simple residual: distance constraint
    // sqrt((x1-x0)^2 + (y1-y0)^2) - target
    const createResidual = () => {
      const x0 = V.W(0);
      const y0 = V.W(0);
      const x1 = V.W(0);
      const y1 = V.W(0);
      const target = V.C(5.0);

      const dx = V.sub(x1, x0);
      const dy = V.sub(y1, y0);
      const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
      return V.sub(dist, target);
    };

    // Test batch sizes from 10 to 10,000
    const batchSizes = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000];

    console.log('\n========================================');
    console.log('GPU BREAK-EVEN ANALYSIS');
    console.log('========================================\n');
    console.log('Batch | Compiled | GPU     | Speedup | Winner');
    console.log('------|----------|---------|---------|--------');

    let breakEvenFound = false;
    let breakEvenSize = 0;

    for (const batchSize of batchSizes) {
      // Create batch of residuals
      const residuals: Value[] = [];
      const allParams: Value[] = [];

      for (let i = 0; i < batchSize; i++) {
        const residual = createResidual();
        residuals.push(residual);

        // Collect params (4 per residual: x0, y0, x1, y1)
        // In reverse topo order, leaves come first
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

      // COMPILED CPU
      const compiled = CompiledFunctions.compile(allParams, (p) => residuals);

      const compiledIterations = Math.max(10, Math.floor(1000 / batchSize));
      const compiledStart = performance.now();
      for (let iter = 0; iter < compiledIterations; iter++) {
        compiled.evaluateSumWithGradient(allParams);
      }
      const compiledTime = (performance.now() - compiledStart) / compiledIterations;

      // GPU
      const { wgslCode, graphInputs } = compileToWGSL(residuals[0]);
      const kernel = new WGSLKernel(ctx.device, wgslCode, graphInputs);

      // Pack batch data
      const inputsPerResidual = graphInputs.length;
      const batchInputs = new Float32Array(batchSize * inputsPerResidual);

      for (let i = 0; i < batchSize; i++) {
        const angle = (i / batchSize) * 2 * Math.PI;
        const dist = 3 + i * 0.01;

        // Find which graphInput is which param
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

          if (j === 0) batchInputs[i * inputsPerResidual + idx] = 0; // x0
          else if (j === 1) batchInputs[i * inputsPerResidual + idx] = 0; // y0
          else if (j === 2) batchInputs[i * inputsPerResidual + idx] = dist * Math.cos(angle); // x1
          else if (j === 3) batchInputs[i * inputsPerResidual + idx] = dist * Math.sin(angle); // y1
        }
      }

      const gpuIterations = Math.max(10, Math.floor(1000 / batchSize));
      const gpuStart = performance.now();
      for (let iter = 0; iter < gpuIterations; iter++) {
        await kernel.execute(batchInputs, batchSize);
      }
      const gpuTime = (performance.now() - gpuStart) / gpuIterations;

      const speedup = compiledTime / gpuTime;
      const winner = speedup > 1.0 ? 'GPU' : 'Compiled';

      if (!breakEvenFound && speedup > 1.0) {
        breakEvenFound = true;
        breakEvenSize = batchSize;
      }

      console.log(
        `${batchSize.toString().padStart(5)} | ` +
        `${compiledTime.toFixed(2).padStart(8)}ms | ` +
        `${gpuTime.toFixed(2).padStart(7)}ms | ` +
        `${speedup.toFixed(2).padStart(7)}x | ` +
        `${winner}`
      );
    }

    console.log('\n========================================');
    if (breakEvenFound) {
      console.log(`Break-even point: ~${breakEvenSize} residuals`);
    } else {
      console.log('GPU did not beat Compiled in tested range');
    }
    console.log('========================================\n');
  });
});
