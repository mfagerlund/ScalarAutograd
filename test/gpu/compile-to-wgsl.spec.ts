/**
 * Test WGSL compilation - Convert Value graphs to GPU shaders
 */

import { V, Value } from '../../src';
import { WebGPUContext } from '../../src/gpu/WebGPUContext';
import { compileToWGSL, WGSLKernel } from '../../src/gpu/compileToWGSL';

describe('WGSL Compilation', () => {
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

  it('should compile simple arithmetic: a * 2 + b', async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    // Build computation graph
    const a = V.W(0); // Placeholder values
    const b = V.W(0);
    const graph = V.add(V.mul(a, V.C(2)), b);

    // Compile to WGSL
    const { wgslCode, graphInputs } = compileToWGSL(graph);

    console.log('\n=== Generated WGSL ===');
    console.log(wgslCode);
    console.log('=====================\n');

    // Verify graph inputs detected correctly
    expect(graphInputs.length).toBe(2); // a and b (not constant 2)
    expect(graphInputs).toContain(a);
    expect(graphInputs).toContain(b);

    // Create kernel
    const kernel = new WGSLKernel(ctx.device, wgslCode, graphInputs);

    // Test with multiple inputs
    // Input format: [a0, b0, a1, b1, a2, b2, ...]
    const batchInputs = new Float32Array([
      1, 2,    // a=1, b=2  → 1*2+2 = 4
      3, 4,    // a=3, b=4  → 3*2+4 = 10
      5, 6,    // a=5, b=6  → 5*2+6 = 16
      -2, 10   // a=-2, b=10 → -2*2+10 = 6
    ]);
    const batchSize = 4;

    // Execute on GPU
    const gpuResults = await kernel.execute(batchInputs, batchSize);

    // Verify results
    const expected = [4, 10, 16, 6];
    console.log('GPU Results:', Array.from(gpuResults));
    console.log('Expected:', expected);

    expect(gpuResults.length).toBe(batchSize);
    for (let i = 0; i < batchSize; i++) {
      expect(gpuResults[i]).toBeCloseTo(expected[i], 5);
    }
  });

  it('should handle complex math operations', async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    // Graph: sin(x) + cos(y) * 2
    const x = V.W(0);
    const y = V.W(0);
    const graph = V.add(V.sin(x), V.mul(V.cos(y), V.C(2)));

    const { wgslCode, graphInputs } = compileToWGSL(graph);

    console.log('\n=== WGSL for sin/cos ===');
    console.log(wgslCode);
    console.log('========================\n');

    const kernel = new WGSLKernel(ctx.device, wgslCode, graphInputs);

    // Test inputs
    const batchInputs = new Float32Array([
      0, 0,                    // sin(0) + cos(0)*2 = 0 + 2 = 2
      Math.PI / 2, Math.PI,    // sin(π/2) + cos(π)*2 = 1 + (-1)*2 = -1
      Math.PI, 0               // sin(π) + cos(0)*2 = 0 + 2 = 2
    ]);

    const gpuResults = await kernel.execute(batchInputs, 3);

    // Compute expected results on CPU
    const expected = [
      Math.sin(0) + Math.cos(0) * 2,
      Math.sin(Math.PI / 2) + Math.cos(Math.PI) * 2,
      Math.sin(Math.PI) + Math.cos(0) * 2
    ];

    console.log('GPU Results:', Array.from(gpuResults));
    console.log('Expected:', expected);

    for (let i = 0; i < 3; i++) {
      expect(gpuResults[i]).toBeCloseTo(expected[i], 5);
    }
  });

  it('should match CPU execution exactly', async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    // Complex graph: (a + b) * (c - d) / 2
    const a = V.W(0);
    const b = V.W(0);
    const c = V.W(0);
    const d = V.W(0);

    const sum = V.add(a, b);
    const diff = V.sub(c, d);
    const prod = V.mul(sum, diff);
    const graph = V.div(prod, V.C(2));

    const { wgslCode, graphInputs } = compileToWGSL(graph);
    const kernel = new WGSLKernel(ctx.device, wgslCode, graphInputs);

    // Test batch
    const testCases = [
      [1, 2, 3, 4],    // (1+2)*(3-4)/2 = 3*(-1)/2 = -1.5
      [5, 5, 10, 2],   // (5+5)*(10-2)/2 = 10*8/2 = 40
      [-3, 7, 0, -1]   // (-3+7)*(0-(-1))/2 = 4*1/2 = 2
    ];

    const batchInputs = new Float32Array(testCases.flat());
    const gpuResults = await kernel.execute(batchInputs, testCases.length);

    // Compute on CPU using Value graph
    const cpuResults: number[] = [];
    for (const [av, bv, cv, dv] of testCases) {
      const aVal = V.W(av);
      const bVal = V.W(bv);
      const cVal = V.W(cv);
      const dVal = V.W(dv);

      const result = V.div(
        V.mul(
          V.add(aVal, bVal),
          V.sub(cVal, dVal)
        ),
        V.C(2)
      );

      cpuResults.push(result.data);
    }

    console.log('GPU Results:', Array.from(gpuResults));
    console.log('CPU Results:', cpuResults);

    // Results should match exactly (within floating-point tolerance)
    for (let i = 0; i < testCases.length; i++) {
      expect(gpuResults[i]).toBeCloseTo(cpuResults[i], 10);
    }

    const maxDiff = Math.max(...cpuResults.map((cpu, i) => Math.abs(cpu - gpuResults[i])));
    console.log(`Max GPU vs CPU diff: ${maxDiff.toExponential(6)}`);
    expect(maxDiff).toBeLessThan(1e-10);
  });

  it('should handle power and exponential operations', async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    // Graph: x^2 + exp(y)
    const x = V.W(0);
    const y = V.W(0);
    const graph = V.add(V.pow(x, 2), V.exp(y));

    const { wgslCode, graphInputs } = compileToWGSL(graph);
    const kernel = new WGSLKernel(ctx.device, wgslCode, graphInputs);

    const batchInputs = new Float32Array([
      2, 0,     // 2^2 + exp(0) = 4 + 1 = 5
      3, 1,     // 3^2 + exp(1) = 9 + e = 9 + 2.718... ≈ 11.718
      -2, -1    // (-2)^2 + exp(-1) = 4 + 1/e ≈ 4.368
    ]);

    const gpuResults = await kernel.execute(batchInputs, 3);

    const expected = [
      Math.pow(2, 2) + Math.exp(0),
      Math.pow(3, 2) + Math.exp(1),
      Math.pow(-2, 2) + Math.exp(-1)
    ];

    console.log('GPU Results:', Array.from(gpuResults));
    console.log('Expected:', expected);

    for (let i = 0; i < 3; i++) {
      expect(gpuResults[i]).toBeCloseTo(expected[i], 5);
    }
  });
});
