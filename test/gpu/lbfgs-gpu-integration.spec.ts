/**
 * Test GPU integration with L-BFGS optimizer
 */

import { V, Value, lbfgs } from '../../src';
import { WebGPUContext } from '../../src/gpu/WebGPUContext';
import { GPUCompiledFunctions } from '../../src/gpu/GPUCompiledFunctions';

describe('L-BFGS GPU Integration', () => {
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

  it('should optimize Rosenbrock function with GPU', async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    // Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    // Minimum at (1, 1)
    const x = V.W(-1.2);
    const y = V.W(1.0);
    const params = [x, y];

    // Build objective as sum of residuals
    const gpuCompiled = await GPUCompiledFunctions.compile(params, (p: Value[]) => {
      const a = V.sub(V.C(1), p[0]);  // (1-x)
      const b = V.sub(p[1], V.pow(p[0], 2));  // (y-x^2)
      return [
        a,  // Residual 1: (1-x)
        V.mul(V.C(10), b)  // Residual 2: 10*(y-x^2)  [scaled to match 100 in sum of squares]
      ];
    });

    console.log('\\n=== GPU L-BFGS Optimization ===');
    console.log(`Initial: x=${x.data.toFixed(4)}, y=${y.data.toFixed(4)}`);

    // Create synchronous wrapper for L-BFGS compatibility
    const syncWrapper = {
      evaluateSumWithGradient: (p: Value[]) => {
        // Block on the promise - not ideal but works for now
        let result: any;
        gpuCompiled.evaluateSumWithGradient(p).then(r => result = r);
        // Spin wait (horrible but simple for testing)
        const start = Date.now();
        while (!result && Date.now() - start < 5000) { /* wait */ }
        return result;
      }
    };

    const result = lbfgs(params, syncWrapper as any, {
      maxIterations: 50,
      verbose: false,
      gradientTolerance: 1e-6
    });

    console.log(`Final:   x=${x.data.toFixed(4)}, y=${y.data.toFixed(4)}`);
    console.log(`Cost:    ${result.finalCost.toFixed(8)}`);
    console.log(`Iterations: ${result.iterations}`);
    console.log(`Reason:  ${result.convergenceReason}`);

    // Should converge close to (1, 1)
    expect(x.data).toBeCloseTo(1.0, 2);
    expect(y.data).toBeCloseTo(1.0, 2);
    expect(result.finalCost).toBeLessThan(1e-6);
  }, 30000);

  it('should optimize multiple residuals problem', async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    // Fit points to a parabola: y = ax^2 + bx + c
    const data = [
      { x: -2, y: 6 },
      { x: -1, y: 2 },
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 2 }
    ];

    // True parabola: y = x^2 - x  (a=1, b=-1, c=0)
    const a = V.W(0);  // Start with bad guess
    const b = V.W(0);
    const c = V.W(0);
    const params = [a, b, c];

    const gpuCompiled = await GPUCompiledFunctions.compile(params, (p: Value[]) => {
      return data.map(point => {
        // Residual: (ax^2 + bx + c) - y
        const yPred = V.add(
          V.add(
            V.mul(p[0], V.C(point.x * point.x)),
            V.mul(p[1], V.C(point.x))
          ),
          p[2]
        );
        return V.sub(yPred, V.C(point.y));
      });
    });

    console.log('\\n=== GPU Parabola Fitting ===');
    console.log(`Initial: a=${a.data.toFixed(4)}, b=${b.data.toFixed(4)}, c=${c.data.toFixed(4)}`);

    const result = await lbfgs(params, gpuCompiled as any, {
      maxIterations: 100,
      verbose: false
    });

    console.log(`Final:   a=${a.data.toFixed(4)}, b=${b.data.toFixed(4)}, c=${c.data.toFixed(4)}`);
    console.log(`Cost:    ${result.finalCost.toFixed(8)}`);
    console.log(`Iterations: ${result.iterations}`);

    // Should find a≈1, b≈-1, c≈0
    expect(a.data).toBeCloseTo(1.0, 1);
    expect(b.data).toBeCloseTo(-1.0, 1);
    expect(c.data).toBeCloseTo(0.0, 1);
  }, 30000);
});
