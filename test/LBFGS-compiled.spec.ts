/**
 * Tests for L-BFGS with compiled objective functions
 */

import { V } from "../src/V";
import { lbfgs } from "../src/LBFGS";
import { testLog } from './testUtils';

describe('L-BFGS with Compiled Objectives', () => {
  it('should solve quadratic with compiled objective', () => {
    // Minimize f(x) = (x - 3)²
    const x = V.W(0);
    const params = [x];

    const objectiveFn = (p: typeof params) => {
      const [x] = p;
      const diff = V.sub(x, V.C(3));
      return V.mul(diff, diff);
    };

    const compiled = V.compileObjective(params, objectiveFn);
    const result = lbfgs(params, compiled);

    expect(result.success).toBe(true);
    expect(x.data).toBeCloseTo(3.0, 5);
    expect(result.finalCost).toBeCloseTo(0, 8);
  });

  it('should match uncompiled results on Rosenbrock function', () => {
    // Rosenbrock: f(x, y) = (1-x)² + 100(y-x²)²
    const objectiveFn = (p: V.Value[]) => {
      const [x, y] = p;
      const a = V.sub(V.C(1), x);
      const b = V.sub(y, V.pow(x, 2));
      return V.add(V.pow(a, 2), V.mul(V.C(100), V.pow(b, 2)));
    };

    // Test with uncompiled
    const x1 = V.W(0.5);
    const y1 = V.W(0.5);
    const params1 = [x1, y1];
    const result1 = lbfgs(params1, objectiveFn, { maxIterations: 100 });

    // Test with compiled
    const x2 = V.W(0.5);
    const y2 = V.W(0.5);
    const params2 = [x2, y2];
    const compiled = V.compileObjective(params2, objectiveFn);
    const result2 = lbfgs(params2, compiled, { maxIterations: 100 });

    testLog('\n=== Rosenbrock: Compiled vs Uncompiled ===');
    testLog(`Uncompiled: x=${x1.data.toFixed(6)}, y=${y1.data.toFixed(6)}, cost=${result1.finalCost.toExponential(4)}, iter=${result1.iterations}`);
    testLog(`Compiled:   x=${x2.data.toFixed(6)}, y=${y2.data.toFixed(6)}, cost=${result2.finalCost.toExponential(4)}, iter=${result2.iterations}`);

    // Both should converge to similar solutions (Rosenbrock is hard, allow some variance)
    expect(result1.success).toBe(true);
    expect(result2.success).toBe(true);
    expect(x2.data).toBeCloseTo(x1.data, 1);  // Within 0.05
    expect(y2.data).toBeCloseTo(y1.data, 1);  // Within 0.05
    expect(result2.finalCost).toBeCloseTo(result1.finalCost, 3);
  });

  it('should handle multi-dimensional quadratic', () => {
    // Minimize f(x) = Σ(xᵢ - i)²
    const n = 10;
    const params = Array.from({ length: n }, (_, i) => V.W(0, `x${i}`));

    const objectiveFn = (p: typeof params) => {
      let sum = V.C(0);
      for (let i = 0; i < n; i++) {
        const diff = V.sub(p[i], V.C(i));
        sum = V.add(sum, V.mul(diff, diff));
      }
      return sum;
    };

    const compiled = V.compileObjective(params, objectiveFn);
    const result = lbfgs(params, compiled);

    testLog(`\n=== 10D Quadratic ===`);
    testLog(`Success: ${result.success}, Iterations: ${result.iterations}, Final cost: ${result.finalCost.toExponential(4)}`);

    expect(result.success).toBe(true);
    for (let i = 0; i < n; i++) {
      expect(params[i].data).toBeCloseTo(i, 5);
    }
  });

  it('should work with Beale function', () => {
    // Beale: f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
    // Global minimum at (3, 0.5) with f = 0
    const objectiveFn = (p: V.Value[]) => {
      const [x, y] = p;
      const t1 = V.sub(V.add(V.C(1.5), V.mul(x, V.sub(y, V.C(1)))), V.C(0));
      const t2 = V.sub(V.add(V.C(2.25), V.mul(x, V.sub(V.pow(y, 2), V.C(1)))), V.C(0));
      const t3 = V.sub(V.add(V.C(2.625), V.mul(x, V.sub(V.pow(y, 3), V.C(1)))), V.C(0));
      return V.add(V.add(V.pow(t1, 2), V.pow(t2, 2)), V.pow(t3, 2));
    };

    const x = V.W(1.0);
    const y = V.W(1.0);
    const params = [x, y];

    const compiled = V.compileObjective(params, objectiveFn);
    const result = lbfgs(params, compiled, { maxIterations: 200 });

    testLog(`\n=== Beale Function ===`);
    testLog(`x=${x.data.toFixed(6)}, y=${y.data.toFixed(6)}, cost=${result.finalCost.toExponential(4)}, iter=${result.iterations}`);

    expect(result.success).toBe(true);
    expect(x.data).toBeCloseTo(3.0, 1);  // Beale is tricky, accept 0.05 tolerance
    expect(y.data).toBeCloseTo(0.5, 1);
  });

  it('should handle unused parameters gracefully', () => {
    // f(x, y) = x² (y is unused)
    const x = V.W(5);
    const y = V.W(10);
    const params = [x, y];

    const objectiveFn = (p: typeof params) => {
      const [x] = p;
      return V.mul(x, x);
    };

    const compiled = V.compileObjective(params, objectiveFn);
    const result = lbfgs(params, compiled);

    expect(result.success).toBe(true);
    expect(x.data).toBeCloseTo(0, 5);
    // y should be unchanged (gradient is zero)
    expect(y.data).toBeCloseTo(10, 5);
  });

  it('should work with exponential functions', () => {
    // Minimize f(x) = e^x + e^(-x) - 2
    // Minimum at x=0 with f=0
    const x = V.W(2.0);
    const params = [x];

    const objectiveFn = (p: typeof params) => {
      const [x] = p;
      return V.sub(V.add(V.exp(x), V.exp(V.mul(x, V.C(-1)))), V.C(2));
    };

    const compiled = V.compileObjective(params, objectiveFn);
    const result = lbfgs(params, compiled);

    testLog(`\n=== Exponential Function ===`);
    testLog(`x=${x.data.toFixed(6)}, cost=${result.finalCost.toExponential(4)}, iter=${result.iterations}`);

    expect(result.success).toBe(true);
    expect(x.data).toBeCloseTo(0, 3);
    expect(result.finalCost).toBeCloseTo(0, 6);
  });

  it('should verify gradient correctness with numerical check', () => {
    // Use simple quadratic for easy verification
    const x = V.W(2.0);
    const y = V.W(3.0);
    const params = [x, y];

    const objectiveFn = (p: typeof params) => {
      const [x, y] = p;
      return V.add(V.mul(x, x), V.mul(y, y));
    };

    const compiled = V.compileObjective(params, objectiveFn);

    // Evaluate compiled gradient - J is the Jacobian matrix, first row is the gradient
    const { J } = compiled.evaluate(params);
    const compiledGradient = J[0];

    // Compute gradient via uncompiled path
    params.forEach(p => p.grad = 0);
    const obj = objectiveFn(params);
    obj.backward();
    const graphGradient = params.map(p => p.grad);

    testLog(`\n=== Gradient Verification ===`);
    testLog(`Compiled gradient: [${compiledGradient.map(g => g.toFixed(6)).join(', ')}]`);
    testLog(`Graph gradient:    [${graphGradient.map(g => g.toFixed(6)).join(', ')}]`);

    expect(compiledGradient[0]).toBeCloseTo(graphGradient[0], 12);
    expect(compiledGradient[1]).toBeCloseTo(graphGradient[1], 12);
  });
});
