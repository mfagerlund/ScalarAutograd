/**
 * Performance benchmarks: L-BFGS compiled vs uncompiled
 */

import { V } from "../src/V";
import { lbfgs } from "../src/LBFGS";
import { testLog } from './testUtils';

describe('L-BFGS Compiled Performance', () => {
  it('should benchmark Rosenbrock: compiled vs uncompiled', () => {
    const objectiveFn = (p: V.Value[]) => {
      const [x, y] = p;
      const a = V.sub(V.C(1), x);
      const b = V.sub(y, V.pow(x, 2));
      return V.add(V.pow(a, 2), V.mul(V.C(100), V.pow(b, 2)));
    };

    // Uncompiled
    const x1 = V.W(-1.2);
    const y1 = V.W(1.0);
    const params1 = [x1, y1];

    const startUncompiled = performance.now();
    const result1 = lbfgs(params1, objectiveFn, { maxIterations: 100, verbose: false });
    const timeUncompiled = performance.now() - startUncompiled;

    // Compiled
    const x2 = V.W(-1.2);
    const y2 = V.W(1.0);
    const params2 = [x2, y2];
    const compiled = V.compileObjective(params2, objectiveFn);

    const startCompiled = performance.now();
    const result2 = lbfgs(params2, compiled, { maxIterations: 100, verbose: false });
    const timeCompiled = performance.now() - startCompiled;

    testLog('\n=== Rosenbrock Benchmark ===');
    testLog(`Uncompiled: ${timeUncompiled.toFixed(2)}ms, ${result1.iterations} iterations, ${result1.functionEvaluations} evals`);
    testLog(`Compiled:   ${timeCompiled.toFixed(2)}ms, ${result2.iterations} iterations, ${result2.functionEvaluations} evals`);
    testLog(`Speedup:    ${(timeUncompiled / timeCompiled).toFixed(2)}x`);
    testLog(`Per-eval speedup: ${(timeUncompiled / result1.functionEvaluations).toFixed(3)}ms vs ${(timeCompiled / result2.functionEvaluations).toFixed(3)}ms`);

    // Both should converge to same solution
    expect(x2.data).toBeCloseTo(x1.data, 3);
    expect(y2.data).toBeCloseTo(y1.data, 3);
  });

  it('should benchmark high-dimensional quadratic', () => {
    const n = 50;

    const objectiveFn = (p: V.Value[]) => {
      let sum = V.C(0);
      for (let i = 0; i < n; i++) {
        const diff = V.sub(p[i], V.C(i * 0.5));
        sum = V.add(sum, V.mul(diff, diff));
      }
      return sum;
    };

    // Uncompiled
    const params1 = Array.from({ length: n }, () => V.W(0));
    const startUncompiled = performance.now();
    const result1 = lbfgs(params1, objectiveFn, { maxIterations: 50, verbose: false });
    const timeUncompiled = performance.now() - startUncompiled;

    // Compiled
    const params2 = Array.from({ length: n }, () => V.W(0));
    const compiled = V.compileObjective(params2, objectiveFn);
    const startCompiled = performance.now();
    const result2 = lbfgs(params2, compiled, { maxIterations: 50, verbose: false });
    const timeCompiled = performance.now() - startCompiled;

    testLog('\n=== 50D Quadratic Benchmark ===');
    testLog(`Uncompiled: ${timeUncompiled.toFixed(2)}ms, ${result1.iterations} iterations, ${result1.functionEvaluations} evals`);
    testLog(`Compiled:   ${timeCompiled.toFixed(2)}ms, ${result2.iterations} iterations, ${result2.functionEvaluations} evals`);
    testLog(`Speedup:    ${(timeUncompiled / timeCompiled).toFixed(2)}x`);

    expect(result1.success).toBe(true);
    expect(result2.success).toBe(true);
  });

  it('should benchmark compilation overhead', () => {
    const objectiveFn = (p: V.Value[]) => {
      const [x, y] = p;
      const a = V.sub(V.C(1), x);
      const b = V.sub(y, V.pow(x, 2));
      return V.add(V.pow(a, 2), V.mul(V.C(100), V.pow(b, 2)));
    };

    const x = V.W(0);
    const y = V.W(0);
    const params = [x, y];

    // Measure compilation time
    const startCompile = performance.now();
    const compiled = V.compileObjective(params, objectiveFn);
    const compileTime = performance.now() - startCompile;

    // Measure single evaluation
    const startEval = performance.now();
    compiled.evaluate(params);
    const evalTime = performance.now() - startEval;

    testLog('\n=== Compilation Overhead ===');
    testLog(`Compilation time: ${compileTime.toFixed(3)}ms`);
    testLog(`Single eval time: ${evalTime.toFixed(3)}ms`);
    testLog(`Break-even: ~${Math.ceil(compileTime / evalTime)} evaluations`);
    testLog(`Typical L-BFGS: 50-200 evals → compilation is ${((50 * evalTime) / compileTime).toFixed(1)}x cheaper than runtime`);

    expect(compileTime).toBeLessThan(50); // Compilation should be fast
  });

  it('should show speedup increases with problem size', () => {
    const sizes = [5, 10, 20, 50];
    const speedups: number[] = [];

    testLog('\n=== Speedup vs Problem Size ===');

    for (const n of sizes) {
      const objectiveFn = (p: V.Value[]) => {
        let sum = V.C(0);
        for (let i = 0; i < n; i++) {
          sum = V.add(sum, V.pow(p[i], 2));
        }
        return sum;
      };

      // Uncompiled
      const params1 = Array.from({ length: n }, () => V.W(Math.random()));
      const start1 = performance.now();
      lbfgs(params1, objectiveFn, { maxIterations: 20, verbose: false });
      const time1 = performance.now() - start1;

      // Compiled
      const params2 = Array.from({ length: n }, () => V.W(Math.random()));
      const compiled = V.compileObjective(params2, objectiveFn);
      const start2 = performance.now();
      lbfgs(params2, compiled, { maxIterations: 20, verbose: false });
      const time2 = performance.now() - start2;

      const speedup = time1 / time2;
      speedups.push(speedup);

      testLog(`n=${n.toString().padStart(2)}: uncompiled=${time1.toFixed(2).padStart(6)}ms, compiled=${time2.toFixed(2).padStart(6)}ms, speedup=${speedup.toFixed(2)}x`);
    }

    // Speedup should generally increase with problem size
    // (Though not strictly monotonic due to timing variance)
    expect(speedups[speedups.length - 1]).toBeGreaterThan(1.5);
  });
});
