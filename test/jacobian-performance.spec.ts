/**
 * Performance comparison: old vs new Jacobian API
 */

import { V } from "../src/V";
import { CompiledResiduals } from "../src/CompiledResiduals";
import { testLog } from './testUtils';

describe("Jacobian API Performance", () => {
  it("should compare uncompiled vs compiled performance", () => {
    const numPoints = 100;
    const xData: number[] = [];
    const yData: number[] = [];

    // Generate test data
    for (let i = 0; i < numPoints; i++) {
      const x = (i / numPoints) * 10;
      const y = 2.0 * Math.exp(0.5 * x) + (Math.random() - 0.5) * 0.1;
      xData.push(x);
      yData.push(y);
    }

    testLog("\n=== PERFORMANCE COMPARISON ===");
    testLog(`Problem: Curve fitting with ${numPoints} data points, 2 parameters`);

    // Test 1: Uncompiled approach
    {
      const a = V.W(1.0, "a");
      const b = V.W(0.1, "b");
      const params = [a, b];

      const residualFn = (p: typeof params) => {
        const [a, b] = p;
        const res = [];
        for (let i = 0; i < numPoints; i++) {
          const pred = V.mul(a, V.exp(V.mul(b, V.C(xData[i]))));
          res.push(V.sub(pred, V.C(yData[i])));
        }
        return res;
      };

      const start = performance.now();
      const result = V.nonlinearLeastSquares(params, residualFn, {
        maxIterations: 10,
        useCompiled: false,
        verbose: false,
      });
      const time = performance.now() - start;

      testLog("\n1. UNCOMPILED:");
      testLog(`   Time: ${time.toFixed(2)}ms`);
      testLog(`   Iterations: ${result.iterations}`);
      testLog(`   Final cost: ${result.finalCost.toExponential(4)}`);
      testLog(`   Parameters: a=${a.data.toFixed(4)}, b=${b.data.toFixed(4)}`);
    }

    // Test 2: Compiled approach (new in-place API)
    {
      const a = V.W(1.0, "a");
      const b = V.W(0.1, "b");
      const params = [a, b];

      const residualFn = (p: typeof params) => {
        const [a, b] = p;
        const res = [];
        for (let i = 0; i < numPoints; i++) {
          const pred = V.mul(a, V.exp(V.mul(b, V.C(xData[i]))));
          res.push(V.sub(pred, V.C(yData[i])));
        }
        return res;
      };

      const start = performance.now();
      const result = V.nonlinearLeastSquares(params, residualFn, {
        maxIterations: 10,
        useCompiled: true,
        verbose: false,
      });
      const time = performance.now() - start;

      testLog("\n2. COMPILED (in-place Jacobian updates):");
      testLog(`   Time: ${time.toFixed(2)}ms`);
      testLog(`   Iterations: ${result.iterations}`);
      testLog(`   Final cost: ${result.finalCost.toExponential(4)}`);
      testLog(`   Parameters: a=${a.data.toFixed(4)}, b=${b.data.toFixed(4)}`);
    }

    testLog("\n==============================================\n");
  });

  it("should benchmark IK scenario: compile once, run many times", () => {
    const numPoints = 100;
    const xData: number[] = [];
    const yData: number[] = [];

    // Generate test data
    for (let i = 0; i < numPoints; i++) {
      const x = (i / numPoints) * 10;
      const y = 2.0 * Math.exp(0.5 * x) + (Math.random() - 0.5) * 0.1;
      xData.push(x);
      yData.push(y);
    }

    testLog("\n=== IK SCENARIO: Compile once, solve multiple times ===");
    testLog(`Problem: ${numPoints} residuals, 2 parameters`);
    testLog(`Scenario: Solve 10 different optimization problems with same structure\n`);

    // Uncompiled: solve 10 times
    let uncompiledTotal = 0;
    {
      for (let run = 0; run < 10; run++) {
        const a = V.W(1.0 + run * 0.1, "a");
        const b = V.W(0.1 + run * 0.01, "b");
        const params = [a, b];

        const residualFn = (p: typeof params) => {
          const [a, b] = p;
          const res = [];
          for (let i = 0; i < numPoints; i++) {
            const pred = V.mul(a, V.exp(V.mul(b, V.C(xData[i]))));
            res.push(V.sub(pred, V.C(yData[i])));
          }
          return res;
        };

        const start = performance.now();
        V.nonlinearLeastSquares(params, residualFn, {
          maxIterations: 10,
          useCompiled: false,
          verbose: false,
        });
        uncompiledTotal += performance.now() - start;
      }
    }

    // Compiled: compile once, solve 10 times
    let compiledTotal = 0;
    let compilationTime = 0;
    let solveTime = 0;
    {
      // Compile ONCE
      const a = V.W(1.0, "a");
      const b = V.W(0.1, "b");
      const params = [a, b];

      const residualFn = (p: typeof params) => {
        const [a, b] = p;
        const res = [];
        for (let i = 0; i < numPoints; i++) {
          const pred = V.mul(a, V.exp(V.mul(b, V.C(xData[i]))));
          res.push(V.sub(pred, V.C(yData[i])));
        }
        return res;
      };

      const compileStart = performance.now();
      const compiled = CompiledResiduals.compile(params, residualFn);
      compilationTime = performance.now() - compileStart;

      // Now solve 10 times using the SAME params with different values
      for (let run = 0; run < 10; run++) {
        // Update the SAME Value objects with new data
        a.data = 1.0 + run * 0.1;
        b.data = 0.1 + run * 0.01;

        const start = performance.now();
        V.nonlinearLeastSquares(params, compiled, {
          maxIterations: 10,
          verbose: false,
        });
        const runTime = performance.now() - start;
        compiledTotal += runTime;
        solveTime += runTime;
      }
    }

    testLog("UNCOMPILED (10 solves):");
    testLog(`  Total time: ${uncompiledTotal.toFixed(2)}ms`);
    testLog(`  Per solve: ${(uncompiledTotal / 10).toFixed(2)}ms`);

    testLog("\nCOMPILED (compile once, solve 10 times):");
    testLog(`  Compilation time: ${compilationTime.toFixed(2)}ms`);
    testLog(`  10 solves: ${solveTime.toFixed(2)}ms`);
    testLog(`  Per solve: ${(solveTime / 10).toFixed(2)}ms`);
    testLog(`  Total (including compilation): ${(compilationTime + compiledTotal).toFixed(2)}ms`);

    const totalWithCompilation = compilationTime + compiledTotal;
    const speedup = uncompiledTotal / totalWithCompilation;
    testLog(`\n  Speedup (including compilation): ${speedup.toFixed(2)}x ${speedup > 1 ? "FASTER" : "SLOWER"}`);

    // Excluding compilation overhead
    const avgUncompiled = uncompiledTotal / 10;
    const avgCompiled = solveTime / 10;
    const runtimeSpeedup = avgUncompiled / avgCompiled;
    testLog(`  Runtime speedup (pure solve time): ${runtimeSpeedup.toFixed(2)}x ${runtimeSpeedup > 1 ? "FASTER" : "SLOWER"}`);

    testLog("\n==============================================\n");
  });

  it("should verify correctness of compiled vs uncompiled", () => {
    const a1 = V.W(1.0, "a");
    const b1 = V.W(0.1, "b");
    const params1 = [a1, b1];

    const a2 = V.W(1.0, "a");
    const b2 = V.W(0.1, "b");
    const params2 = [a2, b2];

    const xData = [0, 1, 2, 3, 4];
    const yData = [2.0, 3.3, 5.4, 9.0, 14.8];

    const residualFn1 = (p: typeof params1) => {
      const [a, b] = p;
      return xData.map((x, i) => {
        const pred = V.mul(a, V.exp(V.mul(b, V.C(x))));
        return V.sub(pred, V.C(yData[i]));
      });
    };

    const residualFn2 = (p: typeof params2) => {
      const [a, b] = p;
      return xData.map((x, i) => {
        const pred = V.mul(a, V.exp(V.mul(b, V.C(x))));
        return V.sub(pred, V.C(yData[i]));
      });
    };

    const result1 = V.nonlinearLeastSquares(params1, residualFn1, {
      maxIterations: 20,
      useCompiled: false,
    });

    const result2 = V.nonlinearLeastSquares(params2, residualFn2, {
      maxIterations: 20,
      useCompiled: true,
    });

    testLog("\n=== CORRECTNESS CHECK ===");
    testLog("Uncompiled result:");
    testLog(`  a=${a1.data.toFixed(6)}, b=${b1.data.toFixed(6)}`);
    testLog(`  cost=${result1.finalCost.toExponential(6)}`);
    testLog(`  iterations=${result1.iterations}`);

    testLog("\nCompiled result:");
    testLog(`  a=${a2.data.toFixed(6)}, b=${b2.data.toFixed(6)}`);
    testLog(`  cost=${result2.finalCost.toExponential(6)}`);
    testLog(`  iterations=${result2.iterations}`);

    // Should produce the same results
    expect(a2.data).toBeCloseTo(a1.data, 4);
    expect(b2.data).toBeCloseTo(b1.data, 4);
    expect(result2.finalCost).toBeCloseTo(result1.finalCost, 4);
  });
});
