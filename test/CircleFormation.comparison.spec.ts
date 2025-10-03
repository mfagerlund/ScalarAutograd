import { describe, it, expect } from 'vitest';
import { V } from "../src/V";
import { Value } from "../src/Value";
import { SGD, Adam, AdamW } from "../src/Optimizers";
import { testLog } from './testUtils';

describe('Circle Formation - All Optimizers Comparison', () => {
  it('should compare NonlinearLeastSquares, SGD, Adam, and AdamW on circle fitting', { timeout: 10000 }, () => {
    const numPoints = 100;
    const trueCx = 10;
    const trueCy = -5;
    const trueR = 15;

    testLog('\n=== Circle Fitting - All Optimizers (100 noisy points, 3 parameters) ===\n');
    testLog('Problem: Fit circle (cx, cy, r) to 100 noisy observations');
    testLog('Parameters: 3 (center x, center y, radius)');
    testLog('Residuals: 100 (distance errors)\n');

    let seed = 12345;
    const seededRandom = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };

    const observations: { x: number; y: number }[] = [];
    for (let i = 0; i < numPoints; i++) {
      const angle = (i / numPoints) * 2 * Math.PI;
      const noise = (seededRandom() - 0.5) * 0.5;
      observations.push({
        x: trueCx + (trueR + noise) * Math.cos(angle),
        y: trueCy + (trueR + noise) * Math.sin(angle)
      });
    }

    function residuals(params: Value[]) {
      const [cx, cy, r] = params;
      return observations.map(p => {
        const dx = V.sub(V.C(p.x), cx);
        const dy = V.sub(V.C(p.y), cy);
        const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
        return V.sub(dist, r);
      });
    }

    function evaluateCost(params: Value[]): number {
      const res = residuals(params);
      return res.reduce((sum, r) => sum + r.data * r.data, 0);
    }

    const nlsCx = V.W(0);
    const nlsCy = V.W(0);
    const nlsR = V.W(5);

    const nlsStart = performance.now();
    const nlsResult = V.nonlinearLeastSquares([nlsCx, nlsCy, nlsR], residuals, {
      maxIterations: 100
    });
    const nlsTime = performance.now() - nlsStart;
    const nlsFinalCost = evaluateCost([nlsCx, nlsCy, nlsR]);

    testLog('--- Nonlinear Least Squares ---');
    testLog(`Iterations: ${nlsResult.iterations}`);
    testLog(`Final cost: ${nlsFinalCost.toExponential(4)}`);
    testLog(`Time: ${nlsTime.toFixed(2)}ms`);
    testLog(`Solution: cx=${nlsCx.data.toFixed(3)}, cy=${nlsCy.data.toFixed(3)}, r=${nlsR.data.toFixed(3)}`);
    testLog(`Convergence: ${nlsResult.convergenceReason}`);

    const sgdCx = V.W(0);
    const sgdCy = V.W(0);
    const sgdR = V.W(5);
    const sgdParams = [sgdCx, sgdCy, sgdR];

    const sgdOptimizer = new SGD(sgdParams, { learningRate: 0.01 });

    const sgdStart = performance.now();
    let sgdIterations = 0;
    let sgdFinalCost = 0;

    for (let i = 0; i < 10000; i++) {
      sgdFinalCost = evaluateCost(sgdParams);

      sgdIterations = i;

      if (sgdFinalCost < 5) break;

      const res = residuals(sgdParams);
      const loss = V.mean(res.map(r => V.square(r)));
      sgdOptimizer.zeroGrad();
      loss.backward();
      sgdOptimizer.step();
    }
    const sgdTime = performance.now() - sgdStart;

    testLog('\n--- Gradient Descent (SGD, lr=0.01) ---');
    testLog(`Iterations: ${sgdIterations}`);
    testLog(`Final cost: ${sgdFinalCost.toExponential(4)}`);
    testLog(`Time: ${sgdTime.toFixed(2)}ms`);
    testLog(`Solution: cx=${sgdCx.data.toFixed(3)}, cy=${sgdCy.data.toFixed(3)}, r=${sgdR.data.toFixed(3)}`);

    const adamCx = V.W(0);
    const adamCy = V.W(0);
    const adamR = V.W(5);
    const adamParams = [adamCx, adamCy, adamR];

    const adamOptimizer = new Adam(adamParams, { learningRate: 0.1 });

    const adamStart = performance.now();
    let adamIterations = 0;
    let adamFinalCost = 0;

    for (let i = 0; i < 10000; i++) {
      adamFinalCost = evaluateCost(adamParams);

      adamIterations = i;

      if (adamFinalCost < 5) break;

      const res = residuals(adamParams);
      const loss = V.mean(res.map(r => V.square(r)));
      adamOptimizer.zeroGrad();
      loss.backward();
      adamOptimizer.step();
    }
    const adamTime = performance.now() - adamStart;

    testLog('\n--- Adam (lr=0.1) ---');
    testLog(`Iterations: ${adamIterations}`);
    testLog(`Final cost: ${adamFinalCost.toExponential(4)}`);
    testLog(`Time: ${adamTime.toFixed(2)}ms`);
    testLog(`Solution: cx=${adamCx.data.toFixed(3)}, cy=${adamCy.data.toFixed(3)}, r=${adamR.data.toFixed(3)}`);

    const adamwCx = V.W(0);
    const adamwCy = V.W(0);
    const adamwR = V.W(5);
    const adamwParams = [adamwCx, adamwCy, adamwR];

    const adamwOptimizer = new AdamW(adamwParams, { learningRate: 0.1, weightDecay: 0 });

    const adamwStart = performance.now();
    let adamwIterations = 0;
    let adamwFinalCost = 0;

    for (let i = 0; i < 10000; i++) {
      adamwFinalCost = evaluateCost(adamwParams);

      adamwIterations = i;

      if (adamwFinalCost < 5) break;

      const res = residuals(adamwParams);
      const loss = V.mean(res.map(r => V.square(r)));
      adamwOptimizer.zeroGrad();
      loss.backward();
      adamwOptimizer.step();
    }
    const adamwTime = performance.now() - adamwStart;

    testLog('\n--- AdamW (lr=0.1, wd=0) ---');
    testLog(`Iterations: ${adamwIterations}`);
    testLog(`Final cost: ${adamwFinalCost.toExponential(4)}`);
    testLog(`Time: ${adamwTime.toFixed(2)}ms`);
    testLog(`Solution: cx=${adamwCx.data.toFixed(3)}, cy=${adamwCy.data.toFixed(3)}, r=${adamwR.data.toFixed(3)}`);

    testLog('\n=== Comparison Summary ===');
    testLog(`True values: cx=${trueCx}, cy=${trueCy}, r=${trueR}`);
    testLog('\nIterations:');
    testLog(`  NLS:   ${nlsResult.iterations}`);
    testLog(`  SGD:   ${sgdIterations} (${(sgdIterations / nlsResult.iterations).toFixed(1)}x)`);
    testLog(`  Adam:  ${adamIterations} (${(adamIterations / nlsResult.iterations).toFixed(1)}x)`);
    testLog(`  AdamW: ${adamwIterations} (${(adamwIterations / nlsResult.iterations).toFixed(1)}x)`);
    testLog('\nTime:');
    testLog(`  NLS:   ${nlsTime.toFixed(2)}ms`);
    testLog(`  SGD:   ${sgdTime.toFixed(2)}ms (${(sgdTime / nlsTime).toFixed(1)}x)`);
    testLog(`  Adam:  ${adamTime.toFixed(2)}ms (${(adamTime / nlsTime).toFixed(1)}x)`);
    testLog(`  AdamW: ${adamwTime.toFixed(2)}ms (${(adamwTime / nlsTime).toFixed(1)}x)`);
    testLog('\nFinal Cost:');
    testLog(`  NLS:   ${nlsFinalCost.toExponential(4)}`);
    testLog(`  SGD:   ${sgdFinalCost.toExponential(4)}`);
    testLog(`  Adam:  ${adamFinalCost.toExponential(4)}`);
    testLog(`  AdamW: ${adamwFinalCost.toExponential(4)}\n`);

    expect(nlsResult.success).toBe(true);
    expect(nlsResult.iterations).toBeLessThan(100);
    expect(nlsFinalCost).toBeLessThan(10);
    expect(sgdFinalCost).toBeLessThan(10);
    expect(adamFinalCost).toBeLessThan(10);
    expect(adamwFinalCost).toBeLessThan(10);
  });
});
