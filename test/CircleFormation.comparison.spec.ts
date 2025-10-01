import { describe, it, expect } from 'vitest';
import { V } from "../src/V";
import { Value } from "../src/Value";
import { SGD, Adam, AdamW } from "../src/Optimizers";

describe('Circle Formation - All Optimizers Comparison', () => {
  it('should compare NonlinearLeastSquares, SGD, Adam, and AdamW on circle fitting', { timeout: 10000 }, () => {
    const numPoints = 100;
    const trueCx = 10;
    const trueCy = -5;
    const trueR = 15;

    console.log('\n=== Circle Fitting - All Optimizers (100 noisy points, 3 parameters) ===\n');
    console.log('Problem: Fit circle (cx, cy, r) to 100 noisy observations');
    console.log('Parameters: 3 (center x, center y, radius)');
    console.log('Residuals: 100 (distance errors)\n');

    const observations: { x: number; y: number }[] = [];
    for (let i = 0; i < numPoints; i++) {
      const angle = (i / numPoints) * 2 * Math.PI;
      const noise = (Math.random() - 0.5) * 0.5;
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

    console.log('--- Nonlinear Least Squares ---');
    console.log(`Iterations: ${nlsResult.iterations}`);
    console.log(`Final cost: ${nlsFinalCost.toExponential(4)}`);
    console.log(`Time: ${nlsTime.toFixed(2)}ms`);
    console.log(`Solution: cx=${nlsCx.data.toFixed(3)}, cy=${nlsCy.data.toFixed(3)}, r=${nlsR.data.toFixed(3)}`);
    console.log(`Convergence: ${nlsResult.convergenceReason}`);

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

      if (sgdFinalCost < 1e-4) break;

      const res = residuals(sgdParams);
      const loss = V.mean(res.map(r => V.square(r)));
      sgdOptimizer.zeroGrad();
      loss.backward();
      sgdOptimizer.step();
    }
    const sgdTime = performance.now() - sgdStart;

    console.log('\n--- Gradient Descent (SGD, lr=0.01) ---');
    console.log(`Iterations: ${sgdIterations}`);
    console.log(`Final cost: ${sgdFinalCost.toExponential(4)}`);
    console.log(`Time: ${sgdTime.toFixed(2)}ms`);
    console.log(`Solution: cx=${sgdCx.data.toFixed(3)}, cy=${sgdCy.data.toFixed(3)}, r=${sgdR.data.toFixed(3)}`);

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

      if (adamFinalCost < 1e-4) break;

      const res = residuals(adamParams);
      const loss = V.mean(res.map(r => V.square(r)));
      adamOptimizer.zeroGrad();
      loss.backward();
      adamOptimizer.step();
    }
    const adamTime = performance.now() - adamStart;

    console.log('\n--- Adam (lr=0.1) ---');
    console.log(`Iterations: ${adamIterations}`);
    console.log(`Final cost: ${adamFinalCost.toExponential(4)}`);
    console.log(`Time: ${adamTime.toFixed(2)}ms`);
    console.log(`Solution: cx=${adamCx.data.toFixed(3)}, cy=${adamCy.data.toFixed(3)}, r=${adamR.data.toFixed(3)}`);

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

      if (adamwFinalCost < 1e-4) break;

      const res = residuals(adamwParams);
      const loss = V.mean(res.map(r => V.square(r)));
      adamwOptimizer.zeroGrad();
      loss.backward();
      adamwOptimizer.step();
    }
    const adamwTime = performance.now() - adamwStart;

    console.log('\n--- AdamW (lr=0.1, wd=0) ---');
    console.log(`Iterations: ${adamwIterations}`);
    console.log(`Final cost: ${adamwFinalCost.toExponential(4)}`);
    console.log(`Time: ${adamwTime.toFixed(2)}ms`);
    console.log(`Solution: cx=${adamwCx.data.toFixed(3)}, cy=${adamwCy.data.toFixed(3)}, r=${adamwR.data.toFixed(3)}`);

    console.log('\n=== Comparison Summary ===');
    console.log(`True values: cx=${trueCx}, cy=${trueCy}, r=${trueR}`);
    console.log('\nIterations:');
    console.log(`  NLS:   ${nlsResult.iterations}`);
    console.log(`  SGD:   ${sgdIterations} (${(sgdIterations / nlsResult.iterations).toFixed(1)}x)`);
    console.log(`  Adam:  ${adamIterations} (${(adamIterations / nlsResult.iterations).toFixed(1)}x)`);
    console.log(`  AdamW: ${adamwIterations} (${(adamwIterations / nlsResult.iterations).toFixed(1)}x)`);
    console.log('\nTime:');
    console.log(`  NLS:   ${nlsTime.toFixed(2)}ms`);
    console.log(`  SGD:   ${sgdTime.toFixed(2)}ms (${(sgdTime / nlsTime).toFixed(1)}x)`);
    console.log(`  Adam:  ${adamTime.toFixed(2)}ms (${(adamTime / nlsTime).toFixed(1)}x)`);
    console.log(`  AdamW: ${adamwTime.toFixed(2)}ms (${(adamwTime / nlsTime).toFixed(1)}x)`);
    console.log('\nFinal Cost:');
    console.log(`  NLS:   ${nlsFinalCost.toExponential(4)}`);
    console.log(`  SGD:   ${sgdFinalCost.toExponential(4)}`);
    console.log(`  Adam:  ${adamFinalCost.toExponential(4)}`);
    console.log(`  AdamW: ${adamwFinalCost.toExponential(4)}\n`);

    expect(nlsResult.success).toBe(true);
    expect(nlsResult.iterations).toBeLessThan(100);
    expect(nlsFinalCost).toBeLessThan(1e-3);
    expect(sgdFinalCost).toBeLessThan(1e-3);
    expect(adamFinalCost).toBeLessThan(1e-3);
    expect(adamwFinalCost).toBeLessThan(1e-3);
  });
});
