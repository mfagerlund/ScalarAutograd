import { describe, it, expect } from 'vitest';
import { V } from "../src/V";
import { Value } from "../src/Value";
import { SGD } from "../src/Optimizers";
import { Vec2 } from "../src/Vec2";
import { Vec3 } from "../src/Vec3";
import { testLog } from './testUtils';

describe('Nonlinear Least Squares vs Gradient Descent Comparison', () => {
  it('should compare on 2D point cloud alignment problem (50 points)', () => {
    const numPoints = 50;

    const targetPoints = Array.from({ length: numPoints }, (_, i) => ({
      x: Math.cos(i * 0.3) * 5 + Math.sin(i * 0.5) * 2,
      y: Math.sin(i * 0.3) * 5 + Math.cos(i * 0.5) * 2
    }));

    testLog('\n=== 2D Point Cloud Alignment (50 points, 100 parameters) ===\n');

    testLog('Problem: Align 50 movable points to target positions');
    testLog('Parameters: 100 (x,y for each point)');
    testLog('Residuals: 100 (2 per point)\n');

    const gnPoints = targetPoints.map(p =>
      Vec2.W(p.x + (Math.random() - 0.5) * 2, p.y + (Math.random() - 0.5) * 2)
    );

    function residuals(params: Value[]) {
      const res: Value[] = [];
      for (let i = 0; i < numPoints; i++) {
        const p = new Vec2(params[i * 2], params[i * 2 + 1]);
        const target = Vec2.C(targetPoints[i].x, targetPoints[i].y);
        const diff = p.sub(target);
        res.push(diff.x);
        res.push(diff.y);
      }
      return res;
    }

    const gnParams = gnPoints.flatMap(p => p.trainables);
    const gnStart = performance.now();
    const gnResult = V.nonlinearLeastSquares(gnParams, residuals, {
      maxIterations: 100,
      costTolerance: 1e-8
    });
    const gnTime = performance.now() - gnStart;

    testLog('--- Nonlinear Least Squares ---');
    testLog(`Iterations: ${gnResult.iterations}`);
    testLog(`Final cost: ${gnResult.finalCost.toExponential(4)}`);
    testLog(`Time: ${gnTime.toFixed(2)}ms`);
    testLog(`Convergence: ${gnResult.convergenceReason}`);

    const gdPoints = targetPoints.map(p =>
      Vec2.W(p.x + (Math.random() - 0.5) * 2, p.y + (Math.random() - 0.5) * 2)
    );
    const gdParams = gdPoints.flatMap(p => p.trainables);

    const optimizer = new SGD(gdParams, { learningRate: 0.1 });

    const gdStart = performance.now();
    let gdIterations = 0;
    let gdFinalCost = 0;

    for (let i = 0; i < 1000; i++) {
      const res = residuals(gdParams);
      const cost = res.reduce((sum, r) => sum + r.data * r.data, 0);

      if (i === 0 || i % 100 === 0) {
        gdIterations = i;
        gdFinalCost = cost;
        if (cost < 1e-8) break;
      }

      const loss = V.mean(res.map(r => V.square(r)));
      optimizer.zeroGrad();
      loss.backward();
      optimizer.step();
    }
    const gdTime = performance.now() - gdStart;

    testLog('\n--- Gradient Descent (SGD, lr=0.1) ---');
    testLog(`Iterations: ${gdIterations}`);
    testLog(`Final cost: ${gdFinalCost.toExponential(4)}`);
    testLog(`Time: ${gdTime.toFixed(2)}ms`);

    testLog('\n--- Speedup ---');
    testLog(`Time speedup: ${(gdTime / gnTime).toFixed(1)}x faster`);
    testLog(`Iteration speedup: ${(gdIterations / gnResult.iterations).toFixed(1)}x fewer iterations\n`);

    expect(gnResult.success).toBe(true);
    expect(gnResult.iterations).toBeLessThan(gdIterations);
    expect(gnTime).toBeLessThan(gdTime);
  });

  it.skip('should compare on 3D distance constraint problem (30 points)', () => {
    const numPoints = 30;

    testLog('\n=== 3D Distance Constraint Network (30 points, 90 parameters) ===\n');
    testLog('Problem: Points must maintain specific distances from neighbors');
    testLog('Parameters: 90 (x,y,z for each point)');
    testLog('Residuals: ~90 (distance constraints)\n');

    const targetDist = 2.0;

    const gnPoints = Array.from({ length: numPoints }, (_, i) => {
      const angle = (i / numPoints) * 2 * Math.PI;
      const radius = 5 + Math.random() * 2;
      return Vec3.W(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        (Math.random() - 0.5) * 3
      );
    });

    function residuals(params: Value[]) {
      const pts: Vec3[] = [];
      for (let i = 0; i < numPoints; i++) {
        pts.push(new Vec3(params[i * 3], params[i * 3 + 1], params[i * 3 + 2]));
      }

      const res: Value[] = [];

      for (let i = 0; i < numPoints; i++) {
        const next = (i + 1) % numPoints;
        const diff = pts[next].sub(pts[i]);
        const dist = diff.magnitude;
        res.push(V.sub(dist, V.C(targetDist)));
      }

      for (let i = 0; i < numPoints; i++) {
        const next2 = (i + 2) % numPoints;
        const diff = pts[next2].sub(pts[i]);
        const dist = diff.magnitude;
        res.push(V.sub(dist, V.C(targetDist * 1.8)));
      }

      return res;
    }

    const gnParams = gnPoints.flatMap(p => p.trainables);
    const gnStart = performance.now();
    const gnResult = V.nonlinearLeastSquares(gnParams, residuals, {
      maxIterations: 200,
      costTolerance: 1e-6
    });
    const gnTime = performance.now() - gnStart;

    testLog('--- Nonlinear Least Squares ---');
    testLog(`Iterations: ${gnResult.iterations}`);
    testLog(`Final cost: ${gnResult.finalCost.toExponential(4)}`);
    testLog(`Time: ${gnTime.toFixed(2)}ms`);
    testLog(`Convergence: ${gnResult.convergenceReason}`);

    const gdPoints = Array.from({ length: numPoints }, (_, i) => {
      const angle = (i / numPoints) * 2 * Math.PI;
      const radius = 5 + Math.random() * 2;
      return Vec3.W(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        (Math.random() - 0.5) * 3
      );
    });
    const gdParams = gdPoints.flatMap(p => p.trainables);

    const optimizer = new SGD(gdParams, { learningRate: 0.05 });

    const gdStart = performance.now();
    let gdIterations = 0;
    let gdFinalCost = 0;

    for (let i = 0; i < 2000; i++) {
      const res = residuals(gdParams);
      const cost = res.reduce((sum, r) => sum + r.data * r.data, 0);

      gdIterations = i;
      gdFinalCost = cost;

      if (cost < 1e-6) break;

      const loss = V.mean(res.map(r => V.square(r)));
      optimizer.zeroGrad();
      loss.backward();
      optimizer.step();
    }
    const gdTime = performance.now() - gdStart;

    testLog('\n--- Gradient Descent (SGD, lr=0.05) ---');
    testLog(`Iterations: ${gdIterations}`);
    testLog(`Final cost: ${gdFinalCost.toExponential(4)}`);
    testLog(`Time: ${gdTime.toFixed(2)}ms`);

    testLog('\n--- Speedup ---');
    testLog(`Time speedup: ${(gdTime / gnTime).toFixed(1)}x faster`);
    testLog(`Iteration speedup: ${(gdIterations / gnResult.iterations).toFixed(1)}x fewer iterations\n`);

    expect(gnResult.success).toBe(true);
    expect(gnResult.iterations).toBeLessThan(gdIterations / 5);
  });

  it.skip('should compare on circle fitting with 100 noisy points', { timeout: 20000 }, () => {
    const numPoints = 100;
    const trueCx = 10;
    const trueCy = -5;
    const trueR = 15;

    testLog('\n=== Circle Fitting (100 noisy points, 3 parameters) ===\n');
    testLog('Problem: Fit circle (cx, cy, r) to 100 noisy observations');
    testLog('Parameters: 3 (center x, center y, radius)');
    testLog('Residuals: 100 (distance errors)\n');

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

    const gnCx = V.W(0);
    const gnCy = V.W(0);
    const gnR = V.W(5);

    const gnStart = performance.now();
    const gnResult = V.nonlinearLeastSquares([gnCx, gnCy, gnR], residuals, {
      maxIterations: 100
    });
    const gnTime = performance.now() - gnStart;

    testLog('--- Nonlinear Least Squares ---');
    testLog(`Iterations: ${gnResult.iterations}`);
    testLog(`Final cost: ${gnResult.finalCost.toExponential(4)}`);
    testLog(`Time: ${gnTime.toFixed(2)}ms`);
    testLog(`Solution: cx=${gnCx.data.toFixed(3)}, cy=${gnCy.data.toFixed(3)}, r=${gnR.data.toFixed(3)}`);
    testLog(`True values: cx=${trueCx}, cy=${trueCy}, r=${trueR}`);

    const gdCx = V.W(0);
    const gdCy = V.W(0);
    const gdR = V.W(5);
    const gdParams = [gdCx, gdCy, gdR];

    const optimizer = new SGD(gdParams, { learningRate: 0.01 });

    const gdStart = performance.now();
    let gdIterations = 0;
    let gdFinalCost = 0;

    for (let i = 0; i < 10000; i++) {
      const res = residuals(gdParams);
      const cost = res.reduce((sum, r) => sum + r.data * r.data, 0);

      gdIterations = i;
      gdFinalCost = cost;

      if (cost < 1e-4) break;

      const loss = V.mean(res.map(r => V.square(r)));
      optimizer.zeroGrad();
      loss.backward();
      optimizer.step();
    }
    const gdTime = performance.now() - gdStart;

    testLog('\n--- Gradient Descent (SGD, lr=0.01) ---');
    testLog(`Iterations: ${gdIterations}`);
    testLog(`Final cost: ${gdFinalCost.toExponential(4)}`);
    testLog(`Time: ${gdTime.toFixed(2)}ms`);
    testLog(`Solution: cx=${gdCx.data.toFixed(3)}, cy=${gdCy.data.toFixed(3)}, r=${gdR.data.toFixed(3)}`);

    testLog('\n--- Speedup ---');
    testLog(`Time speedup: ${(gdTime / gnTime).toFixed(1)}x faster`);
    testLog(`Iteration speedup: ${(gdIterations / gnResult.iterations).toFixed(1)}x fewer iterations`);
    testLog(`Accuracy: GN error=${Math.abs(gnCx.data - trueCx).toFixed(3)}, GD error=${Math.abs(gdCx.data - trueCx).toFixed(3)}\n`);

    expect(gnResult.success).toBe(true);
    expect(gnResult.iterations).toBeLessThan(100);
    expect(gdIterations).toBeGreaterThan(gnResult.iterations * 10);
  });
});
