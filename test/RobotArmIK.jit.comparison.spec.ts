import { describe, it, expect } from 'vitest';
import { V } from "../src/V";
import { Value } from "../src/Value";
import { SGD, Adam } from "../src/Optimizers";
import { compileGradientFunction, applyCompiledGradients } from "../src/jit-compile-value";

interface ArmSegment {
  length: number;
}

function forwardKinematics(angles: Value[], segments: ArmSegment[]): { x: Value; y: Value } {
  let x = V.C(0);
  let y = V.C(0);
  let cumulativeAngle = V.C(0);

  for (let i = 0; i < segments.length; i++) {
    cumulativeAngle = V.add(cumulativeAngle, angles[i]);
    const segmentX = V.mul(V.C(segments[i].length), V.cos(cumulativeAngle));
    const segmentY = V.mul(V.C(segments[i].length), V.sin(cumulativeAngle));
    x = V.add(x, segmentX);
    y = V.add(y, segmentY);
  }

  return { x, y };
}

describe('Robot Arm IK - Graph vs JIT Compiled Comparison', () => {
  it('Demonstrates NLS (non-JIT) vs SGD (Graph) vs SGD (Compiled)', { timeout: 10000 }, () => {
    const segments: ArmSegment[] = [
      { length: 3.0 },
      { length: 2.5 },
      { length: 2.0 }
    ];

    const targetX = 5.0;
    const targetY = 4.0;

    console.log('\n=== Comparison: NLS vs SGD-Graph vs SGD-Compiled ===');
    console.log(`Target: (${targetX}, ${targetY})\n`);

    function residuals(params: Value[]) {
      const endEffector = forwardKinematics(params, segments);
      return [
        V.sub(endEffector.x, V.C(targetX)),
        V.sub(endEffector.y, V.C(targetY))
      ];
    }

    function evaluateCost(params: Value[]): number {
      const res = residuals(params);
      return res.reduce((sum, r) => sum + r.data * r.data, 0);
    }

    const nlsAngles = segments.map(() => V.W(0.1));

    console.log('--- Nonlinear Least Squares (uses autodiff, not yet JIT-optimized) ---');
    const nlsStart = performance.now();
    const nlsResult = V.nonlinearLeastSquares(nlsAngles, residuals, {
      maxIterations: 100,
      costTolerance: 1e-8
    });
    const nlsTime = performance.now() - nlsStart;
    const nlsFinalCost = evaluateCost(nlsAngles);
    const nlsEndEffector = forwardKinematics(nlsAngles, segments);

    console.log(`Iterations: ${nlsResult.iterations}`);
    console.log(`Final cost: ${nlsFinalCost.toExponential(4)}`);
    console.log(`Time: ${nlsTime.toFixed(2)}ms`);
    console.log(`End effector: (${nlsEndEffector.x.data.toFixed(3)}, ${nlsEndEffector.y.data.toFixed(3)})`);

    const graphAngles = segments.map(() => V.W(0.1));
    const graphOptimizer = new SGD(graphAngles, { learningRate: 0.1 });

    console.log('\n--- SGD with Graph-based gradients ---');
    const graphStart = performance.now();
    let graphIterations = 0;
    let graphFinalCost = 0;

    for (let i = 0; i < 10000; i++) {
      graphFinalCost = evaluateCost(graphAngles);
      graphIterations = i;
      if (graphFinalCost < 1e-8) break;

      const res = residuals(graphAngles);
      const loss = V.mean(res.map(r => V.square(r)));
      graphOptimizer.zeroGrad();
      loss.backward();
      graphOptimizer.step();
    }
    const graphTime = performance.now() - graphStart;

    console.log(`Iterations: ${graphIterations}`);
    console.log(`Final cost: ${graphFinalCost.toExponential(4)}`);
    console.log(`Time: ${graphTime.toFixed(2)}ms`);

    const compAngles = segments.map((_, i) => {
      const v = V.W(0.1);
      v.paramName = `angle${i}`;
      return v;
    });
    const compOptimizer = new SGD(compAngles, { learningRate: 0.1 });

    const lossGraph = (() => {
      const res = residuals(compAngles);
      return V.mean(res.map(r => V.square(r)));
    })();
    const compiledGradFn = compileGradientFunction(lossGraph, compAngles);

    console.log('\n--- SGD with JIT-Compiled gradients ---');
    const compStart = performance.now();
    let compIterations = 0;
    let compFinalCost = 0;

    for (let i = 0; i < 10000; i++) {
      compFinalCost = evaluateCost(compAngles);
      compIterations = i;
      if (compFinalCost < 1e-8) break;

      compOptimizer.zeroGrad();
      applyCompiledGradients(compiledGradFn, compAngles);
      compOptimizer.step();
    }
    const compTime = performance.now() - compStart;

    console.log(`Iterations: ${compIterations}`);
    console.log(`Final cost: ${compFinalCost.toExponential(4)}`);
    console.log(`Time: ${compTime.toFixed(2)}ms`);

    console.log('\n=== Three-Way Performance Summary ===');
    console.log('Method           | Iters  | Time      | Speedup vs NLS | Speedup vs Graph');
    console.log('-----------------|--------|-----------|----------------|------------------');
    console.log(`NLS (autodiff)   | ${nlsResult.iterations.toString().padEnd(6)} | ${nlsTime.toFixed(2).padEnd(9)} | 1.00x          | ${(nlsTime / graphTime).toFixed(2)}x`);
    console.log(`SGD (graph)      | ${graphIterations.toString().padEnd(6)} | ${graphTime.toFixed(2).padEnd(9)} | ${(graphTime / nlsTime).toFixed(2)}x          | 1.00x`);
    console.log(`SGD (compiled)   | ${compIterations.toString().padEnd(6)} | ${compTime.toFixed(2).padEnd(9)} | ${(compTime / nlsTime).toFixed(2)}x          | ${(graphTime / compTime).toFixed(2)}x`);
    console.log('\nNotes:');
    console.log('- NLS uses autodiff (.backward() per residual) but not yet JIT-optimized');
    console.log('- NLS could be JIT compiled but requires compiling each residual separately');
    console.log('- SGD iterations are higher but each is faster with JIT compilation');
    console.log(`- JIT provides ${(graphTime / compTime).toFixed(1)}x speedup over graph-based gradients\n`);

    expect(nlsFinalCost).toBeLessThan(1e-6);
    expect(Math.abs(graphFinalCost - compFinalCost)).toBeLessThan(1e-6);
    expect(compTime).toBeLessThan(graphTime);
  });


  it('3-joint arm: SGD with Graph vs Compiled gradients', { timeout: 10000 }, () => {
    const segments: ArmSegment[] = [
      { length: 3.0 },
      { length: 2.5 },
      { length: 2.0 }
    ];

    const targetX = 5.0;
    const targetY = 4.0;

    console.log('\n=== 3-Joint IK: Graph vs Compiled (SGD) ===');
    console.log(`Target: (${targetX}, ${targetY})\n`);

    function residuals(params: Value[]) {
      const endEffector = forwardKinematics(params, segments);
      return [
        V.sub(endEffector.x, V.C(targetX)),
        V.sub(endEffector.y, V.C(targetY))
      ];
    }

    function evaluateCost(params: Value[]): number {
      const res = residuals(params);
      return res.reduce((sum, r) => sum + r.data * r.data, 0);
    }

    const graphAngles = segments.map(() => V.W(0.1));
    const graphOptimizer = new SGD(graphAngles, { learningRate: 0.1 });

    console.log('--- Graph-based SGD ---');
    const graphStart = performance.now();
    let graphIterations = 0;
    let graphFinalCost = 0;

    for (let i = 0; i < 10000; i++) {
      graphFinalCost = evaluateCost(graphAngles);
      graphIterations = i;
      if (graphFinalCost < 1e-8) break;

      const res = residuals(graphAngles);
      const loss = V.mean(res.map(r => V.square(r)));
      graphOptimizer.zeroGrad();
      loss.backward();
      graphOptimizer.step();
    }
    const graphTime = performance.now() - graphStart;
    const graphEndEffector = forwardKinematics(graphAngles, segments);

    console.log(`Iterations: ${graphIterations}`);
    console.log(`Final cost: ${graphFinalCost.toExponential(4)}`);
    console.log(`Time: ${graphTime.toFixed(2)}ms`);
    console.log(`End effector: (${graphEndEffector.x.data.toFixed(3)}, ${graphEndEffector.y.data.toFixed(3)})`);

    const compAngles = segments.map((_, i) => {
      const v = V.W(0.1);
      v.paramName = `angle${i}`;
      return v;
    });
    const compOptimizer = new SGD(compAngles, { learningRate: 0.1 });

    const lossGraph = (() => {
      const res = residuals(compAngles);
      return V.mean(res.map(r => V.square(r)));
    })();
    const compiledGradFn = compileGradientFunction(lossGraph, compAngles);

    console.log('\n--- JIT Compiled SGD ---');
    const compStart = performance.now();
    let compIterations = 0;
    let compFinalCost = 0;

    for (let i = 0; i < 10000; i++) {
      compFinalCost = evaluateCost(compAngles);
      compIterations = i;
      if (compFinalCost < 1e-8) break;

      compOptimizer.zeroGrad();
      applyCompiledGradients(compiledGradFn, compAngles);
      compOptimizer.step();
    }
    const compTime = performance.now() - compStart;
    const compEndEffector = forwardKinematics(compAngles, segments);

    console.log(`Iterations: ${compIterations}`);
    console.log(`Final cost: ${compFinalCost.toExponential(4)}`);
    console.log(`Time: ${compTime.toFixed(2)}ms`);
    console.log(`End effector: (${compEndEffector.x.data.toFixed(3)}, ${compEndEffector.y.data.toFixed(3)})`);

    console.log('\n--- Performance Summary ---');
    console.log(`Iterations: ${graphIterations} (graph) vs ${compIterations} (compiled)`);
    console.log(`Time: ${graphTime.toFixed(2)}ms (graph) vs ${compTime.toFixed(2)}ms (compiled)`);
    console.log(`Speedup: ${(graphTime / compTime).toFixed(2)}x`);
    console.log(`Final cost difference: ${Math.abs(graphFinalCost - compFinalCost).toExponential(2)}\n`);

    expect(compIterations).toBe(graphIterations);
    expect(Math.abs(graphFinalCost - compFinalCost)).toBeLessThan(1e-10);
    expect(compTime).toBeLessThan(graphTime);
  });

  it('8-joint arm: Adam with Graph vs Compiled gradients', { timeout: 30000 }, () => {
    const segments: ArmSegment[] = [
      { length: 4.0 },
      { length: 3.5 },
      { length: 3.0 },
      { length: 2.5 },
      { length: 2.0 },
      { length: 1.5 },
      { length: 1.0 },
      { length: 0.8 }
    ];

    const targetX = 8.0;
    const targetY = 12.0;

    console.log('\n=== 8-Joint IK (THE BEAST): Graph vs Compiled (Adam) ===');
    console.log(`Target: (${targetX}, ${targetY})\n`);

    function residuals(params: Value[]) {
      const endEffector = forwardKinematics(params, segments);
      return [
        V.sub(endEffector.x, V.C(targetX)),
        V.sub(endEffector.y, V.C(targetY))
      ];
    }

    function evaluateCost(params: Value[]): number {
      const res = residuals(params);
      return res.reduce((sum, r) => sum + r.data * r.data, 0);
    }

    const graphAngles = segments.map((_, i) => V.W(0.2 + i * 0.05));
    const graphOptimizer = new Adam(graphAngles, { learningRate: 0.1 });

    console.log('--- Graph-based Adam ---');
    const graphStart = performance.now();
    let graphIterations = 0;
    let graphFinalCost = 0;

    for (let i = 0; i < 20000; i++) {
      graphFinalCost = evaluateCost(graphAngles);
      graphIterations = i;
      if (graphFinalCost < 1e-8) break;

      const res = residuals(graphAngles);
      const loss = V.mean(res.map(r => V.square(r)));
      graphOptimizer.zeroGrad();
      loss.backward();
      graphOptimizer.step();
    }
    const graphTime = performance.now() - graphStart;
    const graphEndEffector = forwardKinematics(graphAngles, segments);

    console.log(`Iterations: ${graphIterations}`);
    console.log(`Final cost: ${graphFinalCost.toExponential(4)}`);
    console.log(`Time: ${graphTime.toFixed(2)}ms`);
    console.log(`End effector: (${graphEndEffector.x.data.toFixed(3)}, ${graphEndEffector.y.data.toFixed(3)})`);

    const compAngles = segments.map((_, i) => {
      const v = V.W(0.2 + i * 0.05);
      v.paramName = `angle${i}`;
      return v;
    });
    const compOptimizer = new Adam(compAngles, { learningRate: 0.1 });

    const lossGraph = (() => {
      const res = residuals(compAngles);
      return V.mean(res.map(r => V.square(r)));
    })();
    const compiledGradFn = compileGradientFunction(lossGraph, compAngles);

    console.log('\n--- JIT Compiled Adam ---');
    const compStart = performance.now();
    let compIterations = 0;
    let compFinalCost = 0;

    for (let i = 0; i < 20000; i++) {
      compFinalCost = evaluateCost(compAngles);
      compIterations = i;
      if (compFinalCost < 1e-8) break;

      compOptimizer.zeroGrad();
      applyCompiledGradients(compiledGradFn, compAngles);
      compOptimizer.step();
    }
    const compTime = performance.now() - compStart;
    const compEndEffector = forwardKinematics(compAngles, segments);

    console.log(`Iterations: ${compIterations}`);
    console.log(`Final cost: ${compFinalCost.toExponential(4)}`);
    console.log(`Time: ${compTime.toFixed(2)}ms`);
    console.log(`End effector: (${compEndEffector.x.data.toFixed(3)}, ${compEndEffector.y.data.toFixed(3)})`);

    console.log('\n--- Performance Summary ---');
    console.log(`Iterations: ${graphIterations} (graph) vs ${compIterations} (compiled)`);
    console.log(`Time: ${graphTime.toFixed(2)}ms (graph) vs ${compTime.toFixed(2)}ms (compiled)`);
    console.log(`Speedup: ${(graphTime / compTime).toFixed(2)}x`);
    console.log(`Final cost difference: ${Math.abs(graphFinalCost - compFinalCost).toExponential(2)}\n`);

    expect(compIterations).toBe(graphIterations);
    expect(Math.abs(graphFinalCost - compFinalCost)).toBeLessThan(1e-10);
    expect(compTime).toBeLessThan(graphTime);
  });

  it('15-joint arm: Adam with Graph vs Compiled - ABSOLUTE UNIT', { timeout: 60000 }, () => {
    const segments: ArmSegment[] = Array.from({ length: 15 }, (_, i) => ({
      length: 3.0 - i * 0.15
    }));

    const targetX = 15.0;
    const targetY = 20.0;

    console.log('\n=== 15-Joint IK (ABSOLUTE UNIT): Graph vs Compiled (Adam) ===');
    console.log(`Target: (${targetX}, ${targetY})`);
    console.log(`Total reach: ${segments.reduce((sum, s) => sum + s.length, 0).toFixed(1)}\n`);

    function residuals(params: Value[]) {
      const endEffector = forwardKinematics(params, segments);
      return [
        V.sub(endEffector.x, V.C(targetX)),
        V.sub(endEffector.y, V.C(targetY))
      ];
    }

    function evaluateCost(params: Value[]): number {
      const res = residuals(params);
      return res.reduce((sum, r) => sum + r.data * r.data, 0);
    }

    const graphAngles = segments.map((_, i) => V.W(0.1 + i * 0.02));
    const graphOptimizer = new Adam(graphAngles, { learningRate: 0.05 });

    console.log('--- Graph-based Adam ---');
    const graphStart = performance.now();
    let graphIterations = 0;
    let graphFinalCost = 0;

    for (let i = 0; i < 30000; i++) {
      graphFinalCost = evaluateCost(graphAngles);
      graphIterations = i;
      if (graphFinalCost < 1e-8) break;

      const res = residuals(graphAngles);
      const loss = V.mean(res.map(r => V.square(r)));
      graphOptimizer.zeroGrad();
      loss.backward();
      graphOptimizer.step();
    }
    const graphTime = performance.now() - graphStart;
    const graphEndEffector = forwardKinematics(graphAngles, segments);

    console.log(`Iterations: ${graphIterations}`);
    console.log(`Final cost: ${graphFinalCost.toExponential(4)}`);
    console.log(`Time: ${graphTime.toFixed(2)}ms`);
    console.log(`End effector: (${graphEndEffector.x.data.toFixed(3)}, ${graphEndEffector.y.data.toFixed(3)})`);

    const compAngles = segments.map((_, i) => {
      const v = V.W(0.1 + i * 0.02);
      v.paramName = `angle${i}`;
      return v;
    });
    const compOptimizer = new Adam(compAngles, { learningRate: 0.05 });

    const lossGraph = (() => {
      const res = residuals(compAngles);
      return V.mean(res.map(r => V.square(r)));
    })();
    const compiledGradFn = compileGradientFunction(lossGraph, compAngles);

    console.log('\n--- JIT Compiled Adam ---');
    const compStart = performance.now();
    let compIterations = 0;
    let compFinalCost = 0;

    for (let i = 0; i < 30000; i++) {
      compFinalCost = evaluateCost(compAngles);
      compIterations = i;
      if (compFinalCost < 1e-8) break;

      compOptimizer.zeroGrad();
      applyCompiledGradients(compiledGradFn, compAngles);
      compOptimizer.step();
    }
    const compTime = performance.now() - compStart;
    const compEndEffector = forwardKinematics(compAngles, segments);

    console.log(`Iterations: ${compIterations}`);
    console.log(`Final cost: ${compFinalCost.toExponential(4)}`);
    console.log(`Time: ${compTime.toFixed(2)}ms`);
    console.log(`End effector: (${compEndEffector.x.data.toFixed(3)}, ${compEndEffector.y.data.toFixed(3)})`);

    console.log('\n=== ABSOLUTE UNIT Performance Summary ===');
    console.log(`Iterations: ${graphIterations} (graph) vs ${compIterations} (compiled)`);
    console.log(`Time: ${graphTime.toFixed(2)}ms (graph) vs ${compTime.toFixed(2)}ms (compiled)`);
    console.log(`Speedup: ${(graphTime / compTime).toFixed(2)}x`);
    console.log(`Time saved: ${(graphTime - compTime).toFixed(2)}ms`);
    console.log(`Final cost difference: ${Math.abs(graphFinalCost - compFinalCost).toExponential(2)}`);
    console.log(`\nConclusion: JIT compilation provides ${(graphTime / compTime).toFixed(1)}x speedup for ${segments.length}-joint IK!\n`);

    expect(compIterations).toBe(graphIterations);
    expect(Math.abs(graphFinalCost - compFinalCost)).toBeLessThan(1e-10);
    expect(compTime).toBeLessThan(graphTime * 0.8);
  });
});
