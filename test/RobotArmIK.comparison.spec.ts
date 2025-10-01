import { describe, it, expect } from 'vitest';
import { V } from "../src/V";
import { Value } from "../src/Value";
import { SGD, Adam, AdamW } from "../src/Optimizers";

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

describe('Robot Arm Inverse Kinematics - All Optimizers Comparison', () => {
  it('should solve 3-joint arm IK with all optimizers', { timeout: 10000 }, () => {
    const segments: ArmSegment[] = [
      { length: 3.0 },
      { length: 2.5 },
      { length: 2.0 }
    ];

    const targetX = 5.0;
    const targetY = 4.0;

    console.log('\n=== Robot Arm IK - 3 Joints ===\n');
    console.log('Problem: Find joint angles to reach target position');
    console.log(`Segments: ${segments.map(s => s.length.toFixed(1)).join(', ')}`);
    console.log(`Target: (${targetX}, ${targetY})`);
    console.log(`Total reach: ${segments.reduce((sum, s) => sum + s.length, 0)}\n`);

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

    const nlsStart = performance.now();
    const nlsResult = V.nonlinearLeastSquares(nlsAngles, residuals, {
      maxIterations: 100,
      costTolerance: 1e-8
    });
    const nlsTime = performance.now() - nlsStart;
    const nlsEndEffector = forwardKinematics(nlsAngles, segments);
    const nlsFinalCost = evaluateCost(nlsAngles);

    console.log('--- Nonlinear Least Squares ---');
    console.log(`Iterations: ${nlsResult.iterations}`);
    console.log(`Final cost: ${nlsFinalCost.toExponential(4)}`);
    console.log(`Time: ${nlsTime.toFixed(2)}ms`);
    console.log(`Angles: [${nlsAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    console.log(`End effector: (${nlsEndEffector.x.data.toFixed(3)}, ${nlsEndEffector.y.data.toFixed(3)})`);
    console.log(`Convergence: ${nlsResult.convergenceReason}`);

    const sgdAngles = segments.map(() => V.W(0.1));
    const sgdOptimizer = new SGD(sgdAngles, { learningRate: 0.1 });

    const sgdStart = performance.now();
    let sgdIterations = 0;
    let sgdFinalCost = 0;

    for (let i = 0; i < 10000; i++) {
      sgdFinalCost = evaluateCost(sgdAngles);
      sgdIterations = i;

      if (sgdFinalCost < 1e-8) break;

      const res = residuals(sgdAngles);
      const loss = V.mean(res.map(r => V.square(r)));
      sgdOptimizer.zeroGrad();
      loss.backward();
      sgdOptimizer.step();
    }
    const sgdTime = performance.now() - sgdStart;
    const sgdEndEffector = forwardKinematics(sgdAngles, segments);

    console.log('\n--- Gradient Descent (SGD, lr=0.1) ---');
    console.log(`Iterations: ${sgdIterations}`);
    console.log(`Final cost: ${sgdFinalCost.toExponential(4)}`);
    console.log(`Time: ${sgdTime.toFixed(2)}ms`);
    console.log(`Angles: [${sgdAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    console.log(`End effector: (${sgdEndEffector.x.data.toFixed(3)}, ${sgdEndEffector.y.data.toFixed(3)})`);

    const adamAngles = segments.map(() => V.W(0.1));
    const adamOptimizer = new Adam(adamAngles, { learningRate: 0.1 });

    const adamStart = performance.now();
    let adamIterations = 0;
    let adamFinalCost = 0;

    for (let i = 0; i < 10000; i++) {
      adamFinalCost = evaluateCost(adamAngles);
      adamIterations = i;

      if (adamFinalCost < 1e-8) break;

      const res = residuals(adamAngles);
      const loss = V.mean(res.map(r => V.square(r)));
      adamOptimizer.zeroGrad();
      loss.backward();
      adamOptimizer.step();
    }
    const adamTime = performance.now() - adamStart;
    const adamEndEffector = forwardKinematics(adamAngles, segments);

    console.log('\n--- Adam (lr=0.1) ---');
    console.log(`Iterations: ${adamIterations}`);
    console.log(`Final cost: ${adamFinalCost.toExponential(4)}`);
    console.log(`Time: ${adamTime.toFixed(2)}ms`);
    console.log(`Angles: [${adamAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    console.log(`End effector: (${adamEndEffector.x.data.toFixed(3)}, ${adamEndEffector.y.data.toFixed(3)})`);

    const adamwAngles = segments.map(() => V.W(0.1));
    const adamwOptimizer = new AdamW(adamwAngles, { learningRate: 0.1, weightDecay: 0 });

    const adamwStart = performance.now();
    let adamwIterations = 0;
    let adamwFinalCost = 0;

    for (let i = 0; i < 10000; i++) {
      adamwFinalCost = evaluateCost(adamwAngles);
      adamwIterations = i;

      if (adamwFinalCost < 1e-8) break;

      const res = residuals(adamwAngles);
      const loss = V.mean(res.map(r => V.square(r)));
      adamwOptimizer.zeroGrad();
      loss.backward();
      adamwOptimizer.step();
    }
    const adamwTime = performance.now() - adamwStart;
    const adamwEndEffector = forwardKinematics(adamwAngles, segments);

    console.log('\n--- AdamW (lr=0.1, wd=0) ---');
    console.log(`Iterations: ${adamwIterations}`);
    console.log(`Final cost: ${adamwFinalCost.toExponential(4)}`);
    console.log(`Time: ${adamwTime.toFixed(2)}ms`);
    console.log(`Angles: [${adamwAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    console.log(`End effector: (${adamwEndEffector.x.data.toFixed(3)}, ${adamwEndEffector.y.data.toFixed(3)})`);

    console.log('\n=== Comparison Summary ===');
    console.log(`Target: (${targetX}, ${targetY})`);
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

    expect(adamFinalCost).toBeLessThan(1e-6);
    expect(adamwFinalCost).toBeLessThan(1e-6);
  });

  it('should solve 8-joint arm IK - THE BEAST', { timeout: 30000 }, () => {
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

    console.log('\n=== Robot Arm IK - 8 JOINTS (THE BEAST) ===\n');
    console.log('Problem: Find joint angles for 8-segment arm to reach target');
    console.log(`Segments: ${segments.map(s => s.length.toFixed(1)).join(', ')}`);
    console.log(`Target: (${targetX}, ${targetY})`);
    console.log(`Total reach: ${segments.reduce((sum, s) => sum + s.length, 0).toFixed(1)}`);
    console.log(`Parameters: ${segments.length} joint angles\n`);

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

    const nlsAngles = segments.map((_, i) => V.W(0.2 + i * 0.05));

    const nlsStart = performance.now();
    const nlsResult = V.nonlinearLeastSquares(nlsAngles, residuals, {
      maxIterations: 200,
      costTolerance: 1e-8
    });
    const nlsTime = performance.now() - nlsStart;
    const nlsEndEffector = forwardKinematics(nlsAngles, segments);
    const nlsFinalCost = evaluateCost(nlsAngles);

    console.log('--- Nonlinear Least Squares ---');
    console.log(`Iterations: ${nlsResult.iterations}`);
    console.log(`Final cost: ${nlsFinalCost.toExponential(4)}`);
    console.log(`Time: ${nlsTime.toFixed(2)}ms`);
    console.log(`Angles: [${nlsAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    console.log(`End effector: (${nlsEndEffector.x.data.toFixed(3)}, ${nlsEndEffector.y.data.toFixed(3)})`);
    console.log(`Distance from target: ${Math.sqrt(nlsFinalCost).toFixed(6)}`);
    console.log(`Convergence: ${nlsResult.convergenceReason}`);

    const sgdAngles = segments.map((_, i) => V.W(0.2 + i * 0.05));
    const sgdOptimizer = new SGD(sgdAngles, { learningRate: 0.05 });

    const sgdStart = performance.now();
    let sgdIterations = 0;
    let sgdFinalCost = 0;

    for (let i = 0; i < 20000; i++) {
      sgdFinalCost = evaluateCost(sgdAngles);
      sgdIterations = i;

      if (sgdFinalCost < 1e-8) break;

      const res = residuals(sgdAngles);
      const loss = V.mean(res.map(r => V.square(r)));
      sgdOptimizer.zeroGrad();
      loss.backward();
      sgdOptimizer.step();
    }
    const sgdTime = performance.now() - sgdStart;
    const sgdEndEffector = forwardKinematics(sgdAngles, segments);

    console.log('\n--- Gradient Descent (SGD, lr=0.05) ---');
    console.log(`Iterations: ${sgdIterations}`);
    console.log(`Final cost: ${sgdFinalCost.toExponential(4)}`);
    console.log(`Time: ${sgdTime.toFixed(2)}ms`);
    console.log(`Angles: [${sgdAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    console.log(`End effector: (${sgdEndEffector.x.data.toFixed(3)}, ${sgdEndEffector.y.data.toFixed(3)})`);
    console.log(`Distance from target: ${Math.sqrt(sgdFinalCost).toFixed(6)}`);

    const adamAngles = segments.map((_, i) => V.W(0.2 + i * 0.05));
    const adamOptimizer = new Adam(adamAngles, { learningRate: 0.1 });

    const adamStart = performance.now();
    let adamIterations = 0;
    let adamFinalCost = 0;

    for (let i = 0; i < 20000; i++) {
      adamFinalCost = evaluateCost(adamAngles);
      adamIterations = i;

      if (adamFinalCost < 1e-8) break;

      const res = residuals(adamAngles);
      const loss = V.mean(res.map(r => V.square(r)));
      adamOptimizer.zeroGrad();
      loss.backward();
      adamOptimizer.step();
    }
    const adamTime = performance.now() - adamStart;
    const adamEndEffector = forwardKinematics(adamAngles, segments);

    console.log('\n--- Adam (lr=0.1) ---');
    console.log(`Iterations: ${adamIterations}`);
    console.log(`Final cost: ${adamFinalCost.toExponential(4)}`);
    console.log(`Time: ${adamTime.toFixed(2)}ms`);
    console.log(`Angles: [${adamAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    console.log(`End effector: (${adamEndEffector.x.data.toFixed(3)}, ${adamEndEffector.y.data.toFixed(3)})`);
    console.log(`Distance from target: ${Math.sqrt(adamFinalCost).toFixed(6)}`);

    const adamwAngles = segments.map((_, i) => V.W(0.2 + i * 0.05));
    const adamwOptimizer = new AdamW(adamwAngles, { learningRate: 0.1, weightDecay: 0 });

    const adamwStart = performance.now();
    let adamwIterations = 0;
    let adamwFinalCost = 0;

    for (let i = 0; i < 20000; i++) {
      adamwFinalCost = evaluateCost(adamwAngles);
      adamwIterations = i;

      if (adamwFinalCost < 1e-8) break;

      const res = residuals(adamwAngles);
      const loss = V.mean(res.map(r => V.square(r)));
      adamwOptimizer.zeroGrad();
      loss.backward();
      adamwOptimizer.step();
    }
    const adamwTime = performance.now() - adamwStart;
    const adamwEndEffector = forwardKinematics(adamwAngles, segments);

    console.log('\n--- AdamW (lr=0.1, wd=0) ---');
    console.log(`Iterations: ${adamwIterations}`);
    console.log(`Final cost: ${adamwFinalCost.toExponential(4)}`);
    console.log(`Time: ${adamwTime.toFixed(2)}ms`);
    console.log(`Angles: [${adamwAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    console.log(`End effector: (${adamwEndEffector.x.data.toFixed(3)}, ${adamwEndEffector.y.data.toFixed(3)})`);
    console.log(`Distance from target: ${Math.sqrt(adamwFinalCost).toFixed(6)}`);

    console.log('\n=== THE BEAST - Comparison Summary ===');
    console.log(`Target: (${targetX}, ${targetY})`);
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
    console.log(`  AdamW: ${adamwFinalCost.toExponential(4)}`);
    console.log('\nDistance from Target:');
    console.log(`  NLS:   ${Math.sqrt(nlsFinalCost).toFixed(6)}`);
    console.log(`  SGD:   ${Math.sqrt(sgdFinalCost).toFixed(6)}`);
    console.log(`  Adam:  ${Math.sqrt(adamFinalCost).toFixed(6)}`);
    console.log(`  AdamW: ${Math.sqrt(adamwFinalCost).toFixed(6)}\n`);

    expect(adamFinalCost).toBeLessThan(1e-6);
    expect(adamwFinalCost).toBeLessThan(1e-6);
  });

  it('should solve 15-joint arm IK - ABSOLUTE UNIT', { timeout: 60000 }, () => {
    const segments: ArmSegment[] = Array.from({ length: 15 }, (_, i) => ({
      length: 3.0 - i * 0.15
    }));

    const targetX = 15.0;
    const targetY = 20.0;

    console.log('\n=== Robot Arm IK - 15 JOINTS (ABSOLUTE UNIT) ===\n');
    console.log('Problem: Find joint angles for 15-segment arm to reach target');
    console.log(`Segment lengths: ${segments.map(s => s.length.toFixed(2)).join(', ')}`);
    console.log(`Target: (${targetX}, ${targetY})`);
    console.log(`Total reach: ${segments.reduce((sum, s) => sum + s.length, 0).toFixed(1)}`);
    console.log(`Parameters: ${segments.length} joint angles\n`);

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

    const nlsAngles = segments.map((_, i) => V.W(0.1 + i * 0.02));

    const nlsStart = performance.now();
    const nlsResult = V.nonlinearLeastSquares(nlsAngles, residuals, {
      maxIterations: 300,
      costTolerance: 1e-8
    });
    const nlsTime = performance.now() - nlsStart;
    const nlsEndEffector = forwardKinematics(nlsAngles, segments);
    const nlsFinalCost = evaluateCost(nlsAngles);

    console.log('--- Nonlinear Least Squares ---');
    console.log(`Iterations: ${nlsResult.iterations}`);
    console.log(`Final cost: ${nlsFinalCost.toExponential(4)}`);
    console.log(`Time: ${nlsTime.toFixed(2)}ms`);
    console.log(`End effector: (${nlsEndEffector.x.data.toFixed(3)}, ${nlsEndEffector.y.data.toFixed(3)})`);
    console.log(`Distance from target: ${Math.sqrt(nlsFinalCost).toFixed(6)}`);
    console.log(`Convergence: ${nlsResult.convergenceReason}`);

    const adamAngles = segments.map((_, i) => V.W(0.1 + i * 0.02));
    const adamOptimizer = new Adam(adamAngles, { learningRate: 0.05 });

    const adamStart = performance.now();
    let adamIterations = 0;
    let adamFinalCost = 0;

    for (let i = 0; i < 30000; i++) {
      adamFinalCost = evaluateCost(adamAngles);
      adamIterations = i;

      if (adamFinalCost < 1e-8) break;

      const res = residuals(adamAngles);
      const loss = V.mean(res.map(r => V.square(r)));
      adamOptimizer.zeroGrad();
      loss.backward();
      adamOptimizer.step();
    }
    const adamTime = performance.now() - adamStart;
    const adamEndEffector = forwardKinematics(adamAngles, segments);

    console.log('\n--- Adam (lr=0.05) ---');
    console.log(`Iterations: ${adamIterations}`);
    console.log(`Final cost: ${adamFinalCost.toExponential(4)}`);
    console.log(`Time: ${adamTime.toFixed(2)}ms`);
    console.log(`End effector: (${adamEndEffector.x.data.toFixed(3)}, ${adamEndEffector.y.data.toFixed(3)})`);
    console.log(`Distance from target: ${Math.sqrt(adamFinalCost).toFixed(6)}`);

    console.log('\n=== ABSOLUTE UNIT - Comparison Summary ===');
    console.log(`Target: (${targetX}, ${targetY})`);
    console.log('\nIterations:');
    console.log(`  NLS:   ${nlsResult.iterations}`);
    console.log(`  Adam:  ${adamIterations} (${(adamIterations / nlsResult.iterations).toFixed(1)}x)`);
    console.log('\nTime:');
    console.log(`  NLS:   ${nlsTime.toFixed(2)}ms`);
    console.log(`  Adam:  ${adamTime.toFixed(2)}ms (${(adamTime / nlsTime).toFixed(1)}x)`);
    console.log('\nFinal Cost:');
    console.log(`  NLS:   ${nlsFinalCost.toExponential(4)}`);
    console.log(`  Adam:  ${adamFinalCost.toExponential(4)}`);
    console.log('\nDistance from Target:');
    console.log(`  NLS:   ${Math.sqrt(nlsFinalCost).toFixed(6)}`);
    console.log(`  Adam:  ${Math.sqrt(adamFinalCost).toFixed(6)}\n`);

    expect(adamFinalCost).toBeLessThan(1e-6);
  });
});
