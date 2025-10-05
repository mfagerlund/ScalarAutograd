import { describe, it, expect } from 'vitest';
import { V } from "../src/V";
import { Value } from "../src/Value";
import { SGD, Adam, AdamW } from "../src/Optimizers";
import { testLog } from './testUtils';

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

describe.skip('Robot Arm Inverse Kinematics - All Optimizers Comparison', () => {
  it('should solve 3-joint arm IK with all optimizers', { timeout: 10000 }, () => {
    const segments: ArmSegment[] = [
      { length: 3.0 },
      { length: 2.5 },
      { length: 2.0 }
    ];

    const targetX = 5.0;
    const targetY = 4.0;

    testLog('\n=== Robot Arm IK - 3 Joints ===\n');
    testLog('Problem: Find joint angles to reach target position');
    testLog(`Segments: ${segments.map(s => s.length.toFixed(1)).join(', ')}`);
    testLog(`Target: (${targetX}, ${targetY})`);
    testLog(`Total reach: ${segments.reduce((sum, s) => sum + s.length, 0)}\n`);

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

    testLog('--- Nonlinear Least Squares ---');
    testLog(`Iterations: ${nlsResult.iterations}`);
    testLog(`Final cost: ${nlsFinalCost.toExponential(4)}`);
    testLog(`Time: ${nlsTime.toFixed(2)}ms`);
    testLog(`Angles: [${nlsAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    testLog(`End effector: (${nlsEndEffector.x.data.toFixed(3)}, ${nlsEndEffector.y.data.toFixed(3)})`);
    testLog(`Convergence: ${nlsResult.convergenceReason}`);

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

    testLog('\n--- Gradient Descent (SGD, lr=0.1) ---');
    testLog(`Iterations: ${sgdIterations}`);
    testLog(`Final cost: ${sgdFinalCost.toExponential(4)}`);
    testLog(`Time: ${sgdTime.toFixed(2)}ms`);
    testLog(`Angles: [${sgdAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    testLog(`End effector: (${sgdEndEffector.x.data.toFixed(3)}, ${sgdEndEffector.y.data.toFixed(3)})`);

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

    testLog('\n--- Adam (lr=0.1) ---');
    testLog(`Iterations: ${adamIterations}`);
    testLog(`Final cost: ${adamFinalCost.toExponential(4)}`);
    testLog(`Time: ${adamTime.toFixed(2)}ms`);
    testLog(`Angles: [${adamAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    testLog(`End effector: (${adamEndEffector.x.data.toFixed(3)}, ${adamEndEffector.y.data.toFixed(3)})`);

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

    testLog('\n--- AdamW (lr=0.1, wd=0) ---');
    testLog(`Iterations: ${adamwIterations}`);
    testLog(`Final cost: ${adamwFinalCost.toExponential(4)}`);
    testLog(`Time: ${adamwTime.toFixed(2)}ms`);
    testLog(`Angles: [${adamwAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    testLog(`End effector: (${adamwEndEffector.x.data.toFixed(3)}, ${adamwEndEffector.y.data.toFixed(3)})`);

    testLog('\n=== Comparison Summary ===');
    testLog(`Target: (${targetX}, ${targetY})`);
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

    testLog('\n=== Robot Arm IK - 8 JOINTS (THE BEAST) ===\n');
    testLog('Problem: Find joint angles for 8-segment arm to reach target');
    testLog(`Segments: ${segments.map(s => s.length.toFixed(1)).join(', ')}`);
    testLog(`Target: (${targetX}, ${targetY})`);
    testLog(`Total reach: ${segments.reduce((sum, s) => sum + s.length, 0).toFixed(1)}`);
    testLog(`Parameters: ${segments.length} joint angles\n`);

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

    testLog('--- Nonlinear Least Squares ---');
    testLog(`Iterations: ${nlsResult.iterations}`);
    testLog(`Final cost: ${nlsFinalCost.toExponential(4)}`);
    testLog(`Time: ${nlsTime.toFixed(2)}ms`);
    testLog(`Angles: [${nlsAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    testLog(`End effector: (${nlsEndEffector.x.data.toFixed(3)}, ${nlsEndEffector.y.data.toFixed(3)})`);
    testLog(`Distance from target: ${Math.sqrt(nlsFinalCost).toFixed(6)}`);
    testLog(`Convergence: ${nlsResult.convergenceReason}`);

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

    testLog('\n--- Gradient Descent (SGD, lr=0.05) ---');
    testLog(`Iterations: ${sgdIterations}`);
    testLog(`Final cost: ${sgdFinalCost.toExponential(4)}`);
    testLog(`Time: ${sgdTime.toFixed(2)}ms`);
    testLog(`Angles: [${sgdAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    testLog(`End effector: (${sgdEndEffector.x.data.toFixed(3)}, ${sgdEndEffector.y.data.toFixed(3)})`);
    testLog(`Distance from target: ${Math.sqrt(sgdFinalCost).toFixed(6)}`);

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

    testLog('\n--- Adam (lr=0.1) ---');
    testLog(`Iterations: ${adamIterations}`);
    testLog(`Final cost: ${adamFinalCost.toExponential(4)}`);
    testLog(`Time: ${adamTime.toFixed(2)}ms`);
    testLog(`Angles: [${adamAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    testLog(`End effector: (${adamEndEffector.x.data.toFixed(3)}, ${adamEndEffector.y.data.toFixed(3)})`);
    testLog(`Distance from target: ${Math.sqrt(adamFinalCost).toFixed(6)}`);

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

    testLog('\n--- AdamW (lr=0.1, wd=0) ---');
    testLog(`Iterations: ${adamwIterations}`);
    testLog(`Final cost: ${adamwFinalCost.toExponential(4)}`);
    testLog(`Time: ${adamwTime.toFixed(2)}ms`);
    testLog(`Angles: [${adamwAngles.map(a => (a.data * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);
    testLog(`End effector: (${adamwEndEffector.x.data.toFixed(3)}, ${adamwEndEffector.y.data.toFixed(3)})`);
    testLog(`Distance from target: ${Math.sqrt(adamwFinalCost).toFixed(6)}`);

    testLog('\n=== THE BEAST - Comparison Summary ===');
    testLog(`Target: (${targetX}, ${targetY})`);
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
    testLog(`  AdamW: ${adamwFinalCost.toExponential(4)}`);
    testLog('\nDistance from Target:');
    testLog(`  NLS:   ${Math.sqrt(nlsFinalCost).toFixed(6)}`);
    testLog(`  SGD:   ${Math.sqrt(sgdFinalCost).toFixed(6)}`);
    testLog(`  Adam:  ${Math.sqrt(adamFinalCost).toFixed(6)}`);
    testLog(`  AdamW: ${Math.sqrt(adamwFinalCost).toFixed(6)}\n`);

    expect(adamFinalCost).toBeLessThan(1e-6);
    expect(adamwFinalCost).toBeLessThan(1e-6);
  });

  it('should solve 15-joint arm IK - ABSOLUTE UNIT', { timeout: 60000 }, () => {
    const segments: ArmSegment[] = Array.from({ length: 15 }, (_, i) => ({
      length: 3.0 - i * 0.15
    }));

    const targetX = 15.0;
    const targetY = 20.0;

    testLog('\n=== Robot Arm IK - 15 JOINTS (ABSOLUTE UNIT) ===\n');
    testLog('Problem: Find joint angles for 15-segment arm to reach target');
    testLog(`Segment lengths: ${segments.map(s => s.length.toFixed(2)).join(', ')}`);
    testLog(`Target: (${targetX}, ${targetY})`);
    testLog(`Total reach: ${segments.reduce((sum, s) => sum + s.length, 0).toFixed(1)}`);
    testLog(`Parameters: ${segments.length} joint angles\n`);

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

    testLog('--- Nonlinear Least Squares ---');
    testLog(`Iterations: ${nlsResult.iterations}`);
    testLog(`Final cost: ${nlsFinalCost.toExponential(4)}`);
    testLog(`Time: ${nlsTime.toFixed(2)}ms`);
    testLog(`End effector: (${nlsEndEffector.x.data.toFixed(3)}, ${nlsEndEffector.y.data.toFixed(3)})`);
    testLog(`Distance from target: ${Math.sqrt(nlsFinalCost).toFixed(6)}`);
    testLog(`Convergence: ${nlsResult.convergenceReason}`);

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

    testLog('\n--- Adam (lr=0.05) ---');
    testLog(`Iterations: ${adamIterations}`);
    testLog(`Final cost: ${adamFinalCost.toExponential(4)}`);
    testLog(`Time: ${adamTime.toFixed(2)}ms`);
    testLog(`End effector: (${adamEndEffector.x.data.toFixed(3)}, ${adamEndEffector.y.data.toFixed(3)})`);
    testLog(`Distance from target: ${Math.sqrt(adamFinalCost).toFixed(6)}`);

    testLog('\n=== ABSOLUTE UNIT - Comparison Summary ===');
    testLog(`Target: (${targetX}, ${targetY})`);
    testLog('\nIterations:');
    testLog(`  NLS:   ${nlsResult.iterations}`);
    testLog(`  Adam:  ${adamIterations} (${(adamIterations / nlsResult.iterations).toFixed(1)}x)`);
    testLog('\nTime:');
    testLog(`  NLS:   ${nlsTime.toFixed(2)}ms`);
    testLog(`  Adam:  ${adamTime.toFixed(2)}ms (${(adamTime / nlsTime).toFixed(1)}x)`);
    testLog('\nFinal Cost:');
    testLog(`  NLS:   ${nlsFinalCost.toExponential(4)}`);
    testLog(`  Adam:  ${adamFinalCost.toExponential(4)}`);
    testLog('\nDistance from Target:');
    testLog(`  NLS:   ${Math.sqrt(nlsFinalCost).toFixed(6)}`);
    testLog(`  Adam:  ${Math.sqrt(adamFinalCost).toFixed(6)}\n`);

    expect(adamFinalCost).toBeLessThan(1e-6);
  });
});

describe('Robot Arm IK - Failing Cases', () => {
  it('should solve 6-joint arm from recorded failing position', () => {
    const testCase = {
      initialAngles: [
        -0.017067482454226915,
        -1.2291678289082928,
        0.73352762283305,
        0.6822383513796382,
        0.6555926345847977,
        0.08978897938656659
      ],
      targetX: 122,
      targetY: 148.5,
      segments: [45, 42, 39, 36, 33, 30]
    };

    const segments: ArmSegment[] = testCase.segments.map(length => ({ length }));
    const targetX = testCase.targetX;
    const targetY = testCase.targetY;

    testLog('\n=== Recorded Failing Case ===');
    testLog(`Target: (${targetX}, ${targetY})`);
    testLog(`Initial angles: [${testCase.initialAngles.map(a => (a * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);

    function residuals(params: Value[]) {
      const endEffector = forwardKinematics(params, segments);
      return [
        V.sub(endEffector.x, V.C(targetX)),
        V.sub(endEffector.y, V.C(targetY))
      ];
    }

    function residualsWithPosture(params: Value[], refAngles: number[], alpha: number) {
      const endEffector = forwardKinematics(params, segments);
      const positionResiduals = [
        V.sub(endEffector.x, V.C(targetX)),
        V.sub(endEffector.y, V.C(targetY))
      ];
      const sqrtAlpha = Math.sqrt(alpha);
      const postureResiduals = params.map((p, i) =>
        V.mul(V.C(sqrtAlpha), V.sub(p, V.C(refAngles[i])))
      );
      return [...positionResiduals, ...postureResiduals];
    }

    function evaluateCost(params: Value[]): number {
      const res = residuals(params);
      return res.reduce((sum, r) => sum + r.data * r.data, 0);
    }

    const nlsAngles = testCase.initialAngles.map((a, i) => V.W(a, `angle_${i}`));

    testLog('\n--- Without fixes (plain NLS) ---');
    const plainResult = V.nonlinearLeastSquares(nlsAngles, residuals, {
      maxIterations: 200,
      costTolerance: 1e-8,
      adaptiveDamping: true
    });
    const plainEndEffector = forwardKinematics(nlsAngles, segments);
    const plainCost = evaluateCost(nlsAngles);
    testLog(`Success: ${plainResult.success}`);
    testLog(`Iterations: ${plainResult.iterations}`);
    testLog(`Final cost: ${plainCost.toExponential(4)}`);
    testLog(`End effector: (${plainEndEffector.x.data.toFixed(3)}, ${plainEndEffector.y.data.toFixed(3)})`);
    testLog(`Convergence: ${plainResult.convergenceReason}`);

    const nlsAnglesPosture = testCase.initialAngles.map((a, i) => V.W(a, `angle_posture_${i}`));
    const refAngles = [...testCase.initialAngles];

    testLog('\n--- With Tikhonov + QR ---');
    const postureResult = V.nonlinearLeastSquares(
      nlsAnglesPosture,
      (params) => residualsWithPosture(params, refAngles, 0.1),
      {
        maxIterations: 200,
        costTolerance: 1e-8,
        adaptiveDamping: true,
        useQR: true
      }
    );
    const postureEndEffector = forwardKinematics(nlsAnglesPosture, segments);
    const postureCost = evaluateCost(nlsAnglesPosture);
    testLog(`Success: ${postureResult.success}`);
    testLog(`Iterations: ${postureResult.iterations}`);
    testLog(`Final cost: ${postureCost.toExponential(4)}`);
    testLog(`End effector: (${postureEndEffector.x.data.toFixed(3)}, ${postureEndEffector.y.data.toFixed(3)})`);
    testLog(`Convergence: ${postureResult.convergenceReason}`);

    const nlsAnglesHomotopy = testCase.initialAngles.map((a, i) => V.W(a, `angle_homotopy_${i}`));
    const homotopySteps = 10;
    let totalIterations = 0;
    let lastResult: any;
    const homotopyRefAngles = [...testCase.initialAngles];

    testLog('\n--- With Tikhonov + QR + Homotopy ---');
    const startPos = forwardKinematics(nlsAnglesHomotopy, segments);
    const startX = startPos.x.data;
    const startY = startPos.y.data;
    testLog(`Start position: (${startX.toFixed(3)}, ${startY.toFixed(3)})`);
    testLog(`Distance to target: ${Math.sqrt((targetX - startX) ** 2 + (targetY - startY) ** 2).toFixed(3)}`);

    for (let step = 0; step < homotopySteps; step++) {
      const currentPos = forwardKinematics(nlsAnglesHomotopy, segments);
      const currentX = currentPos.x.data;
      const currentY = currentPos.y.data;

      const t = (step + 1) / homotopySteps;
      const intermediateX = currentX + t * (targetX - currentX);
      const intermediateY = currentY + t * (targetY - currentY);

      lastResult = V.nonlinearLeastSquares(
        nlsAnglesHomotopy,
        (params) => {
          const endEffector = forwardKinematics(params, segments);
          return [
            V.sub(endEffector.x, V.C(intermediateX)),
            V.sub(endEffector.y, V.C(intermediateY))
          ];
        },
        {
          maxIterations: 200,
          costTolerance: 1e-8,
          adaptiveDamping: true,
          useQR: true,
          trustRegionRadius: 5.0
        }
      );

      totalIterations += lastResult.iterations;

      if (!lastResult.success) {
        testLog(`  Step ${step + 1}/${homotopySteps}: FAILED after ${lastResult.iterations} iterations`);
        break;
      } else {
        testLog(`  Step ${step + 1}/${homotopySteps}: converged in ${lastResult.iterations} iterations`);
      }

      for (let i = 0; i < nlsAnglesHomotopy.length; i++) {
        homotopyRefAngles[i] = nlsAnglesHomotopy[i].data;
      }
    }

    const homotopyEndEffector = forwardKinematics(nlsAnglesHomotopy, segments);
    const homotopyCost = evaluateCost(nlsAnglesHomotopy);
    testLog(`Success: ${lastResult.success}`);
    testLog(`Total iterations: ${totalIterations}`);
    testLog(`Final cost: ${homotopyCost.toExponential(4)}`);
    testLog(`End effector: (${homotopyEndEffector.x.data.toFixed(3)}, ${homotopyEndEffector.y.data.toFixed(3)})`);
    testLog(`Convergence: ${lastResult.convergenceReason}`);

    testLog('\n=== Summary ===');
    testLog('Plain NLS: ' + (plainResult.success ? 'SUCCESS' : 'FAILED') + (plainResult.success ? ` (${plainResult.iterations} iters, cost: ${plainCost.toExponential(2)})` : ''));
    testLog('Tikhonov + QR: ' + (postureResult.success ? 'SUCCESS' : 'FAILED') + (postureResult.success ? ` (cost: ${postureCost.toExponential(2)})` : ''));
    testLog('Homotopy: ' + (lastResult.success ? 'SUCCESS' : 'FAILED'));
    testLog('\nConclusion: Plain NLS now works perfectly after fixing gradient contamination bug.');
    testLog('Tikhonov regularization can hurt when it fights against the solution.');

    expect(plainResult.success).toBe(true);
    expect(plainCost).toBeLessThan(1e-6);
  });

  it('should solve 6-joint arm from second failing position', () => {
    const testCase = {
      initialAngles: [
        0.1013658962924277,
        -1.1450856232445967,
        -0.1387620958008452,
        1.4758998963690313,
        0.3245162350041625,
        0.7012090310763542
      ],
      targetX: 110,
      targetY: 147.5,
      segments: [45, 42, 39, 36, 33, 30]
    };

    const segments: ArmSegment[] = testCase.segments.map(length => ({ length }));
    const targetX = testCase.targetX;
    const targetY = testCase.targetY;

    testLog('\n=== Second Failing Case ===');
    testLog(`Target: (${targetX}, ${targetY})`);
    testLog(`Initial angles: [${testCase.initialAngles.map(a => (a * 180 / Math.PI).toFixed(1)).join(', ')}] deg`);

    function residuals(params: Value[]) {
      const endEffector = forwardKinematics(params, segments);
      return [
        V.sub(endEffector.x, V.C(targetX)),
        V.sub(endEffector.y, V.C(targetY))
      ];
    }

    function residualsWithPosture(params: Value[], refAngles: number[], alpha: number) {
      const endEffector = forwardKinematics(params, segments);
      const positionResiduals = [
        V.sub(endEffector.x, V.C(targetX)),
        V.sub(endEffector.y, V.C(targetY))
      ];
      const sqrtAlpha = Math.sqrt(alpha);
      const postureResiduals = params.map((p, i) =>
        V.mul(V.C(sqrtAlpha), V.sub(p, V.C(refAngles[i])))
      );
      return [...positionResiduals, ...postureResiduals];
    }

    function evaluateCost(params: Value[]): number {
      const res = residuals(params);
      return res.reduce((sum, r) => sum + r.data * r.data, 0);
    }

    const startPos = forwardKinematics(testCase.initialAngles.map(a => V.C(a)), segments);
    testLog(`Start position: (${startPos.x.data.toFixed(3)}, ${startPos.y.data.toFixed(3)})`);
    testLog(`Distance to target: ${Math.sqrt((targetX - startPos.x.data) ** 2 + (targetY - startPos.y.data) ** 2).toFixed(3)}`);

    const nlsAngles = testCase.initialAngles.map((a, i) => V.W(a, `angle_${i}`));

    testLog('\n--- Without fixes (plain NLS) ---');
    const plainResult = V.nonlinearLeastSquares(nlsAngles, residuals, {
      maxIterations: 200,
      costTolerance: 1e-8,
      adaptiveDamping: true
    });
    const plainEndEffector = forwardKinematics(nlsAngles, segments);
    const plainCost = evaluateCost(nlsAngles);
    testLog(`Success: ${plainResult.success}`);
    testLog(`Iterations: ${plainResult.iterations}`);
    testLog(`Final cost: ${plainCost.toExponential(4)}`);
    testLog(`End effector: (${plainEndEffector.x.data.toFixed(3)}, ${plainEndEffector.y.data.toFixed(3)})`);
    testLog(`Convergence: ${plainResult.convergenceReason}`);

    const nlsAnglesPosture = testCase.initialAngles.map((a, i) => V.W(a, `angle_posture_${i}`));
    const refAngles = [...testCase.initialAngles];

    testLog('\n--- With Tikhonov + QR ---');
    const postureResult = V.nonlinearLeastSquares(
      nlsAnglesPosture,
      (params) => residualsWithPosture(params, refAngles, 0.1),
      {
        maxIterations: 200,
        costTolerance: 1e-8,
        adaptiveDamping: true,
        useQR: true,
        maxInnerIterations: 50
      }
    );
    const postureEndEffector = forwardKinematics(nlsAnglesPosture, segments);
    const postureCost = evaluateCost(nlsAnglesPosture);
    testLog(`Success: ${postureResult.success}`);
    testLog(`Iterations: ${postureResult.iterations}`);
    testLog(`Final cost: ${postureCost.toExponential(4)}`);
    testLog(`End effector: (${postureEndEffector.x.data.toFixed(3)}, ${postureEndEffector.y.data.toFixed(3)})`);
    testLog(`Convergence: ${postureResult.convergenceReason}`);

    const adamAngles = testCase.initialAngles.map((a, i) => V.W(a, `angle_adam_${i}`));
    const adamOptimizer = new Adam(adamAngles, { learningRate: 0.1 });

    testLog('\n--- Adam (lr=0.1) as fallback ---');
    let adamIterations = 0;
    let adamFinalCost = 0;

    for (let i = 0; i < 5000; i++) {
      const res = residuals(adamAngles);
      adamFinalCost = res.reduce((sum, r) => sum + r.data * r.data, 0);
      adamIterations = i;

      if (adamFinalCost < 1e-8) break;

      const loss = V.mean(res.map(r => V.square(r)));
      adamOptimizer.zeroGrad();
      loss.backward();
      adamOptimizer.step();
    }
    const adamEndEffector = forwardKinematics(adamAngles, segments);
    testLog(`Iterations: ${adamIterations}`);
    testLog(`Final cost: ${adamFinalCost.toExponential(4)}`);
    testLog(`End effector: (${adamEndEffector.x.data.toFixed(3)}, ${adamEndEffector.y.data.toFixed(3)})`);

    testLog('\n=== Summary ===');
    testLog('Plain NLS: ' + (plainResult.success ? 'SUCCESS' : 'FAILED') + (plainResult.success ? ` (${plainResult.iterations} iters, cost: ${plainCost.toExponential(2)})` : ''));
    testLog('Tikhonov + QR (NLS): ' + (postureResult.success ? 'SUCCESS' : 'FAILED') + (postureResult.success ? ` (cost: ${postureCost.toExponential(2)})` : ''));
    testLog('Adam fallback: ' + (adamFinalCost < 1e-6 ? 'SUCCESS' : 'FAILED') + ` (${adamIterations} iters, cost: ${adamFinalCost.toExponential(2)})`);
    testLog('\nConclusion: Plain NLS now works perfectly (6 iter). Gradient contamination was the real bug!');

    expect(plainResult.success).toBe(true);
    expect(plainCost).toBeLessThan(1e-6);
  });
});
