/**
 * Minimal reproduction of DifferentiablePlaneAlignment gradient bug.
 * Tries to isolate what makes it different from passing tests.
 */

import { V } from "../src/V";
import { Value } from "../src/Value";
import { Vec3 } from "../src/Vec3";
import { CompiledFunctions } from "../src/CompiledFunctions";
import { testLog } from './testUtils';

describe('Plane Alignment Minimal Reproduction', () => {
  it('simple Vec3 operations', () => {
    const params = [
      V.W(1.0, 'ax'), V.W(2.0, 'ay'), V.W(3.0, 'az'),
      V.W(4.0, 'bx'), V.W(5.0, 'by'), V.W(6.0, 'bz'),
    ];
    params.forEach((p, i) => p.paramName = `p${i}`);

    const buildExpr = (p: Value[]) => {
      const a = new Vec3(p[0], p[1], p[2]);
      const b = new Vec3(p[3], p[4], p[5]);
      const dot = Vec3.dot(a, b);
      return [dot];
    };

    // Graph
    const expr = buildExpr(params);
    params.forEach(p => p.grad = 0);
    expr[0].backward();
    const graphGrads = params.map(p => p.grad);

    // Compiled
    const compiled = CompiledFunctions.compile(params, buildExpr);
    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    testLog('\nVec3 dot product:');
    testLog(`Graph:    [${graphGrads.map(g => g.toExponential(3)).join(', ')}]`);
    testLog(`Compiled: [${compiledGrads.map(g => g.toExponential(3)).join(', ')}]`);

    for (let i = 0; i < params.length; i++) {
      expect(compiledGrads[i]).toBeCloseTo(graphGrads[i], 10);
    }
  });

  it('Vec3 cross product', () => {
    const params = [
      V.W(1.0, 'ax'), V.W(0.0, 'ay'), V.W(0.0, 'az'),
      V.W(0.0, 'bx'), V.W(1.0, 'by'), V.W(0.0, 'bz'),
    ];
    params.forEach((p, i) => p.paramName = `p${i}`);

    const buildExpr = (p: Value[]) => {
      const a = new Vec3(p[0], p[1], p[2]);
      const b = new Vec3(p[3], p[4], p[5]);
      const cross = Vec3.cross(a, b);
      // Return magnitude squared
      return [Vec3.dot(cross, cross)];
    };

    // Graph
    const expr = buildExpr(params);
    params.forEach(p => p.grad = 0);
    expr[0].backward();
    const graphGrads = params.map(p => p.grad);

    // Compiled
    const compiled = CompiledFunctions.compile(params, buildExpr);
    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    testLog('\nVec3 cross product magnitude squared:');
    testLog(`Graph:    [${graphGrads.map(g => g.toExponential(3)).join(', ')}]`);
    testLog(`Compiled: [${compiledGrads.map(g => g.toExponential(3)).join(', ')}]`);

    for (let i = 0; i < params.length; i++) {
      expect(compiledGrads[i]).toBeCloseTo(graphGrads[i], 10);
    }
  });

  it('multiple Vec3 cross products (like plane normal)', () => {
    // Simulate what DifferentiablePlaneAlignment does:
    // Average weighted cross products
    const params = [
      V.W(1.0, 'n1x'), V.W(0.0, 'n1y'), V.W(0.0, 'n1z'),
      V.W(0.0, 'n2x'), V.W(1.0, 'n2y'), V.W(0.0, 'n2z'),
      V.W(0.0, 'n3x'), V.W(0.0, 'n3y'), V.W(1.0, 'n3z'),
    ];
    params.forEach((p, i) => p.paramName = `p${i}`);

    const buildExpr = (p: Value[]) => {
      const n1 = new Vec3(p[0], p[1], p[2]);
      const n2 = new Vec3(p[3], p[4], p[5]);
      const n3 = new Vec3(p[6], p[7], p[8]);

      // Weighted cross products (like in DifferentiablePlaneAlignment)
      const cross12 = Vec3.cross(n1, n2);
      const dot12 = Vec3.dot(n1, n2);
      const sep12 = V.sub(V.C(1), dot12);

      const cross23 = Vec3.cross(n2, n3);
      const dot23 = Vec3.dot(n2, n3);
      const sep23 = V.sub(V.C(1), dot23);

      // Weighted sum
      const planeNormal = cross12.mul(sep12).add(cross23.mul(sep23));

      // Normalize
      const mag = planeNormal.magnitude;
      const safeMag = V.max(mag, V.C(1e-12));
      const normalized = planeNormal.div(safeMag);

      // Return magnitude as single residual
      return [normalized.magnitude];
    };

    // Graph
    const expr = buildExpr(params);
    params.forEach(p => p.grad = 0);
    expr[0].backward();
    const graphGrads = params.map(p => p.grad);

    // Compiled
    const compiled = CompiledFunctions.compile(params, buildExpr);
    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    testLog('\nWeighted cross products + normalization:');
    testLog(`Graph:    [${graphGrads.map(g => g.toExponential(3)).join(', ')}]`);
    testLog(`Compiled: [${compiledGrads.map(g => g.toExponential(3)).join(', ')}]`);

    for (let i = 0; i < params.length; i++) {
      const diff = Math.abs(compiledGrads[i] - graphGrads[i]);
      if (diff > 1e-10) {
        testLog(`Mismatch at param ${i}: diff = ${diff.toExponential(6)}`);
      }
      expect(compiledGrads[i]).toBeCloseTo(graphGrads[i], 10);
    }
  });

  it('full vertex energy simulation', () => {
    // Simulate one vertex energy computation from DifferentiablePlaneAlignment
    // Simplified: 3 normals, compute weighted cross products, normalize, compute energy

    const params = [
      V.W(1.0, 'n1x'), V.W(0.0, 'n1y'), V.W(0.0, 'n1z'),
      V.W(0.0, 'n2x'), V.W(1.0, 'n2y'), V.W(0.0, 'n2z'),
      V.W(0.0, 'n3x'), V.W(0.0, 'n3y'), V.W(1.0, 'n3z'),
    ];
    params.forEach((p, i) => p.paramName = `p${i}`);

    const buildExpr = (p: Value[]) => {
      const normals = [
        new Vec3(p[0], p[1], p[2]),
        new Vec3(p[3], p[4], p[5]),
        new Vec3(p[6], p[7], p[8]),
      ];

      // Differentiable plane selection (from DifferentiablePlaneAlignment)
      let planeNormal = Vec3.zero();
      for (let i = 0; i < normals.length; i++) {
        for (let j = i + 1; j < normals.length; j++) {
          const cross = Vec3.cross(normals[i], normals[j]);
          const dotProduct = Vec3.dot(normals[i], normals[j]);
          const separation = V.sub(V.C(1), dotProduct);
          planeNormal = planeNormal.add(cross.mul(separation));
        }
      }

      const planeNormalMag = planeNormal.magnitude;
      const epsilon = V.C(1e-12);
      const safeMag = V.max(planeNormalMag, epsilon);
      const planeNormalNormalized = planeNormal.div(safeMag);

      // Compute energy (distance from normals to plane)
      let energy = V.C(0);
      for (let i = 0; i < normals.length; i++) {
        const dist = V.abs(Vec3.dot(normals[i], planeNormalNormalized));
        energy = V.add(energy, V.mul(dist, dist));
      }

      // Normalize by count
      return [V.div(energy, normals.length)];
    };

    // Graph
    const expr = buildExpr(params);
    params.forEach(p => p.grad = 0);
    expr[0].backward();
    const graphGrads = params.map(p => p.grad);
    const graphValue = expr[0].data;

    // Compiled
    const compiled = CompiledFunctions.compile(params, buildExpr);
    const { value: compiledValue, gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    testLog('\nFull vertex energy simulation:');
    testLog(`Graph value:    ${graphValue.toExponential(10)}`);
    testLog(`Compiled value: ${compiledValue.toExponential(10)}`);
    testLog(`Graph:    [${graphGrads.map(g => g.toExponential(6)).join(', ')}]`);
    testLog(`Compiled: [${compiledGrads.map(g => g.toExponential(6)).join(', ')}]`);

    // Check value
    expect(compiledValue).toBeCloseTo(graphValue, 10);

    // Check gradients
    for (let i = 0; i < params.length; i++) {
      const diff = Math.abs(compiledGrads[i] - graphGrads[i]);
      if (diff > 1e-10) {
        testLog(`❌ Mismatch at param ${i}:`);
        testLog(`   Graph:    ${graphGrads[i].toExponential(15)}`);
        testLog(`   Compiled: ${compiledGrads[i].toExponential(15)}`);
        testLog(`   Diff:     ${diff.toExponential(6)}`);
      }
      expect(compiledGrads[i]).toBeCloseTo(graphGrads[i], 10);
    }
  });
});
