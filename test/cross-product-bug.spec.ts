/**
 * Test cross products and normalization to find the gradient bug.
 */

import { V, Value, Vec3, CompiledResiduals } from '../src';

describe('Cross Product Bug', () => {
  it('should handle single cross product normalized', () => {
    console.log('\n=== SINGLE CROSS PRODUCT ===\n');

    // 6 parameters for 2 vectors
    const params = [
      V.W(1.0), V.W(0.0), V.W(0.0),  // v0
      V.W(0.0), V.W(1.0), V.W(0.0),  // v1
    ];

    let compilationGraph: Value | null = null;

    const compiled = CompiledResiduals.compile(params, (p) => {
      const v0 = new Vec3(p[0], p[1], p[2]);
      const v1 = new Vec3(p[3], p[4], p[5]);

      const cross = Vec3.cross(v0, v1).normalized;

      // Just magnitude squared
      const result = V.add(V.add(V.square(cross.x), V.square(cross.y)), V.square(cross.z));

      compilationGraph = result;
      return [result];
    });

    params.forEach(p => p.grad = 0);
    compilationGraph!.backward();
    const graphGrads = params.map(p => p.grad);

    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    const maxDiff = Math.max(...params.map((_, i) => Math.abs(graphGrads[i] - compiledGrads[i])));
    console.log(`Max gradient diff: ${maxDiff.toExponential(6)}`);

    if (maxDiff < 1e-10) {
      console.log('✅ PASS');
    } else {
      console.log('❌ FAIL');
    }

    expect(maxDiff).toBeLessThan(1e-10);
  });

  it('should handle two cross products multiplied', () => {
    console.log('\n=== TWO CROSS PRODUCTS MULTIPLIED ===\n');

    // 9 parameters for 3 vectors
    const params = [
      V.W(1.0), V.W(0.0), V.W(0.0),  // v0
      V.W(0.0), V.W(1.0), V.W(0.0),  // v1
      V.W(0.0), V.W(0.0), V.W(1.0),  // v2
    ];

    let compilationGraph: Value | null = null;

    const compiled = CompiledResiduals.compile(params, (p) => {
      const v0 = new Vec3(p[0], p[1], p[2]);
      const v1 = new Vec3(p[3], p[4], p[5]);
      const v2 = new Vec3(p[6], p[7], p[8]);

      const cross01 = Vec3.cross(v0, v1).normalized;
      const cross02 = Vec3.cross(v0, v2).normalized;

      // Dot product of the two cross products
      const dot = Vec3.dot(cross01, cross02);
      const result = V.mul(dot, dot);

      compilationGraph = result;
      return [result];
    });

    params.forEach(p => p.grad = 0);
    compilationGraph!.backward();
    const graphGrads = params.map(p => p.grad);

    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    const maxDiff = Math.max(...params.map((_, i) => Math.abs(graphGrads[i] - compiledGrads[i])));
    console.log(`Max gradient diff: ${maxDiff.toExponential(6)}`);

    for (let i = 0; i < params.length; i++) {
      const diff = Math.abs(graphGrads[i] - compiledGrads[i]);
      console.log(`  p[${i}]: graph=${graphGrads[i].toExponential(10)}, compiled=${compiledGrads[i].toExponential(10)}, diff=${diff.toExponential(6)}`);
    }

    if (maxDiff < 1e-10) {
      console.log('✅ PASS');
    } else {
      console.log('❌ FAIL - BUG REPRODUCED!');
    }

    expect(maxDiff).toBeLessThan(1e-10);
  });

  it('should handle cross products with plane normal pattern', () => {
    console.log('\n=== PLANE NORMAL PATTERN ===\n');

    // 9 parameters for 3 vectors
    const params = [
      V.W(1.0), V.W(0.0), V.W(0.1),  // v0
      V.W(0.0), V.W(1.0), V.W(0.05),  // v1
      V.W(0.0), V.W(0.0), V.W(1.0),  // v2
    ];

    let compilationGraph: Value | null = null;

    const compiled = CompiledResiduals.compile(params, (p) => {
      const v0 = new Vec3(p[0], p[1], p[2]);
      const v1 = new Vec3(p[3], p[4], p[5]);
      const v2 = new Vec3(p[6], p[7], p[8]);

      const cross01 = Vec3.cross(v0, v1);
      const cross02 = Vec3.cross(v0, v2);

      // Plane normal: sum of cross products, normalized
      const planeNormal = cross01.add(cross02).normalized;

      // Distance to plane
      const dist = Vec3.dot(v0, planeNormal);
      const result = V.mul(dist, dist);

      compilationGraph = result;
      return [result];
    });

    params.forEach(p => p.grad = 0);
    compilationGraph!.backward();
    const graphGrads = params.map(p => p.grad);

    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    const maxDiff = Math.max(...params.map((_, i) => Math.abs(graphGrads[i] - compiledGrads[i])));
    console.log(`Max gradient diff: ${maxDiff.toExponential(6)}`);

    for (let i = 0; i < params.length; i++) {
      const diff = Math.abs(graphGrads[i] - compiledGrads[i]);
      if (diff > 1e-12) {
        console.log(`  p[${i}]: graph=${graphGrads[i].toExponential(10)}, compiled=${compiledGrads[i].toExponential(10)}, diff=${diff.toExponential(6)}`);
      }
    }

    if (maxDiff < 1e-10) {
      console.log('✅ PASS');
    } else {
      console.log('❌ FAIL - BUG REPRODUCED!');
    }

    expect(maxDiff).toBeLessThan(1e-10);
  });
});
