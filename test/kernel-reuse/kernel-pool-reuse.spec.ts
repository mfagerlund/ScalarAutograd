/**
 * Test kernel pool reuse with realistic scenarios
 */

import { V } from "../../src/V";
import { CompiledResiduals } from "../../src/CompiledResiduals";

describe('Kernel Pool Reuse', () => {
  it('should reuse kernels for identical distance constraints', () => {
    // 10 distance constraints, all with identical structure
    const numConstraints = 10;
    const points: V.Value[] = [];

    for (let i = 0; i < numConstraints * 4; i++) {
      points.push(V.W(Math.random(), `p${i}`));
    }

    const residuals = (params: V.Value[]) => {
      const res: V.Value[] = [];
      for (let i = 0; i < numConstraints; i++) {
        const idx = i * 4;
        const x1 = params[idx];
        const y1 = params[idx + 1];
        const x2 = params[idx + 2];
        const y2 = params[idx + 3];

        const dx = V.sub(x2, x1);
        const dy = V.sub(y2, y1);
        const distSq = V.add(V.mul(dx, dx), V.mul(dy, dy));
        const dist = V.sqrt(distSq);

        res.push(V.sub(dist, V.C(1.0)));
      }
      return res;
    };

    const compiled = CompiledResiduals.compile(points, residuals);

    console.log(`Distance constraints:`);
    console.log(`  Residuals: ${compiled.numResiduals}`);
    console.log(`  Kernels: ${compiled.kernelCount}`);
    console.log(`  Reuse factor: ${compiled.kernelReuseFactor.toFixed(1)}x`);

    // Should only have 1 unique kernel (all distance constraints identical)
    expect(compiled.kernelCount).toBe(1);
    expect(compiled.numResiduals).toBe(10);
    expect(compiled.kernelReuseFactor).toBe(10);
  });

  it('should handle mixed kernel types', () => {
    const a = V.W(1, 'a');
    const b = V.W(2, 'b');
    const c = V.W(3, 'c');
    const params = [a, b, c];

    const residuals = (p: V.Value[]) => [
      V.add(p[0], p[1]),      // Type 1: addition
      V.add(p[1], p[2]),      // Type 1: addition (reuse kernel)
      V.mul(p[0], p[2]),      // Type 2: multiplication
      V.add(p[0], p[2]),      // Type 1: addition (reuse kernel)
      V.mul(p[1], p[2]),      // Type 2: multiplication (reuse kernel)
    ];

    const compiled = CompiledResiduals.compile(params, residuals);

    console.log(`\nMixed operations:`);
    console.log(`  Residuals: ${compiled.numResiduals}`);
    console.log(`  Kernels: ${compiled.kernelCount}`);
    console.log(`  Reuse factor: ${compiled.kernelReuseFactor.toFixed(1)}x`);

    // Should have 2 unique kernels (add and mul)
    expect(compiled.kernelCount).toBe(2);
    expect(compiled.numResiduals).toBe(5);
  });

  it('should dedupe constants across residuals', () => {
    const x = V.W(1, 'x');
    const y = V.W(2, 'y');
    const params = [x, y];

    const residuals = (p: V.Value[]) => [
      V.sub(p[0], V.C(5)),  // Uses constant 5
      V.sub(p[1], V.C(5)),  // Same constant 5 (deduped)
      V.sub(p[0], V.C(10)), // Different constant 10
    ];

    const compiled = CompiledResiduals.compile(params, residuals);

    console.log(`\nConstant deduplication:`);
    console.log(`  Residuals: ${compiled.numResiduals}`);
    console.log(`  Kernels: ${compiled.kernelCount}`);

    // All have same structure (sub), but different constants
    // However, constants are deduped in registry!
    // Kernel sees same structure, so should be 1 kernel
    expect(compiled.kernelCount).toBe(1);
  });

  it('should validate results match non-compiled version', () => {
    const x1 = V.W(0, 'x1');
    const y1 = V.W(0, 'y1');
    const x2 = V.W(3, 'x2');
    const y2 = V.W(4, 'y2');
    const params = [x1, y1, x2, y2];

    const residualFn = (p: V.Value[]) => {
      const dx = V.sub(p[2], p[0]);
      const dy = V.sub(p[3], p[1]);
      const distSq = V.add(V.mul(dx, dx), V.mul(dy, dy));
      const dist = V.sqrt(distSq);
      return [V.sub(dist, V.C(5))];
    };

    const compiled = CompiledResiduals.compile(params, residualFn);

    // Evaluate with compiled
    const result = compiled.evaluate(params);

    // Expected: distance = sqrt(9 + 16) = 5, residual = 0
    expect(result.residuals[0]).toBeCloseTo(0, 10);

    // Expected Jacobian: d(dist)/d(x1) = -dx/dist = -3/5 = -0.6
    expect(result.J[0][0]).toBeCloseTo(-0.6, 10);
    expect(result.J[0][1]).toBeCloseTo(-0.8, 10);
    expect(result.J[0][2]).toBeCloseTo(0.6, 10);
    expect(result.J[0][3]).toBeCloseTo(0.8, 10);
  });

  it('should handle 100 parallel line constraints', () => {
    const numLines = 100;
    const points: V.Value[] = [];

    for (let i = 0; i < numLines * 4; i++) {
      points.push(V.W(Math.random() * 100, `p${i}`));
    }

    const residuals = (params: V.Value[]) => {
      const res: V.Value[] = [];
      for (let i = 0; i < numLines; i++) {
        const idx = i * 4;
        // Line 1: params[idx] to params[idx+1]
        // Line 2: params[idx+2] to params[idx+3]

        // For parallel: cross product of direction vectors = 0
        const dx1 = V.sub(params[idx + 1], params[idx]);
        const dx2 = V.sub(params[idx + 3], params[idx + 2]);

        // Simplified 1D cross product: dx1 - dx2
        res.push(V.sub(dx1, dx2));
      }
      return res;
    };

    const compiled = CompiledResiduals.compile(points, residuals);

    console.log(`\nParallel line constraints:`);
    console.log(`  Residuals: ${compiled.numResiduals}`);
    console.log(`  Kernels: ${compiled.kernelCount}`);
    console.log(`  Reuse factor: ${compiled.kernelReuseFactor.toFixed(1)}x`);

    expect(compiled.kernelCount).toBeLessThan(5); // Should be ~1
    expect(compiled.numResiduals).toBe(100);
  });

  it('should work with curve fitting residuals', () => {
    const a = V.W(1.5, 'a');
    const b = V.W(0.3, 'b');
    const params = [a, b];

    const xData = [0, 1, 2, 3, 4, 5];
    const yData = [1.0, 1.35, 1.82, 2.46, 3.32, 4.48];

    const residuals = (p: V.Value[]) => {
      const res: V.Value[] = [];
      for (let i = 0; i < xData.length; i++) {
        const pred = V.mul(p[0], V.exp(V.mul(p[1], V.C(xData[i]))));
        res.push(V.sub(pred, V.C(yData[i])));
      }
      return res;
    };

    const compiled = CompiledResiduals.compile(params, residuals);

    console.log(`\nCurve fitting:`);
    console.log(`  Residuals: ${compiled.numResiduals}`);
    console.log(`  Kernels: ${compiled.kernelCount}`);
    console.log(`  Reuse factor: ${compiled.kernelReuseFactor.toFixed(1)}x`);

    // All curve fitting residuals have same structure
    expect(compiled.kernelCount).toBe(1);
    expect(compiled.numResiduals).toBe(6);
  });
});
