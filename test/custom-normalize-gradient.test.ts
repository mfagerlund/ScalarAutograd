import { describe, it, expect } from 'vitest';
import { V, Value, Vec3 } from '../src/index';

describe('Custom Normalize Gradients', () => {
  function checkGradient(
    fn: (params: Value[]) => Vec3,
    params: Value[],
    epsilon = 1e-6,
    tolerance = 1e-4
  ): void {
    const output = fn(params);
    const outputs = [output.x, output.y, output.z];

    for (const out of outputs) {
      out.backward(true);
      const analyticalGrads = params.map(p => p.grad);

      const numericalGrads = params.map((param, i) => {
        param.grad = 0;
        const original = param.data;

        param.data = original + epsilon;
        const fPlus = fn(params.map(p => V.C(p.data)));
        const outPlus = out === output.x ? fPlus.x.data : out === output.y ? fPlus.y.data : fPlus.z.data;

        param.data = original - epsilon;
        const fMinus = fn(params.map(p => V.C(p.data)));
        const outMinus = out === output.x ? fMinus.x.data : out === output.y ? fMinus.y.data : fMinus.z.data;

        param.data = original;
        return (outPlus - outMinus) / (2 * epsilon);
      });

      analyticalGrads.forEach((analytical, i) => {
        expect(Math.abs(analytical - numericalGrads[i])).toBeLessThan(tolerance);
      });
    }
  }

  it('should compute correct gradients for unit-ish vector', () => {
    const x = V.W(1.0);
    const y = V.W(2.0);
    const z = V.W(3.0);
    const params = [x, y, z];

    checkGradient(
      (p) => new Vec3(p[0], p[1], p[2]).normalizedCustomGrad(),
      params
    );
  });

  it('should compute correct gradients for small vector', () => {
    const x = V.W(0.1);
    const y = V.W(0.2);
    const z = V.W(0.05);
    const params = [x, y, z];

    checkGradient(
      (p) => new Vec3(p[0], p[1], p[2]).normalizedCustomGrad(),
      params
    );
  });

  it('should compute correct gradients for axis-aligned vector', () => {
    const x = V.W(0.0);
    const y = V.W(5.0);
    const z = V.W(0.0);
    const params = [x, y, z];

    checkGradient(
      (p) => new Vec3(p[0], p[1], p[2]).normalizedCustomGrad(),
      params
    );
  });

  it('should match standard normalization for forward pass', () => {
    const x = V.C(1.0);
    const y = V.C(2.0);
    const z = V.C(3.0);
    const vec = new Vec3(x, y, z);

    const standard = vec.normalized;
    const custom = vec.normalizedCustomGrad();

    expect(Math.abs(standard.x.data - custom.x.data)).toBeLessThan(1e-10);
    expect(Math.abs(standard.y.data - custom.y.data)).toBeLessThan(1e-10);
    expect(Math.abs(standard.z.data - custom.z.data)).toBeLessThan(1e-10);
  });

  it('should handle large magnitude vector', () => {
    const x = V.W(100.0);
    const y = V.W(200.0);
    const z = V.W(300.0);
    const params = [x, y, z];

    checkGradient(
      (p) => new Vec3(p[0], p[1], p[2]).normalizedCustomGrad(),
      params
    );
  });
});
