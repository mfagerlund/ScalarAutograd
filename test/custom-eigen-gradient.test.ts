import { describe, it, expect } from 'vitest';
import { V, Value, Matrix3x3 } from '../src/index';

describe('Custom Eigenvalue Gradients', () => {
  // Helper: numerical gradient check
  function checkGradient(
    fn: (params: Value[]) => Value,
    params: Value[],
    epsilon = 1e-6,
    tolerance = 1e-4
  ): void {
    // Compute analytical gradients
    const output = fn(params);
    output.backward();
    const analyticalGrads = params.map(p => p.grad);

    // Compute numerical gradients
    const numericalGrads = params.map((param, i) => {
      param.grad = 0; // Reset
      const original = param.data;

      param.data = original + epsilon;
      const fPlus = fn(params.map(p => V.C(p.data))).data;

      param.data = original - epsilon;
      const fMinus = fn(params.map(p => V.C(p.data))).data;

      param.data = original; // Restore
      return (fPlus - fMinus) / (2 * epsilon);
    });

    // Compare
    analyticalGrads.forEach((analytical, i) => {
      expect(Math.abs(analytical - numericalGrads[i])).toBeLessThan(tolerance);
    });
  }

  it('should compute correct gradients for diagonal matrix', () => {
    const c00 = V.W(2.0);
    const c01 = V.W(0.0);
    const c02 = V.W(0.0);
    const c11 = V.W(1.0);
    const c12 = V.W(0.0);
    const c22 = V.W(0.5);

    const params = [c00, c01, c02, c11, c12, c22];

    checkGradient(
      (p) => Matrix3x3.smallestEigenvalueCustomGrad(p[0], p[1], p[2], p[3], p[4], p[5]),
      params
    );
  });

  it('should compute correct gradients for symmetric matrix', () => {
    const c00 = V.W(2.0);
    const c01 = V.W(0.5);
    const c02 = V.W(0.3);
    const c11 = V.W(1.5);
    const c12 = V.W(0.2);
    const c22 = V.W(1.0);

    const params = [c00, c01, c02, c11, c12, c22];

    checkGradient(
      (p) => Matrix3x3.smallestEigenvalueCustomGrad(p[0], p[1], p[2], p[3], p[4], p[5]),
      params
    );
  });

  it('should match standard eigenvalue computation for forward pass', () => {
    const c00 = V.C(2.0);
    const c01 = V.C(0.5);
    const c02 = V.C(0.3);
    const c11 = V.C(1.5);
    const c12 = V.C(0.2);
    const c22 = V.C(1.0);

    const standard = Matrix3x3.smallestEigenvalue(c00, c01, c02, c11, c12, c22);
    const custom = Matrix3x3.smallestEigenvalueCustomGrad(c00, c01, c02, c11, c12, c22);

    expect(Math.abs(standard.data - custom.data)).toBeLessThan(1e-10);
  });

  it('should handle near-identity matrix', () => {
    const c00 = V.W(1.0);
    const c01 = V.W(0.01);
    const c02 = V.W(0.02);
    const c11 = V.W(1.1);
    const c12 = V.W(0.03);
    const c22 = V.W(0.9);

    const params = [c00, c01, c02, c11, c12, c22];

    checkGradient(
      (p) => Matrix3x3.smallestEigenvalueCustomGrad(p[0], p[1], p[2], p[3], p[4], p[5]),
      params
    );
  });

  it('should handle matrix with small off-diagonals', () => {
    const c00 = V.W(3.0);
    const c01 = V.W(0.1);
    const c02 = V.W(0.2);
    const c11 = V.W(2.0);
    const c12 = V.W(0.1);
    const c22 = V.W(1.0);

    const params = [c00, c01, c02, c11, c12, c22];

    checkGradient(
      (p) => Matrix3x3.smallestEigenvalueCustomGrad(p[0], p[1], p[2], p[3], p[4], p[5]),
      params
    );
  });
});
