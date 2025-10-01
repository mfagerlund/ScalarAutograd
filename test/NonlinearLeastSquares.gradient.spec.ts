import { describe, it, expect } from 'vitest';
import { V } from '../V';
import { Value } from '../Value';

describe('Gradient Norm Convergence', () => {
  it('should converge faster with gradient tolerance on flat cost surface', () => {
    const x = V.W(5);
    const y = V.W(5);

    function residuals(params: Value[]) {
      return [
        V.mul(V.sub(params[0], V.C(0)), V.C(0.01)),
        V.mul(V.sub(params[1], V.C(0)), V.C(0.01))
      ];
    }

    const withGradient = V.nonlinearLeastSquares([x, y], residuals, {
      gradientTolerance: 1e-4,
      costTolerance: 1e-20,
      paramTolerance: 1e-8,
      verbose: false
    });

    x.data = 5;
    y.data = 5;

    const withoutGradient = V.nonlinearLeastSquares([x, y], residuals, {
      gradientTolerance: 1e-20,
      costTolerance: 1e-20,
      paramTolerance: 1e-8,
      maxIterations: 100,
      verbose: false
    });

    expect(withGradient.success).toBe(true);
    expect(withGradient.convergenceReason).toBe('Gradient tolerance reached');
    expect(withGradient.iterations).toBeLessThan(withoutGradient.iterations);
  });

  it('should detect convergence even with non-zero residual', () => {
    const a = V.W(1);
    const b = V.W(1);

    function residuals(params: Value[]) {
      const [a, b] = params;
      return [
        V.sub(V.add(a, b), V.C(4)),
        V.sub(V.sub(a, b), V.C(2))
      ];
    }

    const result = V.nonlinearLeastSquares([a, b], residuals, {
      gradientTolerance: 1e-6,
      paramTolerance: 1e-8,
      verbose: false
    });

    expect(result.success).toBe(true);
    expect(a.data).toBeCloseTo(3, 6);
    expect(b.data).toBeCloseTo(1, 6);
  });

  it('should handle overdetermined system efficiently', () => {
    const slope = V.W(0);
    const intercept = V.W(0);

    const observations = [
      { x: 1, y: 2.1 },
      { x: 2, y: 4.05 },
      { x: 3, y: 5.9 },
      { x: 4, y: 8.1 },
      { x: 5, y: 9.95 }
    ];

    function residuals(params: Value[]) {
      const [slope, intercept] = params;
      return observations.map(obs => {
        const predicted = V.add(V.mul(slope, V.C(obs.x)), intercept);
        return V.sub(predicted, V.C(obs.y));
      });
    }

    const result = V.nonlinearLeastSquares([slope, intercept], residuals, {
      gradientTolerance: 1e-6,
      paramTolerance: 1e-8,
      verbose: false
    });

    expect(result.success).toBe(true);
    expect(slope.data).toBeCloseTo(1.99, 1);
    expect(intercept.data).toBeCloseTo(0.07, 1);
  });

  it('should converge on quadratic bowl with gradient tolerance', () => {
    const x = V.W(10);
    const y = V.W(10);

    function residuals(params: Value[]) {
      const [x, y] = params;
      return [
        V.sub(x, V.C(5)),
        V.sub(y, V.C(3)),
        V.mul(V.square(V.sub(x, V.C(5))), V.C(0.1)),
        V.mul(V.square(V.sub(y, V.C(3))), V.C(0.1))
      ];
    }

    const result = V.nonlinearLeastSquares([x, y], residuals, {
      gradientTolerance: 1e-6,
      paramTolerance: 1e-8,
      costTolerance: 1e-20,
      maxIterations: 50,
      verbose: false
    });

    expect(result.success).toBe(true);
    expect(x.data).toBeCloseTo(5, 3);
    expect(y.data).toBeCloseTo(3, 3);
  });

  it('should handle scaled residuals where gradient detects convergence first', () => {
    const params = [V.W(0.5), V.W(1.5), V.W(2.5)];

    function residuals(p: Value[]) {
      return [
        V.mul(V.sub(p[0], V.C(1)), V.C(10)),
        V.mul(V.sub(p[1], V.C(2)), V.C(10)),
        V.mul(V.sub(p[2], V.C(3)), V.C(10))
      ];
    }

    const result = V.nonlinearLeastSquares(params, residuals, {
      gradientTolerance: 1e-2,
      paramTolerance: 1e-8,
      costTolerance: 1e-20,
      verbose: false
    });

    expect(result.success).toBe(true);
    expect(result.convergenceReason).toBe('Gradient tolerance reached');
    expect(params[0].data).toBeCloseTo(1, 2);
    expect(params[1].data).toBeCloseTo(2, 2);
    expect(params[2].data).toBeCloseTo(3, 2);
  });

  it('should demonstrate gradient tolerance benefit on slowly converging problem', () => {
    const numPoints = 20;
    const observations = Array.from({ length: numPoints }, (_, i) => ({
      x: i * 0.5,
      y: Math.sin(i * 0.5) * 2 + (Math.random() - 0.5) * 0.1
    }));

    function residuals(params: Value[]) {
      const [freq, amp, phase] = params;
      return observations.map(obs => {
        const predicted = V.mul(V.sin(V.add(V.mul(freq, V.C(obs.x)), phase)), amp);
        return V.sub(predicted, V.C(obs.y));
      });
    }

    const p1 = [V.W(0.8), V.W(1.5), V.W(0.2)];
    const withGrad = V.nonlinearLeastSquares(p1, residuals, {
      gradientTolerance: 1e-3,
      paramTolerance: 1e-10,
      costTolerance: 1e-10,
      maxIterations: 50,
      verbose: false
    });

    const p2 = [V.W(0.8), V.W(1.5), V.W(0.2)];
    const withoutGrad = V.nonlinearLeastSquares(p2, residuals, {
      gradientTolerance: 1e-20,
      paramTolerance: 1e-10,
      costTolerance: 1e-10,
      maxIterations: 50,
      verbose: false
    });

    expect(withGrad.success).toBe(true);
    expect(withGrad.iterations).toBeLessThan(withoutGrad.iterations);
  });
});
