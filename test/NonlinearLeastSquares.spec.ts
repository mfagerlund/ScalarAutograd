import { describe, it, expect } from 'vitest';
import { V } from '../V';
import { Value } from '../Value';

describe('NonlinearLeastSquares', () => {
  it('should solve simple quadratic problem: minimize (x-5)^2 + (y-3)^2', () => {
    const x = V.W(0);
    const y = V.W(0);

    function residuals(params: Value[]) {
      return [
        V.sub(params[0], V.C(5)),
        V.sub(params[1], V.C(3))
      ];
    }

    const result = V.nonlinearLeastSquares([x, y], residuals, { maxIterations: 100 });

    expect(result.success).toBe(true);
    expect(x.data).toBeCloseTo(5, 4);
    expect(y.data).toBeCloseTo(3, 4);
    expect(result.finalCost).toBeLessThan(1e-8);
  });

  it('should solve problem with initial values close to solution', () => {
    const x = V.W(4.5);
    const y = V.W(3.2);

    function residuals(params: Value[]) {
      return [
        V.sub(params[0], V.C(5)),
        V.sub(params[1], V.C(3))
      ];
    }

    const result = V.nonlinearLeastSquares([x, y], residuals);

    expect(result.success).toBe(true);
    expect(x.data).toBeCloseTo(5, 3);
    expect(y.data).toBeCloseTo(3, 3);
  });

  it('should handle sparse problem with many parameters', () => {
    const params = Array.from({ length: 10 }, (_, i) => V.W(i));

    function residuals(p: Value[]) {
      return [
        V.sub(p[0], p[1]),
        V.sub(p[2], p[3]),
        V.sub(p[4], V.C(10))
      ];
    }

    const result = V.nonlinearLeastSquares(params, residuals);

    expect(result.success).toBe(true);
    expect(params[4].data).toBeCloseTo(10, 4);
    expect(Math.abs(params[0].data - params[1].data)).toBeLessThan(1e-4);
    expect(Math.abs(params[2].data - params[3].data)).toBeLessThan(1e-4);
  });

  it('should minimize circle fitting problem', () => {
    const cx = V.W(0);
    const cy = V.W(0);
    const r = V.W(1);

    const points = [
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: -1, y: 0 },
      { x: 0, y: -1 }
    ];

    function residuals(params: Value[]) {
      const [cx, cy, r] = params;
      return points.map(p => {
        const dx = V.sub(V.C(p.x), cx);
        const dy = V.sub(V.C(p.y), cy);
        const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
        return V.sub(dist, r);
      });
    }

    const result = V.nonlinearLeastSquares([cx, cy, r], residuals, { verbose: false });

    expect(result.success).toBe(true);
    expect(cx.data).toBeCloseTo(0, 3);
    expect(cy.data).toBeCloseTo(0, 3);
    expect(r.data).toBeCloseTo(1, 3);
  });

  it('should handle nonlinear problem with multiplication', () => {
    const a = V.W(0.5);
    const b = V.W(0.5);

    function residuals(params: Value[]) {
      const [a, b] = params;
      return [
        V.sub(V.mul(a, V.C(2)), V.C(4)),
        V.sub(V.mul(b, V.C(3)), V.C(9))
      ];
    }

    const result = V.nonlinearLeastSquares([a, b], residuals);

    expect(result.success).toBe(true);
    expect(a.data).toBeCloseTo(2, 6);
    expect(b.data).toBeCloseTo(3, 6);
  });

  it('should report max iterations when not converged', () => {
    const x = V.W(0);

    function residuals(params: Value[]) {
      return [V.exp(params[0])];
    }

    const result = V.nonlinearLeastSquares([x], residuals, { maxIterations: 2 });

    expect(result.success).toBe(false);
    expect(result.convergenceReason).toBe('Max iterations reached');
    expect(result.iterations).toBe(2);
  });

  it('should handle cost below threshold convergence', () => {
    const x = V.W(5.0001);

    function residuals(params: Value[]) {
      return [V.sub(params[0], V.C(5))];
    }

    const result = V.nonlinearLeastSquares([x], residuals, { costTolerance: 1e-4 });

    expect(result.success).toBe(true);
    expect(result.convergenceReason).toContain('Cost');
    expect(result.finalCost).toBeLessThan(1e-4);
  });
});
