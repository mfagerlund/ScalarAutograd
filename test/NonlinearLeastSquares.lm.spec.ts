import { describe, it, expect } from 'vitest';
import { V } from "../src/V";
import { Value } from "../src/Value";
import { Vec2 } from "../src/Vec2";
import { Vec3 } from "../src/Vec3";
import { testLog } from './testUtils';

describe('Levenberg-Marquardt vs Pure Gauss-Newton', () => {
  describe('Easy problems (both should work)', () => {
    it('should solve simple quadratic with both GN and LM', () => {
      const testBoth = (adaptiveDamping: boolean) => {
        const x = V.W(0);
        const y = V.W(0);

        function residuals(params: Value[]) {
          return [
            V.sub(params[0], V.C(5)),
            V.sub(params[1], V.C(3))
          ];
        }

        const result = V.nonlinearLeastSquares([x, y], residuals, { adaptiveDamping });
        return { result, x: x.data, y: y.data };
      };

      const gn = testBoth(false);
      const lm = testBoth(true);

      expect(gn.result.success).toBe(true);
      expect(lm.result.success).toBe(true);

      expect(gn.x).toBeCloseTo(5, 5);
      expect(gn.y).toBeCloseTo(3, 5);
      expect(lm.x).toBeCloseTo(5, 5);
      expect(lm.y).toBeCloseTo(3, 5);

      testLog('\nSimple quadratic:');
      testLog(`  GN: ${gn.result.iterations} iterations, cost=${gn.result.finalCost.toExponential(2)}`);
      testLog(`  LM: ${lm.result.iterations} iterations, cost=${lm.result.finalCost.toExponential(2)}`);
    });

    it('should solve circle fitting with both methods', () => {
      const points = Array.from({ length: 20 }, (_, i) => {
        const angle = (i / 20) * 2 * Math.PI;
        return {
          x: 10 + 5 * Math.cos(angle) + (Math.random() - 0.5) * 0.1,
          y: -5 + 5 * Math.sin(angle) + (Math.random() - 0.5) * 0.1
        };
      });

      function residuals(params: Value[]) {
        const [cx, cy, r] = params;
        return points.map(p => {
          const dx = V.sub(V.C(p.x), cx);
          const dy = V.sub(V.C(p.y), cy);
          const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
          return V.sub(dist, r);
        });
      }

      const gnParams = [V.W(0), V.W(0), V.W(1)];
      const lmParams = [V.W(0), V.W(0), V.W(1)];

      const gnResult = V.nonlinearLeastSquares(gnParams, residuals, { adaptiveDamping: false });
      const lmResult = V.nonlinearLeastSquares(lmParams, residuals, { adaptiveDamping: true });

      expect(gnResult.success).toBe(true);
      expect(lmResult.success).toBe(true);

      expect(gnParams[0].data).toBeCloseTo(10, 1);
      expect(lmParams[0].data).toBeCloseTo(10, 1);

      testLog('\nCircle fitting (good start):');
      testLog(`  GN: ${gnResult.iterations} iterations, cost=${gnResult.finalCost.toExponential(2)}`);
      testLog(`  LM: ${lmResult.iterations} iterations, cost=${lmResult.finalCost.toExponential(2)}`);
    });
  });

  describe('Hard problems (LM should be more robust)', () => {
    it('should handle bad starting point for circle fitting', () => {
      const points = Array.from({ length: 30 }, (_, i) => {
        const angle = (i / 30) * 2 * Math.PI;
        return {
          x: 10 + 5 * Math.cos(angle),
          y: -5 + 5 * Math.sin(angle)
        };
      });

      function residuals(params: Value[]) {
        const [cx, cy, r] = params;
        return points.map(p => {
          const dx = V.sub(V.C(p.x), cx);
          const dy = V.sub(V.C(p.y), cy);
          const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
          return V.sub(dist, r);
        });
      }

      const gnParams = [V.W(100), V.W(100), V.W(0.1)];
      const lmParams = [V.W(100), V.W(100), V.W(0.1)];

      const gnResult = V.nonlinearLeastSquares(gnParams, residuals, {
        adaptiveDamping: false,
        maxIterations: 50
      });
      const lmResult = V.nonlinearLeastSquares(lmParams, residuals, {
        adaptiveDamping: true,
        maxIterations: 50
      });

      testLog('\nCircle fitting (bad start: cx=100, cy=100, r=0.1):');
      testLog(`  GN: success=${gnResult.success}, iterations=${gnResult.iterations}, cost=${gnResult.finalCost.toExponential(2)}`);
      testLog(`  LM: success=${lmResult.success}, iterations=${lmResult.iterations}, cost=${lmResult.finalCost.toExponential(2)}`);

      expect(lmResult.success).toBe(true);
      expect(lmParams[0].data).toBeCloseTo(10, 0);
      expect(lmParams[1].data).toBeCloseTo(-5, 0);
      expect(lmParams[2].data).toBeCloseTo(5, 0);
    });

    it('should handle Rosenbrock-like problem from poor start', () => {
      const x = V.W(-1.5);
      const y = V.W(2.5);

      function residuals(params: Value[]) {
        const [x, y] = params;
        const r1 = V.sub(V.C(1), x);
        const r2 = V.mul(V.C(10), V.sub(y, V.square(x)));
        return [r1, r2];
      };

      const lmResult = V.nonlinearLeastSquares([x, y], residuals, {
        adaptiveDamping: true,
        verbose: false,
        maxIterations: 100
      });

      testLog('\nRosenbrock valley (bad start: x=-1.5, y=2.5):');
      testLog(`  LM: success=${lmResult.success}, iterations=${lmResult.iterations}`);
      testLog(`  Solution: x=${x.data.toFixed(4)}, y=${y.data.toFixed(4)}`);
      testLog(`  Expected: x=1, y=1`);

      expect(lmResult.success).toBe(true);
      expect(x.data).toBeCloseTo(1, 1);
      expect(y.data).toBeCloseTo(1, 1);
    });

    it('should handle 3D distance network from random initialization', () => {
      const numPoints = 10;
      const targetDist = 2.0;

      const points = Array.from({ length: numPoints }, () =>
        Vec3.W(
          (Math.random() - 0.5) * 20,
          (Math.random() - 0.5) * 20,
          (Math.random() - 0.5) * 20
        )
      );

      function residuals(params: Value[]) {
        const pts: Vec3[] = [];
        for (let i = 0; i < numPoints; i++) {
          pts.push(new Vec3(params[i * 3], params[i * 3 + 1], params[i * 3 + 2]));
        }

        const res: Value[] = [];
        for (let i = 0; i < numPoints; i++) {
          const next = (i + 1) % numPoints;
          const diff = pts[next].sub(pts[i]);
          const dist = diff.magnitude;
          res.push(V.sub(dist, V.C(targetDist)));
        }

        return res;
      }

      const allParams = points.flatMap(p => p.trainables);
      const result = V.nonlinearLeastSquares(allParams, residuals, {
        adaptiveDamping: true,
        maxIterations: 200
      });

      testLog('\n3D distance network (random start):');
      testLog(`  LM: success=${result.success}, iterations=${result.iterations}, cost=${result.finalCost.toExponential(2)}`);

      expect(result.success).toBe(true);
      expect(result.finalCost).toBeLessThan(1e-4);
    });
  });

  describe('Verify LM behavior', () => {
    it('should show lambda decreasing as optimization progresses', () => {
      const x = V.W(0);
      const y = V.W(0);

      function residuals(params: Value[]) {
        return [
          V.sub(params[0], V.C(5)),
          V.sub(params[1], V.C(3))
        ];
      }

      testLog('\nLevenberg-Marquardt lambda adaptation:');
      const result = V.nonlinearLeastSquares([x, y], residuals, {
        adaptiveDamping: true,
        verbose: true,
        initialDamping: 1.0
      });

      expect(result.success).toBe(true);
    });

    it('should default to LM when no option specified', () => {
      const x = V.W(0);
      const y = V.W(0);

      function residuals(params: Value[]) {
        return [
          V.sub(params[0], V.C(5)),
          V.sub(params[1], V.C(3))
        ];
      }

      const result = V.nonlinearLeastSquares([x, y], residuals);

      expect(result.success).toBe(true);
      expect(x.data).toBeCloseTo(5, 4);
      expect(y.data).toBeCloseTo(3, 4);
    });

    it('should allow explicit GN by disabling adaptive damping', () => {
      const x = V.W(0);
      const y = V.W(0);

      function residuals(params: Value[]) {
        return [
          V.sub(params[0], V.C(5)),
          V.sub(params[1], V.C(3))
        ];
      }

      const result = V.nonlinearLeastSquares([x, y], residuals, {
        adaptiveDamping: false
      });

      expect(result.success).toBe(true);
      expect(x.data).toBeCloseTo(5, 4);
      expect(y.data).toBeCloseTo(3, 4);
    });
  });

  describe('Performance comparison', () => {
    it('should compare GN vs LM on moderately difficult problem', () => {
      const numPoints = 20;

      const testMethod = (adaptiveDamping: boolean) => {
        const points = Array.from({ length: numPoints }, (_, i) => {
          const angle = (i / numPoints) * Math.PI;
          return Vec2.W(
            Math.cos(angle) * 5 + (Math.random() - 0.5) * 8,
            Math.sin(angle) * 5 + (Math.random() - 0.5) * 8
          );
        });

        function residuals(params: Value[]) {
          const pts: Vec2[] = [];
          for (let i = 0; i < numPoints; i++) {
            pts.push(new Vec2(params[i * 2], params[i * 2 + 1]));
          }

          const res: Value[] = [];
          for (let i = 0; i < numPoints - 1; i++) {
            const diff = pts[i + 1].sub(pts[i]);
            const dist = diff.magnitude;
            res.push(V.sub(dist, V.C(1)));
          }

          return res;
        }

        const allParams = points.flatMap(p => p.trainables);
        const start = performance.now();
        const result = V.nonlinearLeastSquares(allParams, residuals, {
          adaptiveDamping,
          maxIterations: 100
        });
        const time = performance.now() - start;

        return { result, time };
      };

      const gn = testMethod(false);
      const lm = testMethod(true);

      testLog('\n20-point chain alignment (moderate difficulty):');
      testLog(`  GN: success=${gn.result.success}, ${gn.result.iterations} iter, ${gn.time.toFixed(1)}ms, cost=${gn.result.finalCost.toExponential(2)}`);
      testLog(`  LM: success=${lm.result.success}, ${lm.result.iterations} iter, ${lm.time.toFixed(1)}ms, cost=${lm.result.finalCost.toExponential(2)}`);

      expect(gn.result.success || lm.result.success).toBe(true);
    });
  });
});
