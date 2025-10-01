import { describe, it, expect } from 'vitest';
import { V } from './V';
import { Value } from './Value';
import { Vec2 } from './Vec2';
import { Vec3 } from './Vec3';

describe('NonlinearLeastSquares Test Suite - Trivial to Complex', () => {
  describe('Level 1: Trivial', () => {
    it('should optimize single parameter to target value', () => {
      const x = V.W(0);

      function residuals(params: Value[]) {
        return [V.sub(params[0], V.C(42))];
      }

      const result = V.nonlinearLeastSquares([x], residuals);

      expect(result.success).toBe(true);
      expect(x.data).toBeCloseTo(42, 4);
    });

    it('should optimize two independent parameters', () => {
      const x = V.W(0);
      const y = V.W(0);

      function residuals(params: Value[]) {
        return [
          V.sub(params[0], V.C(10)),
          V.sub(params[1], V.C(-5))
        ];
      }

      const result = V.nonlinearLeastSquares([x, y], residuals);

      expect(result.success).toBe(true);
      expect(x.data).toBeCloseTo(10, 4);
      expect(y.data).toBeCloseTo(-5, 4);
    });
  });

  describe('Level 2: Simple 2D geometry', () => {
    it('should move 2D point to target position', () => {
      const point = Vec2.W(0, 0);
      const target = Vec2.C(5, 3);

      function residuals(params: Value[]) {
        const p = new Vec2(params[0], params[1]);
        const diff = p.sub(target);
        return [diff.x, diff.y];
      }

      const result = V.nonlinearLeastSquares(point.trainables, residuals);

      expect(result.success).toBe(true);
      expect(point.x.data).toBeCloseTo(5, 6);
      expect(point.y.data).toBeCloseTo(3, 6);
    });

    it('should optimize distance between two points', () => {
      const p1 = Vec2.W(0, 0);
      const p2 = Vec2.W(5, 5);
      const targetDist = 10;

      function residuals(params: Value[]) {
        const a = new Vec2(params[0], params[1]);
        const b = new Vec2(params[2], params[3]);
        const diff = b.sub(a);
        const dist = diff.magnitude;
        return [V.sub(dist, V.C(targetDist))];
      }

      const result = V.nonlinearLeastSquares([...p1.trainables, ...p2.trainables], residuals);

      expect(result.success).toBe(true);
      const actualDist = Math.sqrt(
        Math.pow(p2.x.data - p1.x.data, 2) +
        Math.pow(p2.y.data - p1.y.data, 2)
      );
      expect(actualDist).toBeCloseTo(targetDist, 3);
    });
  });

  describe('Level 3: Colinear points with distance constraints', () => {
    it('should align 3 points in a line with unit spacing', () => {
      const points = [
        Vec2.W(0, 0),
        Vec2.W(0.5, 0.8),
        Vec2.W(1.2, 1.5)
      ];

      function residuals(params: Value[]) {
        const p0 = new Vec2(params[0], params[1]);
        const p1 = new Vec2(params[2], params[3]);
        const p2 = new Vec2(params[4], params[5]);

        const res: Value[] = [];

        const v01 = p1.sub(p0);
        const v12 = p2.sub(p1);

        const dist01 = v01.magnitude;
        const dist12 = v12.magnitude;
        res.push(V.sub(dist01, V.C(1)));
        res.push(V.sub(dist12, V.C(1)));

        const dir01 = v01.normalized;
        const dir12 = v12.normalized;
        const dot = Vec2.dot(dir01, dir12);
        res.push(V.sub(dot, V.C(1)));

        return res;
      }

      const allParams = points.flatMap(p => p.trainables);
      const result = V.nonlinearLeastSquares(allParams, residuals, { maxIterations: 200 });

      expect(result.success).toBe(true);

      const p0 = points[0];
      const p1 = points[1];
      const p2 = points[2];

      const dist01 = Math.sqrt(
        Math.pow(p1.x.data - p0.x.data, 2) +
        Math.pow(p1.y.data - p0.y.data, 2)
      );
      const dist12 = Math.sqrt(
        Math.pow(p2.x.data - p1.x.data, 2) +
        Math.pow(p2.y.data - p1.y.data, 2)
      );

      expect(dist01).toBeCloseTo(1, 2);
      expect(dist12).toBeCloseTo(1, 2);

      const dx1 = p1.x.data - p0.x.data;
      const dy1 = p1.y.data - p0.y.data;
      const dx2 = p2.x.data - p1.x.data;
      const dy2 = p2.y.data - p1.y.data;
      const angle = Math.abs(Math.atan2(dy1, dx1) - Math.atan2(dy2, dx2));
      expect(angle).toBeLessThan(0.03);
    });

    it('should align 5 points in a line with unit spacing', () => {
      const numPoints = 5;
      const points = Array.from({ length: numPoints }, (_, i) =>
        Vec2.W(i + Math.random() * 0.3, i * 0.8 + Math.random() * 0.3)
      );

      function residuals(params: Value[]) {
        const pts = [];
        for (let i = 0; i < numPoints; i++) {
          pts.push(new Vec2(params[i * 2], params[i * 2 + 1]));
        }

        const res: Value[] = [];

        for (let i = 0; i < numPoints - 1; i++) {
          const v = pts[i + 1].sub(pts[i]);
          const dist = v.magnitude;
          res.push(V.sub(dist, V.C(1)));
        }

        for (let i = 0; i < numPoints - 2; i++) {
          const v1 = pts[i + 1].sub(pts[i]);
          const v2 = pts[i + 2].sub(pts[i + 1]);
          const dir1 = v1.normalized;
          const dir2 = v2.normalized;
          const dot = Vec2.dot(dir1, dir2);
          res.push(V.sub(dot, V.C(1)));
        }

        return res;
      }

      const allParams = points.flatMap(p => p.trainables);
      const result = V.nonlinearLeastSquares(allParams, residuals, { maxIterations: 200 });

      expect(result.success).toBe(true);

      for (let i = 0; i < numPoints - 1; i++) {
        const dist = Math.sqrt(
          Math.pow(points[i + 1].x.data - points[i].x.data, 2) +
          Math.pow(points[i + 1].y.data - points[i].y.data, 2)
        );
        expect(dist).toBeCloseTo(1, 2);
      }
    });
  });

  describe('Level 4: 3D geometry', () => {
    it('should optimize 3D point to target', () => {
      const point = Vec3.W(0, 0, 0);
      const target = Vec3.C(1, 2, 3);

      function residuals(params: Value[]) {
        const p = new Vec3(params[0], params[1], params[2]);
        const diff = p.sub(target);
        return diff.trainables;
      }

      const result = V.nonlinearLeastSquares(point.trainables, residuals);

      expect(result.success).toBe(true);
      expect(point.x.data).toBeCloseTo(1, 6);
      expect(point.y.data).toBeCloseTo(2, 6);
      expect(point.z.data).toBeCloseTo(3, 6);
    });

    it('should create perpendicular 3D vectors', () => {
      const v1 = Vec3.W(1, 0, 0);
      const v2 = Vec3.W(0, 2, 0);

      function residuals(params: Value[]) {
        const a = new Vec3(params[0], params[1], params[2]);
        const b = new Vec3(params[3], params[4], params[5]);

        return [
          V.sub(a.magnitude, V.C(1)),
          V.sub(b.magnitude, V.C(1)),
          Vec3.dot(a, b)
        ];
      }

      const result = V.nonlinearLeastSquares([...v1.trainables, ...v2.trainables], residuals);

      expect(result.success).toBe(true);
      expect(v1.magnitude.data).toBeCloseTo(1, 2);
      expect(v2.magnitude.data).toBeCloseTo(1, 2);

      const dot = v1.x.data * v2.x.data + v1.y.data * v2.y.data + v1.z.data * v2.z.data;
      expect(Math.abs(dot)).toBeLessThan(0.05);
    });
  });

  describe('Level 5: Complex systems', () => {
    it('should fit circle through noisy points', () => {
      const cx = V.W(0.5);
      const cy = V.W(0.5);
      const r = V.W(2);

      const trueCenter = { x: 3, y: 4 };
      const trueRadius = 5;
      const numPoints = 12;

      const points: { x: number; y: number }[] = [];
      for (let i = 0; i < numPoints; i++) {
        const angle = (i / numPoints) * 2 * Math.PI;
        const noise = (Math.random() - 0.5) * 0.1;
        points.push({
          x: trueCenter.x + (trueRadius + noise) * Math.cos(angle),
          y: trueCenter.y + (trueRadius + noise) * Math.sin(angle)
        });
      }

      function residuals(params: Value[]) {
        const [cx, cy, r] = params;
        return points.map(p => {
          const dx = V.sub(V.C(p.x), cx);
          const dy = V.sub(V.C(p.y), cy);
          const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
          return V.sub(dist, r);
        });
      }

      const result = V.nonlinearLeastSquares([cx, cy, r], residuals, {
        maxIterations: 100,
        verbose: false
      });

      expect(result.success).toBe(true);
      expect(cx.data).toBeCloseTo(trueCenter.x, 1);
      expect(cy.data).toBeCloseTo(trueCenter.y, 1);
      expect(r.data).toBeCloseTo(trueRadius, 1);
    });

    it('should solve constrained optimization: points on sphere with distances', () => {
      const points = [
        Vec3.W(1, 0, 0),
        Vec3.W(0, 1, 0),
        Vec3.W(0, 0, 1),
        Vec3.W(-1, 0, 0)
      ];

      const sphereRadius = 2;
      const targetDist = 2.5;

      function residuals(params: Value[]) {
        const pts = [];
        for (let i = 0; i < 4; i++) {
          pts.push(new Vec3(params[i * 3], params[i * 3 + 1], params[i * 3 + 2]));
        }

        const res: Value[] = [];

        for (let i = 0; i < 4; i++) {
          const distFromOrigin = pts[i].magnitude;
          res.push(V.sub(distFromOrigin, V.C(sphereRadius)));
        }

        for (let i = 0; i < 3; i++) {
          const diff = pts[i + 1].sub(pts[i]);
          const dist = diff.magnitude;
          res.push(V.sub(dist, V.C(targetDist)));
        }

        return res;
      }

      const allParams = points.flatMap(p => p.trainables);
      const result = V.nonlinearLeastSquares(allParams, residuals, { maxIterations: 200 });

      expect(result.success).toBe(true);

      for (let i = 0; i < 4; i++) {
        const dist = Math.sqrt(
          points[i].x.data ** 2 +
          points[i].y.data ** 2 +
          points[i].z.data ** 2
        );
        expect(dist).toBeCloseTo(sphereRadius, 2);
      }

      for (let i = 0; i < 3; i++) {
        const dist = Math.sqrt(
          (points[i + 1].x.data - points[i].x.data) ** 2 +
          (points[i + 1].y.data - points[i].y.data) ** 2 +
          (points[i + 1].z.data - points[i].z.data) ** 2
        );
        expect(dist).toBeCloseTo(targetDist, 2);
      }
    });
  });
});
