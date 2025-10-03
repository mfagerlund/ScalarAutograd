import { describe, it, expect } from 'vitest';
import { Geometry } from '../src/Geometry';
import { Vec3 } from '../src/Vec3';
import { V } from '../src/V';

describe('Geometry', () => {
  it('computes triangle normal correctly', () => {
    const a = Vec3.C(0, 0, 0);
    const b = Vec3.C(1, 0, 0);
    const c = Vec3.C(0, 1, 0);

    const normal = Geometry.triangleNormal(a, b, c);

    expect(normal.x.data).toBeCloseTo(0);
    expect(normal.y.data).toBeCloseTo(0);
    expect(normal.z.data).toBeCloseTo(1);
  });

  it('computes triangle area correctly', () => {
    const a = Vec3.C(0, 0, 0);
    const b = Vec3.C(1, 0, 0);
    const c = Vec3.C(0, 1, 0);

    const area = Geometry.triangleArea(a, b, c);

    expect(area.data).toBeCloseTo(0.5);
  });

  it('computes interior angle correctly', () => {
    const a = Vec3.C(1, 0, 0);
    const b = Vec3.C(0, 0, 0);
    const c = Vec3.C(0, 1, 0);

    const angle = Geometry.interiorAngle(a, b, c);

    expect(angle.data).toBeCloseTo(Math.PI / 2); // 90 degrees
  });

  it('computes zero angle defect for flat patch', () => {
    const center = Vec3.C(0, 0, 0);
    const neighbors = [
      Vec3.C(1, 0, 0),
      Vec3.C(0.5, 0.866, 0),
      Vec3.C(-0.5, 0.866, 0),
      Vec3.C(-1, 0, 0),
      Vec3.C(-0.5, -0.866, 0),
      Vec3.C(0.5, -0.866, 0),
    ];

    const defect = Geometry.angleDefect(center, neighbors);

    expect(defect.data).toBeCloseTo(0, 3);
  });

  it('computes dihedral angle correctly', () => {
    const n1 = Vec3.C(0, 0, 1).normalized;
    const n2 = Vec3.C(1, 0, 0).normalized;

    const angle = Geometry.dihedralAngle(n1, n2);

    expect(angle.data).toBeCloseTo(Math.PI / 2);
  });

  it('computes centroid correctly', () => {
    const points = [Vec3.C(1, 0, 0), Vec3.C(0, 1, 0), Vec3.C(0, 0, 1)];

    const centroid = Geometry.centroid(points);

    expect(centroid.x.data).toBeCloseTo(1 / 3);
    expect(centroid.y.data).toBeCloseTo(1 / 3);
    expect(centroid.z.data).toBeCloseTo(1 / 3);
  });

  it('projects point to plane correctly', () => {
    const point = Vec3.C(0, 0, 5);
    const planePoint = Vec3.C(0, 0, 0);
    const planeNormal = Vec3.C(0, 0, 1).normalized;

    const projected = Geometry.projectToPlane(point, planePoint, planeNormal);

    expect(projected.x.data).toBeCloseTo(0);
    expect(projected.y.data).toBeCloseTo(0);
    expect(projected.z.data).toBeCloseTo(0);
  });

  it('computes distance to plane correctly', () => {
    const point = Vec3.C(0, 0, 5);
    const planePoint = Vec3.C(0, 0, 0);
    const planeNormal = Vec3.C(0, 0, 1).normalized;

    const distance = Geometry.distanceToPlane(point, planePoint, planeNormal);

    expect(distance.data).toBeCloseTo(5);
  });

  it('computes bounding box correctly', () => {
    const points = [Vec3.C(-1, 2, 3), Vec3.C(4, -5, 6), Vec3.C(0, 0, 0)];

    const bbox = Geometry.boundingBox(points);

    expect(bbox.min.x.data).toBeCloseTo(-1);
    expect(bbox.min.y.data).toBeCloseTo(-5);
    expect(bbox.min.z.data).toBeCloseTo(0);
    expect(bbox.max.x.data).toBeCloseTo(4);
    expect(bbox.max.y.data).toBeCloseTo(2);
    expect(bbox.max.z.data).toBeCloseTo(6);
  });
});
