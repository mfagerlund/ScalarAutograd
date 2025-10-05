import { Vec3 } from './Vec3';
import { V } from './V';
import { Value } from './Value';

/**
 * Geometric utility functions for triangle mesh operations.
 * All methods are differentiable and support automatic gradient computation.
 * @public
 */
export class Geometry {
  /**
   * Compute the normal of a triangle defined by three vertices.
   * Uses right-hand rule: normal points in direction of (b-a) × (c-a).
   * Returns a UNIT normal vector.
   *
   * @param a - First vertex
   * @param b - Second vertex
   * @param c - Third vertex
   * @returns Unit normal vector
   */
  static triangleNormal(a: Vec3, b: Vec3, c: Vec3): Vec3 {
    const edge1 = b.sub(a);
    const edge2 = c.sub(a);
    const cross = Vec3.cross(edge1, edge2);
    return cross.normalized;
  }

  /**
   * Compute the area of a triangle defined by three vertices.
   * Uses the cross product formula: Area = 0.5 * ||(b-a) × (c-a)||
   *
   * @param a - First vertex
   * @param b - Second vertex
   * @param c - Third vertex
   * @returns Triangle area
   */
  static triangleArea(a: Vec3, b: Vec3, c: Vec3): Value {
    const edge1 = b.sub(a);
    const edge2 = c.sub(a);
    const cross = Vec3.cross(edge1, edge2);
    return V.mul(0.5, cross.magnitude);
  }

  /**
   * Compute the interior angle at vertex b in triangle abc.
   * Returns angle in radians, range [0, π].
   *
   * Formula: angle = acos(dot(normalize(a-b), normalize(c-b)))
   *
   * @param a - First neighboring vertex
   * @param b - Vertex where angle is measured
   * @param c - Second neighboring vertex
   * @returns Interior angle in radians
   */
  static interiorAngle(a: Vec3, b: Vec3, c: Vec3): Value {
    const ba = a.sub(b).normalized;
    const bc = c.sub(b).normalized;
    return Vec3.angleBetween(ba, bc);
  }

  /**
   * Compute the angle defect at a vertex.
   * Angle defect = 2π - sum of interior angles around the vertex.
   * For flat regions, angle defect = 0.
   * For positive Gaussian curvature (sphere-like), angle defect > 0.
   * For negative Gaussian curvature (saddle-like), angle defect < 0.
   *
   * @param center - The vertex at which to compute angle defect
   * @param neighbors - Ordered list of neighboring vertices (CCW or CW)
   * @returns Angle defect in radians
   */
  static angleDefect(center: Vec3, neighbors: Vec3[]): Value {
    let angleSum = V.C(0);

    const n = neighbors.length;
    for (let i = 0; i < n; i++) {
      const prev = neighbors[i];
      const next = neighbors[(i + 1) % n];
      const angle = Geometry.interiorAngle(prev, center, next);
      angleSum = V.add(angleSum, angle);
    }

    return V.sub(2 * Math.PI, angleSum);
  }

  /**
   * Compute the centroid (center of mass) of a set of points.
   *
   * @param points - Array of points
   * @returns Centroid
   */
  static centroid(points: Vec3[]): Vec3 {
    if (points.length === 0) {
      return Vec3.zero();
    }

    let sum = Vec3.zero();
    for (const p of points) {
      sum = sum.add(p);
    }
    return sum.div(points.length);
  }

  /**
   * Compute area-weighted normal at a vertex.
   * This is the sum of face normals weighted by face areas,
   * then normalized. Provides a smooth approximation of the
   * surface normal at a vertex.
   *
   * @param center - The vertex
   * @param triangles - Array of triangles as [Vec3, Vec3, Vec3] where center is one of the vertices
   * @returns Area-weighted unit normal
   */
  static vertexNormal(center: Vec3, triangles: [Vec3, Vec3, Vec3][]): Vec3 {
    let weightedSum = Vec3.zero();

    for (const [a, b, c] of triangles) {
      const normal = Geometry.triangleNormal(a, b, c);
      const area = Geometry.triangleArea(a, b, c);
      weightedSum = weightedSum.add(normal.mul(area));
    }

    return weightedSum.normalized;
  }

  /**
   * Project point onto a plane defined by a point and normal.
   *
   * @param point - Point to project
   * @param planePoint - A point on the plane
   * @param planeNormal - Unit normal of the plane
   * @returns Projected point
   */
  static projectToPlane(point: Vec3, planePoint: Vec3, planeNormal: Vec3): Vec3 {
    const toPoint = point.sub(planePoint);
    const distance = Vec3.dot(toPoint, planeNormal);
    const offset = planeNormal.mul(distance);
    return point.sub(offset);
  }

  /**
   * Compute signed distance from point to plane.
   * Positive if point is on the side the normal points to.
   *
   * @param point - Point to measure
   * @param planePoint - A point on the plane
   * @param planeNormal - Unit normal of the plane
   * @returns Signed distance
   */
  static distanceToPlane(point: Vec3, planePoint: Vec3, planeNormal: Vec3): Value {
    const toPoint = point.sub(planePoint);
    return Vec3.dot(toPoint, planeNormal);
  }

  /**
   * Check if a set of normals are coplanar (lie in a common plane).
   * Returns a measure of non-coplanarity (0 = perfectly coplanar).
   * Uses the smallest eigenvalue of the normal covariance matrix.
   *
   * This is a simplified version - the full implementation would
   * compute eigenvalues. For now, returns sum of cross products.
   *
   * @param normals - Array of unit normal vectors
   * @returns Measure of non-coplanarity (0 = coplanar)
   */
  static normalCoplanarity(normals: Vec3[]): Value {
    if (normals.length < 2) return V.C(0);

    // Simple measure: sum of squared cross products between all pairs
    // If all normals are coplanar, all cross products lie along the same axis
    let deviation = V.C(0);

    for (let i = 0; i < normals.length; i++) {
      for (let j = i + 1; j < normals.length; j++) {
        const cross = Vec3.cross(normals[i], normals[j]);
        deviation = V.add(deviation, cross.sqrMagnitude);
      }
    }

    return deviation;
  }

  /**
   * Compute the dihedral angle between two triangles sharing an edge.
   * Returns angle in radians, range [0, π].
   * Angle = 0 means triangles are coplanar.
   * Angle = π means triangles fold completely back on themselves.
   *
   * @param n1 - Normal of first triangle
   * @param n2 - Normal of second triangle
   * @returns Dihedral angle in radians
   */
  static dihedralAngle(n1: Vec3, n2: Vec3): Value {
    const dotProd = Vec3.dot(n1, n2);
    const clampedDot = V.clamp(dotProd, -1, 1);
    return V.acos(clampedDot);
  }

  /**
   * Compute the average of multiple vectors.
   *
   * @param vectors - Array of vectors
   * @returns Average vector (not normalized)
   */
  static average(vectors: Vec3[]): Vec3 {
    if (vectors.length === 0) return Vec3.zero();

    let sum = Vec3.zero();
    for (const v of vectors) {
      sum = sum.add(v);
    }
    return sum.div(vectors.length);
  }

  /**
   * Compute bounding box of a set of points.
   *
   * @param points - Array of points
   * @returns Object with min and max corners
   */
  static boundingBox(points: Vec3[]): { min: Vec3; max: Vec3 } {
    if (points.length === 0) {
      return { min: Vec3.zero(), max: Vec3.zero() };
    }

    let min = points[0].clone();
    let max = points[0].clone();

    for (let i = 1; i < points.length; i++) {
      min = Vec3.min(min, points[i]);
      max = Vec3.max(max, points[i]);
    }

    return { min, max };
  }
}
