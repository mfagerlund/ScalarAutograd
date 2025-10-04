import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './EnergyRegistry';

/**
 * Eigenvalue proxy energy for developable surfaces.
 * Approximates the smallest eigenvalue of the covariance matrix.
 *
 * Theory:
 * - Smallest eigenvalue λ_min ≈ Trace(C) - sqrt(||C||²_F)
 * - Line distribution: λ_min ≈ 0 (developable ✓)
 * - Plane distribution: λ_min > 0 (NOT developable ✗)
 */
export class EigenProxyEnergy {
  static readonly name = 'Eigenvalue Proxy (Trace - Frobenius)';
  /**
   * Compute total eigenvalue proxy energy for the mesh.
   */
  static compute(mesh: TriangleMesh): Value {
    const vertexEnergies: Value[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      vertexEnergies.push(this.computeVertexEnergy(i, mesh));
    }

    return V.sum(vertexEnergies);
  }

  /**
   * Compute per-vertex residuals for compiled optimization.
   */
  static computeResiduals(mesh: TriangleMesh): Value[] {
    const residuals: Value[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      const vertexEnergy = this.computeVertexEnergy(i, mesh);
      residuals.push(vertexEnergy);
    }

    return residuals;
  }

  /**
   * Compute eigenvalue proxy energy for a single vertex.
   * Splits star into two regions to allow hinge formation.
   */
  static computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);

    // Split into two regions
    const mid = Math.floor(star.length / 2);
    const region1 = star.slice(0, mid);
    const region2 = star.slice(mid);

    const energy1 = this.computeRegionEigenProxy(region1, mesh);
    const energy2 = this.computeRegionEigenProxy(region2, mesh);

    return V.add(energy1, energy2);
  }

  /**
   * Compute smallest eigenvalue proxy for a region's normals.
   * Uses: λ_min ≈ Trace(C) - sqrt(||C||²_F)
   */
  private static computeRegionEigenProxy(region: number[], mesh: TriangleMesh): Value {
    if (region.length === 0) return V.C(0);

    // Get all normals in this region (normalized)
    const normals: Vec3[] = [];
    for (const faceIdx of region) {
      normals.push(mesh.getFaceNormal(faceIdx).normalized);
    }

    const n = normals.length;
    if (n < 2) return V.C(0);

    // Compute mean normal
    let meanNormal = Vec3.zero();
    for (const normal of normals) {
      meanNormal = meanNormal.add(normal);
    }
    meanNormal = meanNormal.div(n);

    // Compute covariance matrix elements
    let c00 = V.C(0), c01 = V.C(0), c02 = V.C(0);
    let c11 = V.C(0), c12 = V.C(0), c22 = V.C(0);

    for (const normal of normals) {
      const dx = normal.x.sub(meanNormal.x);
      const dy = normal.y.sub(meanNormal.y);
      const dz = normal.z.sub(meanNormal.z);

      c00 = V.add(c00, V.mul(dx, dx));
      c01 = V.add(c01, V.mul(dx, dy));
      c02 = V.add(c02, V.mul(dx, dz));
      c11 = V.add(c11, V.mul(dy, dy));
      c12 = V.add(c12, V.mul(dy, dz));
      c22 = V.add(c22, V.mul(dz, dz));
    }

    // Normalize by n
    c00 = V.div(c00, n);
    c01 = V.div(c01, n);
    c02 = V.div(c02, n);
    c11 = V.div(c11, n);
    c12 = V.div(c12, n);
    c22 = V.div(c22, n);

    // Compute trace: Tr(C) = c00 + c11 + c22
    const trace = V.add(V.add(c00, c11), c22);

    // Compute Frobenius norm squared: ||C||²_F = sum of all squared elements
    // For symmetric matrix: ||C||²_F = c00² + c11² + c22² + 2(c01² + c02² + c12²)
    const frobSq = V.add(
      V.add(V.mul(c00, c00), V.mul(c11, c11)),
      V.add(
        V.mul(c22, c22),
        V.mul(
          V.C(2),
          V.add(V.add(V.mul(c01, c01), V.mul(c02, c02)), V.mul(c12, c12))
        )
      )
    );

    // Smallest eigenvalue proxy: λ_min ≈ Trace - sqrt(||C||²_F)
    const sqrtFrob = V.sqrt(frobSq);
    const lambdaMinProxy = V.sub(trace, sqrtFrob);

    // Return absolute value (proxy can be slightly negative due to approximation)
    return V.abs(lambdaMinProxy);
  }

  /**
   * Classify vertices as hinges or seams based on energy threshold
   */
  static classifyVertices(
    mesh: TriangleMesh,
    hingeThreshold: number = 0.1
  ): { hingeVertices: number[]; seamVertices: number[] } {
    const hingeVertices: number[] = [];
    const seamVertices: number[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      const energy = this.computeVertexEnergy(i, mesh).data;
      if (energy < hingeThreshold) {
        hingeVertices.push(i);
      } else {
        seamVertices.push(i);
      }
    }

    return { hingeVertices, seamVertices };
  }
}
// Register with energy registry
EnergyRegistry.register(EigenProxyEnergy);
