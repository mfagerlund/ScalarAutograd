import { Value, V, Vec3, Matrix3x3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './utils/EnergyRegistry';

/**
 * Eigenvalue Proxy Energy - CUSTOM IMPLEMENTATION (not from Stein et al. 2018 paper).
 *
 * Experimental approximation of the paper's covariance energy using a closed-form proxy.
 * Approximates the smallest eigenvalue of the covariance matrix without eigensolver.
 *
 * Theory:
 * - Smallest eigenvalue λ_min ≈ Trace(C) - sqrt(||C||²_F)
 * - Line distribution: λ_min ≈ 0 (developable ✓)
 * - Plane distribution: λ_min > 0 (NOT developable ✗)
 *
 * **Limitations**:
 * - Approximation may not be accurate for all normal distributions
 * - Uses centered covariance (different from paper's outer product formulation)
 * - Fixed midpoint split (not paper's approach)
 */
export class EigenProxyEnergy {
  static readonly name = 'Eigenvalue Proxy (Custom Grad)';
  static readonly description = 'Custom: exact λ_min with analytical gradients';
  static readonly supportsCompilation = true;
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
      if (i % 100 === 0) {
      }
      const vertexEnergy = this.computeVertexEnergy(i, mesh);
      residuals.push(vertexEnergy);
    }

    return residuals;
  }

  /**
   * Compute eigenvalue proxy energy for a single vertex.
   * Uses all faces in the star (no splitting).
   */
  static computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);
    if (star.length === 3) return V.C(0); // Skip valence-3 (triple points per paper)

    return this.computeRegionEigenProxy(star, mesh);
  }

  /**
   * Compute smallest eigenvalue for a region's normals.
   * Uses exact eigenvalue computation (no proxy approximation).
   *
   * **Properties**:
   * - Exact smallest eigenvalue of centered covariance matrix
   * - Differentiable (except at repeated eigenvalues)
   * - Ridge regularization ε·I improves stability
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

    // Normalize by n and add strong ridge for numerical stability
    const ridge = V.C(1e-4);
    c00 = V.add(V.div(c00, n), ridge);
    c01 = V.div(c01, n);
    c02 = V.div(c02, n);
    c11 = V.add(V.div(c11, n), ridge);
    c12 = V.div(c12, n);
    c22 = V.add(V.div(c22, n), ridge);

    // Normalize by trace to improve conditioning
    const trace = V.add(V.add(c00, c11), c22);
    const epsilon = V.C(1e-12);
    const safeTrace = V.max(trace, epsilon);

    c00 = V.div(c00, safeTrace);
    c01 = V.div(c01, safeTrace);
    c02 = V.div(c02, safeTrace);
    c11 = V.div(c11, safeTrace);
    c12 = V.div(c12, safeTrace);
    c22 = V.div(c22, safeTrace);

    // Exact smallest eigenvalue with custom analytical gradients
    const lambda = Matrix3x3.smallestEigenvalueCustomGrad(c00, c01, c02, c11, c12, c22);

    // Scale back and clamp to zero
    return V.max(V.mul(lambda, safeTrace), V.C(0));
  }
}
// Register with energy registry
EnergyRegistry.register(EigenProxyEnergy);
