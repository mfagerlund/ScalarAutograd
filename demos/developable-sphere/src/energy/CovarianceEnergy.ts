import { Value, V, Vec3, Matrix3x3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './EnergyRegistry';

/**
 * Covariance Energy (E^λ) from "Developability of Triangle Meshes" (Stein et al., SIGGRAPH 2018).
 *
 * This energy measures how close vertex star normals are to being coplanar:
 * - Build angle-weighted covariance matrix: A_i = Σ θ_ijk · N_ijk · N_ijk^T
 * - Energy is the smallest eigenvalue: E^λ = Σ λ_min(A_i)
 *
 * A perfect hinge has E^λ = 0 (all normals lie in a plane).
 *
 * Properties:
 * - Cheaper than E^P: just eigenvalue computation per vertex
 * - Smooth (except at repeated eigenvalues)
 * - Tessellation-invariant (due to angle weighting)
 * - Intrinsic variant avoids "spike" artifacts
 */
export class CovarianceEnergy {
  static readonly name = 'Covariance (Smallest Eigenvalue)';
  static readonly description = 'Paper: E^λ = Σλ_min(A_i), angle-weighted';

  /**
   * Compute total covariance energy for the mesh.
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
   * Compute covariance energy for a single vertex.
   *
   * E^λ_i = λ_min(A_i), where A_i = Σ θ_ijk · N_ijk · N_ijk^T
   *
   * IMPORTANT: The paper's formulation uses the outer product matrix directly,
   * NOT a centered covariance matrix. This works because we're looking for
   * normals that lie in a plane through the origin (on the unit sphere).
   */
  static computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);
    if (star.length === 3) return V.C(0); // Skip valence-3 (triple points per paper)

    // Build angle-weighted outer product matrix
    let c00 = V.C(0),
      c01 = V.C(0),
      c02 = V.C(0);
    let c11 = V.C(0),
      c12 = V.C(0),
      c22 = V.C(0);

    for (const faceIdx of star) {
      const normal = mesh.getFaceNormal(faceIdx).normalized;
      const angle = mesh.getInteriorAngle(faceIdx, vertexIdx);

      // Add weighted outer product: θ · n · n^T
      c00 = V.add(c00, V.mul(angle, V.mul(normal.x, normal.x)));
      c01 = V.add(c01, V.mul(angle, V.mul(normal.x, normal.y)));
      c02 = V.add(c02, V.mul(angle, V.mul(normal.x, normal.z)));
      c11 = V.add(c11, V.mul(angle, V.mul(normal.y, normal.y)));
      c12 = V.add(c12, V.mul(angle, V.mul(normal.y, normal.z)));
      c22 = V.add(c22, V.mul(angle, V.mul(normal.z, normal.z)));
    }

    // Compute smallest eigenvalue
    const lambda = Matrix3x3.smallestEigenvalue(c00, c01, c02, c11, c12, c22);

    // Return absolute value (should be non-negative, but numerical issues can occur)
    return V.abs(lambda);
  }

  /**
   * Compute intrinsic covariance energy (avoids spike artifacts).
   *
   * Instead of using normals directly in 3D, project them to the tangent plane
   * at the vertex via the spherical exponential map. This creates a 2×2 covariance
   * matrix in the tangent space.
   *
   * For a star with normals {N_ijk}, compute:
   * 1. Mean normal direction: N̄ = normalize(Σ N_ijk)
   * 2. For each normal, compute angle φ and tangent direction v
   * 3. Build 2D covariance: Ã = Σ θ_ijk · (φ_ijk · v_ijk)(φ_ijk · v_ijk)^T
   * 4. Smallest eigenvalue of Ã
   */
  static computeVertexEnergyIntrinsic(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);
    if (star.length === 3) return V.C(0);

    // Compute mean normal direction
    let meanNormal = Vec3.zero();
    for (const faceIdx of star) {
      meanNormal = meanNormal.add(mesh.getFaceNormal(faceIdx).normalized);
    }
    meanNormal = meanNormal.normalized;

    // Build 2D covariance matrix in tangent space
    let c00 = V.C(0),
      c01 = V.C(0),
      c11 = V.C(0);

    for (const faceIdx of star) {
      const normal = mesh.getFaceNormal(faceIdx).normalized;
      const angle = mesh.getInteriorAngle(faceIdx, vertexIdx);

      // Compute spherical angle and tangent direction
      const cosAngle = Vec3.dot(normal, meanNormal);
      const phi = V.acos(V.clamp(cosAngle, -0.99999, 0.99999)); // Angle from mean

      // Tangent direction: v = normalize(N - (N·N̄)N̄)
      const projection = meanNormal.mul(cosAngle);
      const tangent = normal.sub(projection);
      const tangentMag = tangent.magnitude;

      // Avoid division by zero for nearly aligned normals
      const epsilon = V.C(1e-12);
      const safeTangentMag = V.max(tangentMag, epsilon);
      const v = tangent.div(safeTangentMag);

      // Scaled tangent vector: Ñ = φ · v (lives in tangent plane)
      const n_tilde_x = V.mul(phi, v.x);
      const n_tilde_y = V.mul(phi, v.y);
      // Note: we only need 2D coords in tangent plane, using x and y components

      // Add weighted outer product: θ · Ñ · Ñ^T (2×2 matrix)
      c00 = V.add(c00, V.mul(angle, V.mul(n_tilde_x, n_tilde_x)));
      c01 = V.add(c01, V.mul(angle, V.mul(n_tilde_x, n_tilde_y)));
      c11 = V.add(c11, V.mul(angle, V.mul(n_tilde_y, n_tilde_y)));
    }

    // For 2×2 symmetric matrix, smallest eigenvalue is:
    // λ_min = (tr - sqrt(tr² - 4·det)) / 2
    const trace = V.add(c00, c11);
    const det = V.sub(V.mul(c00, c11), V.mul(c01, c01));
    const discriminant = V.sub(V.mul(trace, trace), V.mul(V.C(4), det));
    const sqrtDiscriminant = V.sqrt(V.max(discriminant, V.C(0))); // Clamp for numerical stability
    const lambda = V.div(V.sub(trace, sqrtDiscriminant), 2);

    return V.abs(lambda);
  }

  /**
   * Compute max-variant covariance energy (suppresses ruling branches).
   *
   * Instead of summing projections, take the maximum:
   * λ^max_i = min_{|u|=1} max_{ijk} ⟨u, N_ijk⟩²
   *
   * This penalizes the worst-case alignment, favoring straight rulings.
   *
   * Efficient evaluation: check triple sites in spherical Voronoi diagram.
   * For simplicity, we approximate by testing all pairs of normals.
   */
  static computeVertexEnergyMax(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);
    if (star.length === 3) return V.C(0);

    // Get all normals
    const normals: Vec3[] = [];
    for (const faceIdx of star) {
      normals.push(mesh.getFaceNormal(faceIdx).normalized);
    }

    // Approximation: the optimal u is perpendicular to the "widest" pair
    // For each pair, compute perpendicular direction and check max projection
    let minMaxProjection: Value | null = null;

    for (let i = 0; i < normals.length; i++) {
      for (let j = i + 1; j < normals.length; j++) {
        // u perpendicular to both normals[i] and normals[j]
        const u = Vec3.cross(normals[i], normals[j]);
        const uMag = u.magnitude;

        // Skip nearly parallel normals
        if (uMag.data < 1e-6) continue;

        const uNorm = u.div(uMag);

        // Find max projection squared
        let maxProj = V.C(0);
        for (const n of normals) {
          const proj = Vec3.dot(n, uNorm);
          const projSq = V.mul(proj, proj);
          maxProj = V.max(maxProj, projSq);
        }

        if (minMaxProjection === null || maxProj.data < minMaxProjection.data) {
          minMaxProjection = maxProj;
        }
      }
    }

    return minMaxProjection || V.C(0);
  }

  /**
   * Classify vertices as hinges or seams based on energy threshold.
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

  /**
   * Compute quality metrics: developability percentage and number of developable regions.
   * Better metric than raw developability% because small regions can inflate the score.
   */
  static computeQualityMetrics(
    mesh: TriangleMesh,
    hingeThreshold: number = 0.1
  ): { developabilityPct: number; numRegions: number; qualityScore: number } {
    const { hingeVertices, seamVertices } = this.classifyVertices(mesh, hingeThreshold);

    const developabilityPct = (hingeVertices.length / mesh.vertices.length) * 100;

    // Estimate number of regions (same heuristic as CombinatorialEnergy)
    const estimatedAvgSeamLength = Math.max(1, Math.sqrt(mesh.vertices.length) / 2);
    const numRegions = Math.max(1, seamVertices.length / estimatedAvgSeamLength);

    // Quality score: developability per region (penalize fragmentation)
    // Avoid division by zero when developability is 0
    const qualityScore = developabilityPct > 0 ? developabilityPct / Math.sqrt(numRegions) : 0;

    return { developabilityPct, numRegions, qualityScore };
  }
}

// Register with energy registry
EnergyRegistry.register(CovarianceEnergy);
