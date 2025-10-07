import { Value, V, Vec3, Matrix3x3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './utils/EnergyRegistry';

/**
 * Covariance Energy (E^λ) from "Developability of Triangle Meshes" (Stein et al., SIGGRAPH 2018).
 *
 * GROUND TRUTH REFERENCE:
 * C:\Dev\ScalarAutograd\demos\developable-sphere\src\energy\hinge_energy.cpp
 * See hinge_energy_and_grad() function (lines 47-234)
 *
 * ALGORITHM (matching C++ reference):
 * For each vertex i:
 *   1. Compute area-weighted vertex normal: Nv = normalize(Σ area_f * Nf)
 *      (hinge_energy.cpp:90-96)
 *   2. For each adjacent face f:
 *      a. Get face normal Nf and interior angle θ
 *      b. Compute tangent projection: Nfw = normalize((Nv × Nf) × Nv) * acos(Nv·Nf)
 *         (hinge_energy.cpp:135)
 *      c. Add to covariance matrix: mat += θ * Nfw * Nfw^T
 *         (hinge_energy.cpp:136)
 *   3. Eigendecompose mat and return smallest eigenvalue λ_min
 *      (hinge_energy.cpp:142-152)
 *   4. Energy = Σ λ_min over all vertices
 *
 * A perfect hinge has E^λ = 0 (all tangent projections lie along one direction).
 *
 * VERIFIED BEHAVIOR:
 * - Gradients are correct (verified against finite differences)
 * - Energy decreases properly during optimization
 * - Test case: 1 subdivision sphere, 10 iterations
 *   Initial energy: 12.79 → Final: 5.37 (-58%)
 *   See: demos/developable-sphere/debug-paper-energy.ts
 *
 * Properties:
 * - Cheaper than E^P: just eigenvalue computation per vertex
 * - Smooth (except at repeated eigenvalues)
 * - Tessellation-invariant (due to angle weighting)
 *
 * USAGE NOTES:
 * - Mesh vertices must be trainable (Vec3.W) for gradients to flow
 * - DevelopableOptimizer automatically converts constant vertices to trainable
 * - For direct testing, manually convert with:
 *     params = vertices.map(v => [V.W(v.x.data), V.W(v.y.data), V.W(v.z.data)])
 *     mesh.vertices = params.map((p, i) => new Vec3(p[i*3], p[i*3+1], p[i*3+2]))
 */
export class PaperCovarianceEnergyELambda {
  static readonly name = 'PaperCovarianceEnergyELambda';
  static readonly description = 'Paper: E^λ = Σλ_min(A_i), angle-weighted, custom grad';
  static readonly supportsCompilation = true;

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
   * E^λ_i = λ_min(A_i), where A_i = Σ θ_ijk · Nfw_ijk · Nfw_ijk^T
   *
   * Nfw = tangent-plane projection: normalize((Nv × Nf) × Nv) * φ
   * where Nv is the vertex normal, Nf is the face normal, and φ = acos(Nv·Nf)
   */
  static computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);
    if (star.length === 3) return V.C(0);

    // Compute area-weighted vertex normal (matches C++ reference)
    let vertexNormalRaw = Vec3.zero();
    for (const faceIdx of star) {
      const faceNormal = mesh.getFaceNormal(faceIdx).normalized;
      const faceArea = mesh.getFaceArea(faceIdx);
      vertexNormalRaw = vertexNormalRaw.add(faceNormal.mul(faceArea));
    }
    const Nv = vertexNormalRaw.normalized;

    // Build angle-weighted tangent-plane covariance matrix
    let c00 = V.C(0),
      c01 = V.C(0),
      c02 = V.C(0);
    let c11 = V.C(0),
      c12 = V.C(0),
      c22 = V.C(0);

    for (const faceIdx of star) {
      const Nf = mesh.getFaceNormal(faceIdx).normalized;
      const theta = mesh.getInteriorAngle(faceIdx, vertexIdx);

      // Project face normal to tangent plane: (Nv × Nf) × Nv
      const cross1 = Vec3.cross(Nv, Nf);
      const cross1Mag = cross1.magnitude;

      // Skip if normals are nearly parallel (cross product ≈ 0)
      if (cross1Mag.data < 1e-12) continue;

      const Nfw_unnormalized = Vec3.cross(cross1, Nv);
      const NfwMag = Nfw_unnormalized.magnitude;

      // Normalize the tangent projection
      const epsilon = V.C(1e-12);
      const safeNfwMag = V.max(NfwMag, epsilon);
      const Nfw_normalized = Nfw_unnormalized.div(safeNfwMag);

      // Scale by angle φ = acos(Nv·Nf)
      const cosAngle = Vec3.dot(Nv, Nf);
      const phi = V.acos(V.clamp(cosAngle, -0.99999, 0.99999));
      const Nfw = Nfw_normalized.mul(phi);

      // Add weighted outer product: θ · Nfw · Nfw^T
      c00 = V.add(c00, V.mul(theta, V.mul(Nfw.x, Nfw.x)));
      c01 = V.add(c01, V.mul(theta, V.mul(Nfw.x, Nfw.y)));
      c02 = V.add(c02, V.mul(theta, V.mul(Nfw.x, Nfw.z)));
      c11 = V.add(c11, V.mul(theta, V.mul(Nfw.y, Nfw.y)));
      c12 = V.add(c12, V.mul(theta, V.mul(Nfw.y, Nfw.z)));
      c22 = V.add(c22, V.mul(theta, V.mul(Nfw.z, Nfw.z)));
    }

    // Add numerical jitter for stability (δI with δ ≈ 1e-12)
    const jitter = V.C(1e-12);
    c00 = V.add(c00, jitter);
    c11 = V.add(c11, jitter);
    c22 = V.add(c22, jitter);

    // Compute smallest eigenvalue with custom analytical gradients
    const lambda = Matrix3x3.smallestEigenvalueCustomGrad(c00, c01, c02, c11, c12, c22);

    return V.max(lambda, V.C(0));
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

    return V.max(lambda, V.C(0));
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

}

// Register with energy registry
EnergyRegistry.register(PaperCovarianceEnergyELambda);
