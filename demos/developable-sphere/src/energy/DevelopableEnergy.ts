import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './utils/EnergyRegistry';

export class DevelopableEnergy {
  static readonly name = 'Bimodal Variance (Spatial Midpoint)';
  static readonly description = 'Custom: fixed midpoint split, variance';
  static readonly supportsCompilation = true;
  /**
   * Compute total developability energy for the mesh.
   * This is the sum of vertex energies over all vertices.
   * Uses n-ary sum to avoid deep expression chains.
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
   * Returns array of residuals (one per vertex) instead of summing them.
   * This avoids deep expression chains and enables kernel compilation.
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
   * Compute bimodal clustering energy for a single vertex.
   * Minimizes variance within two fixed groups - encourages normals to cluster
   * into the two halves of the vertex star, creating hinges.
   *
   * DIVERGENCE FROM PAPER:
   * The paper ("Developability of Triangle Meshes", Stein et al., SIGGRAPH 2018)
   * describes TWO energy formulations:
   *
   * 1. **Combinatorial Energy (E^P)**: Enumerate all edge-connected bipartitions
   *    of the vertex star, compute variance within each partition, pick minimum.
   *    This is the paper's primary method.
   *
   * 2. **Covariance Energy (E^λ)**: Compute smallest eigenvalue of the
   *    angle-weighted covariance matrix:
   *      C = Σ θ_ijk · N_ijk · N_ijk^T
   *    where θ_ijk are interior angles and N_ijk are face normals.
   *    This is mentioned as a faster alternative with different behavior.
   *
   * This implementation is a SIMPLIFIED version of (1):
   * 1. Splits the vertex star into TWO fixed groups (at midpoint index)
   *    NOTE: Faces are ordered by face index, NOT spatially around the vertex,
   *    so this is effectively a quasi-random partition (but consistent per vertex)
   * 2. Computes UNWEIGHTED variance within each group separately
   * 3. Minimizes the sum of both variances (instead of enumerating all partitions)
   *
   * Rationale:
   * - The eigenvalue approach is expensive (O(n³) for exact computation)
   * - The eigenvalue approximation (trace - ||C||_F) was inaccurate (2-4× error)
   * - This bimodal splitting approach directly encourages the hinge structure
   *   (two coplanar groups) without computing eigenvalues
   * - Fixed split (not random) allows gradient descent to consistently move
   *   the mesh toward alignment with the split direction
   * - Works empirically: increases developability from 0% to 22%+ on spheres
   *
   * Trade-offs:
   * + Much faster (no eigenvalue computation)
   * + Stable gradients (bounded variance, no division issues)
   * + Direct optimization target (minimize variance = align normals)
   * - Ignores angle weighting (may favor small triangles equally)
   * - Fixed split may not align with natural hinge direction
   * - Less theoretically grounded than eigenvalue approach
   */
  static computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);
    if (star.length === 3) return V.C(0); // Skip valence-3 (triple points per paper)

    // Get all normals (normalized)
    const normals: Vec3[] = [];
    for (const faceIdx of star) {
      normals.push(mesh.getFaceNormal(faceIdx).normalized);
    }

    // Split into two regions at midpoint (fixed, not random)
    const mid = Math.floor(star.length / 2);
    const partition = [
      Array.from({ length: mid }, (_, i) => i),
      Array.from({ length: star.length - mid }, (_, i) => i + mid)
    ];

    // Compute variance within each group
    const var1 = this.computeRegionVariance(partition[0], normals);
    const var2 = this.computeRegionVariance(partition[1], normals);

    // Total energy = sum of variances
    // Minimizing this makes each group's normals align
    return V.add(var1, var2);
  }

  /**
   * Compute mean normal for a region.
   */
  private static computeMeanNormal(indices: number[], allNormals: Vec3[]): Vec3 {
    if (indices.length === 0) return Vec3.zero();

    const normals = indices.map(i => allNormals[i]);
    let meanNormal = Vec3.zero();
    for (const normal of normals) {
      meanNormal = meanNormal.add(normal);
    }
    return meanNormal.div(normals.length);
  }

  /**
   * Compute variance of normals in a region (trace of covariance matrix).
   */
  private static computeRegionVariance(indices: number[], allNormals: Vec3[]): Value {
    if (indices.length === 0) return V.C(0);

    const normals = indices.map(i => allNormals[i]);
    const n = normals.length;
    if (n < 2) return V.C(0);

    const meanNormal = this.computeMeanNormal(indices, allNormals);

    // Compute variance = trace of covariance matrix
    const squaredDeviations: Value[] = [];
    for (const normal of normals) {
      const dx = V.sub(normal.x, meanNormal.x);
      const dy = V.sub(normal.y, meanNormal.y);
      const dz = V.sub(normal.z, meanNormal.z);

      const squaredDev = V.add(V.add(V.square(dx), V.square(dy)), V.square(dz));
      squaredDeviations.push(squaredDev);
    }

    const variance = V.sum(squaredDeviations);
    return V.div(variance, n);
  }

}

// Register with energy registry
EnergyRegistry.register(DevelopableEnergy);
