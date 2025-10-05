import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './EnergyRegistry';

/**
 * Alignment-based Bimodal Energy - CUSTOM IMPLEMENTATION (not from Stein et al. 2018 paper).
 *
 * Experimental alternative using alignment metric instead of variance.
 *
 * Instead of minimizing variance (scatter around mean), this energy measures
 * how well normals align with their group's average normal direction.
 *
 * Key difference from variance:
 * - Variance: sum of squared distances from mean (penalizes spread)
 * - Alignment: sum of (1 - dot(normal, mean)) (penalizes misalignment)
 *
 * This is more directly related to the developability condition: we want
 * normals in each group to point in the same direction (dot product = 1),
 * not just cluster together in space.
 *
 * **Limitations**:
 * - Uses fixed midpoint split (not optimal partition search like paper's E^P)
 * - Not the paper's recommended energy function
 */
export class AlignmentBimodalEnergy {
  static readonly name = 'Bimodal Alignment (Spatial Midpoint)';
  static readonly description = 'Custom: midpoint split, alignment metric';

  /**
   * Compute total energy for the mesh.
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
   * Compute alignment energy for a single vertex.
   *
   * Algorithm:
   * 1. Split vertex star at midpoint (quasi-random, same as DevelopableEnergy)
   * 2. For each group, compute average normal direction
   * 3. Measure alignment: sum of (1 - dot(normal_i, avg_normal))
   * 4. Perfect alignment: dot = 1, energy = 0
   * 5. Orthogonal: dot = 0, energy = 1
   * 6. Opposite: dot = -1, energy = 2
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

    // Split into two regions at midpoint (quasi-random, consistent per vertex)
    const mid = Math.floor(star.length / 2);
    const partition = [
      Array.from({ length: mid }, (_, i) => i),
      Array.from({ length: star.length - mid }, (_, i) => i + mid)
    ];

    // Compute alignment energy within each group
    const align1 = this.computeRegionAlignment(partition[0], normals);
    const align2 = this.computeRegionAlignment(partition[1], normals);

    // Total energy = sum of misalignments
    return V.add(align1, align2);
  }

  /**
   * Compute alignment energy for a region.
   *
   * Energy = sum over normals of (1 - dot(normal, avg_normal))
   *
   * This measures how well each normal aligns with the group average.
   * - If all normals are perfectly aligned: energy = 0
   * - If normals are random: energy ≈ n (where n = number of normals)
   */
  private static computeRegionAlignment(indices: number[], allNormals: Vec3[]): Value {
    if (indices.length === 0) return V.C(0);

    const normals = indices.map(i => allNormals[i]);
    const n = normals.length;
    if (n < 2) return V.C(0);

    // Compute mean normal (unnormalized for now)
    let meanNormal = Vec3.zero();
    for (const normal of normals) {
      meanNormal = meanNormal.add(normal);
    }
    meanNormal = meanNormal.div(n);

    // Normalize the mean
    const meanNormalized = meanNormal.normalized;

    // Compute alignment: sum of (1 - dot(n_i, mean))
    const contributions: Value[] = [];
    for (const normal of normals) {
      const dotProd = Vec3.dot(normal, meanNormalized);
      // Clamp dot product to [-1, 1] for numerical stability
      const clampedDot = V.clamp(dotProd, -1, 1);
      // Energy contribution: 1 - dot (ranges from 0 to 2)
      const contribution = V.sub(V.C(1), clampedDot);
      contributions.push(contribution);
    }

    const misalignment = V.sum(contributions);
    // Average over the region
    return V.div(misalignment, n);
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
EnergyRegistry.register(AlignmentBimodalEnergy);
