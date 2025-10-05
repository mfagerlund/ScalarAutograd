import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './EnergyRegistry';

/**
 * Ridge-Based Bimodal Energy - CUSTOM IMPLEMENTATION (not from Stein et al. 2018 paper).
 *
 * Experimental heuristic using dihedral angle detection.
 *
 * PRINCIPLE: Find natural hinge edges by detecting dihedral angle discontinuities.
 *
 * For a developable surface folded at a hinge:
 * - Faces on each side are coplanar (small dihedral angles within each group)
 * - At the hinge edge, there's a sharp angle change (large dihedral angle)
 *
 * Algorithm:
 * 1. Compute dihedral angles at all edges in the vertex star
 * 2. Find the TWO edges with the maximum dihedral angles (entry and exit ridges)
 * 3. Split the star between these two ridges
 * 4. Measure alignment within each side
 * 5. Penalize tiny segments (n < 2) to avoid degenerate single-face clusters
 *
 * This is geometrically motivated: hinges form at ridges/valleys where
 * the surface bends sharply.
 *
 * **Properties**:
 * - Finds TWO ridges: detects entry and exit of the hinge (not arbitrary halfway split)
 * - Deterministic and cheap: O(k log k) for sorting dihedral angles
 * - Regularized: penalizes 1-face segments with zero intra-cluster error
 * - Smooth alignment metric: average (1 - dot(normal, mean))
 *
 * **Limitations**:
 * - Discrete ridge selection (non-differentiable, uses .data comparison)
 * - Unsigned dihedral (ignores convex/concave sign - would need edge vectors)
 * - Not the paper's recommended energy function
 */
export class RidgeBasedEnergy {
  static readonly name = 'Bimodal Alignment (Ridge Detection)';
  static readonly className = 'RidgeBasedEnergy';
  static readonly description = 'Custom: split at max dihedral angle';
  static readonly supportsCompilation = false;

  static compute(mesh: TriangleMesh): Value {
    const vertexEnergies: Value[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      vertexEnergies.push(this.computeVertexEnergy(i, mesh));
    }

    return V.sum(vertexEnergies);
  }

  static computeResiduals(mesh: TriangleMesh): Value[] {
    const residuals: Value[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      residuals.push(this.computeVertexEnergy(i, mesh));
    }

    return residuals;
  }

  /**
   * Compute bimodal alignment energy using ridge detection.
   */
  static computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);
    if (star.length === 3) return V.C(0); // Skip valence-3 (triple points per paper)

    // Get all normals
    const normals: Vec3[] = [];
    for (const faceIdx of star) {
      normals.push(mesh.getFaceNormal(faceIdx).normalized);
    }

    // Find the two ridge edges (entry and exit of the hinge)
    const [ridge1, ridge2] = this.findTwoRidges(normals);

    // Split between the two ridges
    const partition: [number, number] = [ridge1, ridge2];

    // Compute alignment energy for this split
    return this.computePartitionAlignment(partition, normals);
  }

  /**
   * Find the two edges with maximum dihedral angles (the two ridges).
   * Returns [ridge1, ridge2] indices defining the split between the two ridges.
   *
   * For a developable surface with a hinge, there are typically TWO ridge
   * lines (entry and exit of the bend). This finds both and splits between them.
   *
   * The split creates:
   * - Segment 1: [ridge1, ridge2) - one side of the hinge
   * - Segment 2: [ridge2, ridge1) - the other side of the hinge
   */
  private static findTwoRidges(normals: Vec3[]): [number, number] {
    const n = normals.length;
    if (n < 2) return [0, 1];

    // Compute all dihedral angles
    const angles: { idx: number; angle: number }[] = [];
    for (let i = 0; i < n; i++) {
      const next = (i + 1) % n;
      const dotProd = Vec3.dot(normals[i], normals[next]).data;
      const angle = Math.acos(Math.max(-1, Math.min(1, dotProd)));
      angles.push({ idx: next, angle });
    }

    // Sort by angle descending
    angles.sort((a, b) => b.angle - a.angle);

    // Take the top two ridges (highest dihedral angles)
    const ridge1 = angles[0].idx;
    const ridge2 = angles.length > 1 ? angles[1].idx : (ridge1 + Math.floor(n / 2)) % n;

    // Ensure ridge1 < ridge2 for consistent ordering
    if (ridge1 < ridge2) {
      return [ridge1, ridge2];
    } else {
      return [ridge2, ridge1];
    }
  }

  /**
   * Compute alignment energy for a partition.
   */
  private static computePartitionAlignment(
    partition: [number, number],
    normals: Vec3[]
  ): Value {
    const [start, end] = partition;
    const n = normals.length;

    // Build segment indices
    const segment1: number[] = [];
    const segment2: number[] = [];

    if (start < end) {
      for (let i = start; i < end; i++) segment1.push(i);
      for (let i = end; i < n; i++) segment2.push(i);
      for (let i = 0; i < start; i++) segment2.push(i);
    } else {
      for (let i = start; i < n; i++) segment1.push(i);
      for (let i = 0; i < end; i++) segment1.push(i);
      for (let i = end; i < start; i++) segment2.push(i);
    }

    const align1 = this.computeSegmentAlignment(segment1, normals);
    const align2 = this.computeSegmentAlignment(segment2, normals);

    return V.add(align1, align2);
  }

  /**
   * Compute alignment energy for a segment.
   *
   * Regularizes tiny segments (n < 2) to avoid degenerate cases where
   * single-face segments get zero energy.
   */
  private static computeSegmentAlignment(indices: number[], normals: Vec3[]): Value {
    if (indices.length === 0) return V.C(0);

    const segmentNormals = indices.map(i => normals[i]);
    const n = segmentNormals.length;

    // Penalize tiny segments (n < 2) to avoid degenerate single-face clusters
    if (n < 2) {
      return V.C(1.0); // High penalty for 1-face segments
    }

    // Compute mean normal
    let meanNormal = Vec3.zero();
    for (const normal of segmentNormals) {
      meanNormal = meanNormal.add(normal);
    }
    meanNormal = meanNormal.div(n).normalized;

    // Compute alignment: sum of (1 - dot(n_i, mean))
    const misalignments: Value[] = [];
    for (const normal of segmentNormals) {
      const dotProd = Vec3.dot(normal, meanNormal);
      const clampedDot = V.clamp(dotProd, -1, 1);
      misalignments.push(V.sub(V.C(1), clampedDot));
    }

    return V.div(V.sum(misalignments), n);
  }

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
EnergyRegistry.register(RidgeBasedEnergy);
