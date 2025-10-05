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
 * 2. Find the edge with the maximum dihedral angle (the "ridge")
 * 3. Split the star at this edge
 * 4. Measure alignment within each side
 *
 * This is geometrically motivated: hinges form at ridges/valleys where
 * the surface bends sharply.
 *
 * **Limitations**:
 * - Discrete ridge selection (non-differentiable)
 * - Not the paper's recommended energy function
 */
export class RidgeBasedEnergy {
  static readonly name = 'Bimodal Alignment (Ridge Detection)';
  static readonly description = 'Custom: split at max dihedral angle';

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

    // Find the edge with maximum dihedral angle (the ridge)
    const ridgeEdgeIdx = this.findRidgeEdge(star, normals);

    // Split at the ridge edge
    const partition = this.createSplitAtEdge(ridgeEdgeIdx, star.length);

    // Compute alignment energy for this split
    return this.computePartitionAlignment(partition, normals);
  }

  /**
   * Find the edge with the maximum dihedral angle.
   * Returns the index in the star where the ridge occurs.
   */
  private static findRidgeEdge(_star: number[], normals: Vec3[]): number {
    const n = normals.length;
    if (n < 2) return 0;

    let maxAngle = -1;
    let ridgeIdx = 0;

    // Check dihedral angle between each pair of adjacent faces
    for (let i = 0; i < n; i++) {
      const next = (i + 1) % n;

      // Dihedral angle = angle between normals
      // cos(angle) = n1 · n2
      const dotProd = Vec3.dot(normals[i], normals[next]).data;
      const angle = Math.acos(Math.max(-1, Math.min(1, dotProd)));

      if (angle > maxAngle) {
        maxAngle = angle;
        ridgeIdx = next; // Split AFTER the edge with max angle
      }
    }

    return ridgeIdx;
  }

  /**
   * Create a bipartition split at the given edge index.
   * Returns [start, end] where:
   * - Segment 1: [start, end)
   * - Segment 2: [end, start)
   */
  private static createSplitAtEdge(edgeIdx: number, starSize: number): [number, number] {
    if (starSize < 2) return [0, 1];

    // Split at the ridge edge
    // Segment 1 starts at the ridge and goes halfway around
    const start = edgeIdx;
    const length = Math.floor(starSize / 2);
    const end = (start + length) % starSize;

    return [start, end];
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
   */
  private static computeSegmentAlignment(indices: number[], normals: Vec3[]): Value {
    if (indices.length === 0) return V.C(0);

    const segmentNormals = indices.map(i => normals[i]);
    const n = segmentNormals.length;

    if (n < 1) return V.C(0);

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
