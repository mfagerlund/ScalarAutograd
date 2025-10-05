import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './EnergyRegistry';

/**
 * Contiguous Bimodal Clustering Energy
 *
 * Similar to DevelopableEnergy but uses a contiguous (edge-connected) partition
 * instead of an arbitrary split. Grows two regions from opposite sides of the
 * vertex star to ensure both groups are spatially connected.
 *
 * This addresses the concern that the fixed midpoint split in DevelopableEnergy
 * might only work due to lucky mesh topology from icosphere generation.
 */
export class ContiguousBimodalEnergy {
  static readonly name = 'Contiguous Bimodal Variance';
  static readonly description = 'Custom: contiguous partition, variance';
  static readonly supportsCompilation = true;
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
   * Compute bimodal clustering energy with contiguous partition.
   *
   * Algorithm:
   * 1. Build adjacency graph of faces in the vertex star
   * 2. Pick two "seed" faces on opposite sides (first and ~middle)
   * 3. Grow both regions simultaneously via BFS until all faces assigned
   * 4. Compute variance within each region
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

    // Build adjacency: which faces share edges?
    const adjacency = this.buildAdjacency(star, mesh);

    // Grow contiguous partition
    const partition = this.growContiguousPartition(star, adjacency);

    // Compute variance within each group
    const var1 = this.computeRegionVariance(partition[0], normals);
    const var2 = this.computeRegionVariance(partition[1], normals);

    return V.add(var1, var2);
  }

  /**
   * Build adjacency map for faces in the star.
   * Two faces are adjacent if they share an edge.
   */
  private static buildAdjacency(star: number[], mesh: TriangleMesh): Map<number, number[]> {
    const adjacency = new Map<number, number[]>();

    for (const faceIdx of star) {
      adjacency.set(faceIdx, []);
    }

    for (let i = 0; i < star.length; i++) {
      for (let j = i + 1; j < star.length; j++) {
        if (mesh.facesShareEdge(star[i], star[j])) {
          adjacency.get(star[i])!.push(star[j]);
          adjacency.get(star[j])!.push(star[i]);
        }
      }
    }

    return adjacency;
  }

  /**
   * Grow two contiguous regions from opposite seed points.
   * Returns partition as [region1_indices, region2_indices] into the normals array.
   */
  private static growContiguousPartition(
    star: number[],
    adjacency: Map<number, number[]>
  ): [number[], number[]] {
    if (star.length < 2) return [[0], []];

    // Seed faces: first and roughly opposite (middle of array)
    const seed1 = star[0];
    const seed2 = star[Math.floor(star.length / 2)];

    const assigned = new Set<number>();
    const region1Faces = new Set<number>([seed1]);
    const region2Faces = new Set<number>([seed2]);
    assigned.add(seed1);
    assigned.add(seed2);

    // BFS queues
    const queue1 = [seed1];
    const queue2 = [seed2];

    // Grow both regions simultaneously
    while (assigned.size < star.length) {
      let grown = false;

      // Try to grow region 1
      if (queue1.length > 0) {
        const current = queue1.shift()!;
        for (const neighbor of adjacency.get(current) || []) {
          if (!assigned.has(neighbor)) {
            region1Faces.add(neighbor);
            assigned.add(neighbor);
            queue1.push(neighbor);
            grown = true;
          }
        }
      }

      // Try to grow region 2
      if (queue2.length > 0) {
        const current = queue2.shift()!;
        for (const neighbor of adjacency.get(current) || []) {
          if (!assigned.has(neighbor)) {
            region2Faces.add(neighbor);
            assigned.add(neighbor);
            queue2.push(neighbor);
            grown = true;
          }
        }
      }

      // If neither region can grow, assign remaining faces to smaller region
      if (!grown) {
        const remaining = star.filter(f => !assigned.has(f));
        const targetRegion = region1Faces.size < region2Faces.size ? region1Faces : region2Faces;
        for (const face of remaining) {
          targetRegion.add(face);
          assigned.add(face);
        }
      }
    }

    // Convert face indices to normal array indices
    const region1Indices: number[] = [];
    const region2Indices: number[] = [];

    for (let i = 0; i < star.length; i++) {
      if (region1Faces.has(star[i])) {
        region1Indices.push(i);
      } else {
        region2Indices.push(i);
      }
    }

    return [region1Indices, region2Indices];
  }

  /**
   * Compute variance of normals in a region (trace of covariance matrix).
   */
  private static computeRegionVariance(indices: number[], allNormals: Vec3[]): Value {
    if (indices.length === 0) return V.C(0);

    const normals = indices.map(i => allNormals[i]);
    const n = normals.length;
    if (n < 2) return V.C(0);

    // Compute mean normal
    let meanNormal = Vec3.zero();
    for (const normal of normals) {
      meanNormal = meanNormal.add(normal);
    }
    meanNormal = meanNormal.div(n);

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

  /**
   * Classify vertices as hinges or seams based on energy threshold
   */
  static classifyVertices(
    mesh: TriangleMesh,
    hingeThreshold: number = 1e-3
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
EnergyRegistry.register(ContiguousBimodalEnergy);
