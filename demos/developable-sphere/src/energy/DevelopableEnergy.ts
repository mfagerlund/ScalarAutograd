import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';

export class DevelopableEnergy {
  /**
   * Compute total developability energy for the mesh.
   * This is the sum of vertex energies over all vertices.
   */
  static compute(mesh: TriangleMesh): Value {
    let total = V.C(0);

    for (let i = 0; i < mesh.vertices.length; i++) {
      const vertexEnergy = this.computeVertexEnergy(i, mesh);
      total = V.add(total, vertexEnergy);
    }

    return total;
  }

  /**
   * Compute developability energy for a single vertex.
   * Uses a simple heuristic partition and computes smooth energy.
   */
  static computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);

    if (star.length < 2) {
      return V.C(0); // Boundary vertex or isolated
    }

    // Simple heuristic: partition into two halves
    // This works well for convex vertex stars
    const mid = Math.floor(star.length / 2);
    const region1 = star.slice(0, mid);
    const region2 = star.slice(mid);

    // Compute energy for each region
    const energy1 = this.computeRegionEnergy(region1, mesh);
    const energy2 = this.computeRegionEnergy(region2, mesh);

    // Total energy is sum of both regions
    return V.add(energy1, energy2);
  }

  /**
   * Compute energy for a single region.
   * Measures how far the normals deviate from being coplanar.
   */
  private static computeRegionEnergy(region: number[], mesh: TriangleMesh): Value {
    const n = region.length;

    if (n === 0) {
      return V.C(0);
    }

    // Compute average normal
    let avgNormal = Vec3.zero();
    for (const faceIdx of region) {
      const normal = mesh.getFaceNormal(faceIdx);
      avgNormal = avgNormal.add(normal);
    }
    avgNormal = avgNormal.div(n);

    // Sum squared deviations from average
    let deviation = V.C(0);
    for (const faceIdx of region) {
      const normal = mesh.getFaceNormal(faceIdx);
      const diff = normal.sub(avgNormal);
      deviation = V.add(deviation, diff.sqrMagnitude);
    }

    // Normalize by n²
    const normalizationFactor = n * n;
    return V.div(deviation, normalizationFactor);
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
