import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './EnergyRegistry';

/**
 * Bounding box energy for developable surfaces.
 * Measures the spread of face normals on the unit sphere using a bounding box metric.
 * More compact normal distributions indicate better developability.
 */
export class BoundingBoxEnergy {
  static readonly name = 'Bounding Box Spread';
  /**
   * Compute total bounding box energy for the mesh.
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
   * Returns array of residuals (one per vertex).
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
   * Compute bounding box energy for a single vertex.
   */
  static computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);

    // Split into two regions
    const mid = Math.floor(star.length / 2);
    const region1 = star.slice(0, mid);
    const region2 = star.slice(mid);

    const energy1 = this.computeRegionBoundingBox(region1, mesh);
    const energy2 = this.computeRegionBoundingBox(region2, mesh);

    return V.add(energy1, energy2);
  }

  private static computeRegionBoundingBox(region: number[], mesh: TriangleMesh): Value {
    if (region.length === 0) return V.C(0);

    // Get all normals in this region
    const normals: Vec3[] = [];
    for (const faceIdx of region) {
      normals.push(mesh.getFaceNormal(faceIdx).normalized);
    }

    // Find maximum pairwise distance (longest axis)
    let maxSpread1 = V.C(0);
    for (let i = 0; i < normals.length; i++) {
      for (let j = i + 1; j < normals.length; j++) {
        const dist = normals[i].sub(normals[j]).magnitude;
        maxSpread1 = V.max(maxSpread1, dist);
      }
    }

    // Compute perpendicular spread using average normal
    const avgNormal = normals.reduce((sum, n) => sum.add(n), Vec3.zero()).div(normals.length);

    let perpSpread = V.C(0);
    for (const n of normals) {
      const proj = Vec3.dot(n.sub(avgNormal), avgNormal.normalized);
      const perp = n.sub(avgNormal.normalized.mul(proj));
      perpSpread = V.add(perpSpread, perp.magnitude);
    }
    perpSpread = V.div(perpSpread, normals.length);

    // Bounding box area = longest axis * perpendicular spread
    return V.mul(maxSpread1, perpSpread);
  }

  /**
   * Classify vertices as hinges or seams based on energy threshold
   */
  static classifyVertices(
    mesh: TriangleMesh,
    hingeThreshold: number = 0.3
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
EnergyRegistry.register(BoundingBoxEnergy);
