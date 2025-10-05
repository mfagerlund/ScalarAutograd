import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';

/**
 * Base class for developable energy functions with common patterns.
 *
 * Handles:
 * - Valence-3 vertex skipping (per paper: they must be flat, serve as triple points)
 * - Star validation (< 2 faces = no energy)
 * - Standard compute/computeResiduals patterns
 */
export abstract class BaseEnergy {
  abstract readonly name: string;
  abstract readonly description: string;

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
      residuals.push(this.computeVertexEnergy(i, mesh));
    }

    return residuals;
  }

  /**
   * Compute energy for a single vertex.
   * Subclasses implement the actual energy computation.
   */
  protected static computeVertexEnergy(_vertexIdx: number, _mesh: TriangleMesh): Value {
    throw new Error('Must implement computeVertexEnergy in subclass');
  }

  /**
   * Standard vertex validation and skip logic.
   * Returns null if vertex should be skipped, otherwise returns star.
   */
  protected static validateVertex(vertexIdx: number, mesh: TriangleMesh): number[] | null {
    const star = mesh.getVertexStar(vertexIdx);

    // Skip degenerate vertices
    if (star.length < 2) return null;

    // Skip valence-3 vertices (triple points per paper)
    // Protocol line 145: "Omit valence-3 vertices from the energy (they must be flat)"
    if (star.length === 3) return null;

    return star;
  }

  /**
   * Gather normalized normals from vertex star.
   */
  protected static getStarNormals(star: number[], mesh: TriangleMesh): Vec3[] {
    const normals: Vec3[] = [];
    for (const faceIdx of star) {
      normals.push(mesh.getFaceNormal(faceIdx).normalized);
    }
    return normals;
  }

  /**
   * Gather corner angles from vertex star.
   */
  protected static getStarAngles(vertexIdx: number, star: number[], mesh: TriangleMesh): Value[] {
    const angles: Value[] = [];
    for (const faceIdx of star) {
      angles.push(mesh.getInteriorAngle(faceIdx, vertexIdx));
    }
    return angles;
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
}
