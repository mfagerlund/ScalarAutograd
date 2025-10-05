import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './EnergyRegistry';

/**
 * Great Circle Energy - CUSTOM IMPLEMENTATION (not from Stein et al. 2018 paper).
 *
 * This is an experimental alternative to the paper's covariance energy.
 * Measures deviation from normals lying on a great circle.
 *
 * **Concept**:
 * A developable vertex has normals forming a 1D arc on the Gaussian sphere (not a 2D patch).
 * This energy:
 * 1. Finds the two most separated normals → they define a plane through the origin
 * 2. Measures how far all other normals deviate from that plane
 * 3. Zero energy = all normals lie on a great circle = developable
 *
 * **Why this works**:
 * - Perfect hinge: normals form an arc → all coplanar → energy = 0
 * - Non-developable: normals span 2D patch → some far from any plane → energy > 0
 * - Unlike covariance, this explicitly minimizes the **minor axis** of the Gauss image
 *
 * **Implementation**:
 * - For each vertex, find max-separated normal pair (i, j)
 * - Compute plane normal: p = normalize(n_i × n_j)
 * - Energy = Σ |n_k · p|² for all normals n_k
 *
 * **Limitations**:
 * - Uses discrete max-selection (non-differentiable choice)
 * - No angle weighting (not tessellation-invariant like paper's version)
 *
 * This is simpler than eigenvalue methods and directly targets collapsing
 * the Gauss image onto a 1D curve.
 */
export class GreatCircleEnergy {
  static readonly name = 'Great Circle (Minor Axis)';
  static readonly description = 'Custom: max-sep pair defines plane';
  static readonly supportsCompilation = true;

  /**
   * Compute total great circle energy for the mesh.
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
   * Compute great circle energy for a single vertex.
   *
   * Algorithm:
   * 1. Find pair of normals with maximum angular separation
   * 2. Their cross product defines the "minor axis" direction
   * 3. Energy = sum of squared distances from all normals to the spanning plane
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

    // Find most separated normal pair
    // Separation metric: 1 - cos(angle) = 1 - (n_i · n_j)
    // Maximizing this finds the pair spanning the widest arc
    let maxSepValue = -Infinity;
    let maxSepIdx1 = 0;
    let maxSepIdx2 = 0;

    for (let i = 0; i < normals.length; i++) {
      for (let j = i + 1; j < normals.length; j++) {
        const dotProduct = Vec3.dot(normals[i], normals[j]);
        const separation = V.sub(V.C(1), dotProduct);
        if (separation.data > maxSepValue) {
          maxSepValue = separation.data;
          maxSepIdx1 = i;
          maxSepIdx2 = j;
        }
      }
    }

    const n1 = normals[maxSepIdx1];
    const n2 = normals[maxSepIdx2];

    // Compute plane normal = n1 × n2 (perpendicular to the plane they span)
    // This is the "minor axis" - the direction we want to collapse
    const planeNormal = Vec3.cross(n1, n2);
    const planeNormalMag = planeNormal.magnitude;

    // Handle degenerate case: if n1 and n2 are parallel, any plane works
    const epsilon = V.C(1e-12);
    const safeMag = V.max(planeNormalMag, epsilon);
    const planeNormalNormalized = planeNormal.div(safeMag);

    // Energy = sum of squared distances to plane
    // Distance from normal n to plane = |n · p| (since normals are unit vectors)
    let energy = V.C(0);
    for (const n of normals) {
      const dist = V.abs(Vec3.dot(n, planeNormalNormalized));
      energy = V.add(energy, V.mul(dist, dist));
    }

    return energy;
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

// Register with energy registry
EnergyRegistry.register(GreatCircleEnergy);
