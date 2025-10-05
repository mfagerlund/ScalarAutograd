import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './EnergyRegistry';

/**
 * Great Circle Energy Extended - CUSTOM IMPLEMENTATION (not from Stein et al. 2018 paper).
 *
 * Weighted and normalized variant of GreatCircleEnergy.
 *
 * Improvements over base GreatCircleEnergy:
 * 1. Angle weighting: uses interior angles θ_ijk for tessellation invariance (paper-inspired)
 * 2. Valence-3 skip: allows triple points (from paper recommendation)
 * 3. Normalize by count: scale-invariant across different valences
 *
 * **Limitations**:
 * Still uses discrete max-separated pair selection (non-differentiable choice).
 * Not the paper's recommended covariance energy.
 */
export class GreatCircleEnergyEx {
  static readonly name = 'Great Circle Extended';
  static readonly description = 'Custom: angle-weighted, max-sep pair';
  static readonly supportsCompilation = true;

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
      const vertexEnergy = this.computeVertexEnergy(i, mesh);
      residuals.push(vertexEnergy);
    }

    return residuals;
  }

  static computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);
    if (star.length === 3) return V.C(0); // Skip valence-3 (triple points)

    // Get all normals and angles
    const normals: Vec3[] = [];
    const angles: Value[] = [];
    for (const faceIdx of star) {
      normals.push(mesh.getFaceNormal(faceIdx).normalized);
      angles.push(mesh.getInteriorAngle(faceIdx, vertexIdx));
    }

    // Find most separated normal pair (discrete choice)
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

    // Compute plane normal
    const planeNormal = Vec3.cross(n1, n2);
    const planeNormalMag = planeNormal.magnitude;

    const epsilon = V.C(1e-12);
    const safeMag = V.max(planeNormalMag, epsilon);
    const planeNormalNormalized = planeNormal.div(safeMag);

    // Angle-weighted energy
    let energy = V.C(0);
    for (let i = 0; i < normals.length; i++) {
      const dist = V.abs(Vec3.dot(normals[i], planeNormalNormalized));
      const weightedDist = V.mul(angles[i], V.mul(dist, dist));
      energy = V.add(energy, weightedDist);
    }

    // Normalize by vertex count
    return V.div(energy, star.length);
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

EnergyRegistry.register(GreatCircleEnergyEx);
