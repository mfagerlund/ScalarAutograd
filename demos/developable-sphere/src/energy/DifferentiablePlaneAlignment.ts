import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './EnergyRegistry';

/**
 * Differentiable Plane Alignment Energy - CUSTOM IMPLEMENTATION (not from Stein et al. 2018 paper).
 *
 * Fully smooth great circle energy - experimental alternative to the paper's approach.
 *
 * Improvements over GreatCircleEnergy:
 * 1. Angle weighting: uses interior angles θ_ijk for tessellation invariance (paper-inspired)
 * 2. Differentiable plane selection: averages all pairwise cross products
 *    weighted by separation instead of discrete max
 * 3. Valence-3 skip: allows triple points (from paper)
 * 4. Normalize by count: scale-invariant
 *
 * **Key innovation**: No discrete "max" operation → fully differentiable.
 * The plane normal is a smooth function of all normals:
 *   p = Σ_ij w_ij · (n_i × n_j), where w_ij = 1 - n_i·n_j
 *
 * This should have better gradient flow than the discrete max variant.
 * However, this is NOT the paper's covariance energy and may not achieve the same results.
 */
export class DifferentiablePlaneAlignment {
  static readonly name = 'Differentiable Plane Alignment';
  static readonly description = 'Custom: smooth avg all cross products';
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

    // Differentiable plane selection: average all pairwise cross products
    // weighted by angles and separation for tessellation invariance
    // Normalize cross products for numerical stability
    let planeNormal = Vec3.zero();
    const epsilon = V.C(1e-12);
    for (let i = 0; i < normals.length; i++) {
      for (let j = i + 1; j < normals.length; j++) {
        const cross = Vec3.cross(normals[i], normals[j]);
        const crossMag = cross.magnitude;
        const safeCrossMag = V.max(crossMag, epsilon);
        const crossNorm = cross.div(safeCrossMag);

        const dotProduct = Vec3.dot(normals[i], normals[j]);
        const separation = V.sub(V.C(1), dotProduct); // 1 - cos(angle)
        const weight = V.mul(V.mul(angles[i], angles[j]), separation); // θ_i θ_j (1−dot)
        planeNormal = planeNormal.add(crossNorm.mul(weight));
      }
    }

    const planeNormalMag = planeNormal.magnitude;
    const safeMag = V.max(planeNormalMag, epsilon);
    const planeNormalNormalized = planeNormal.div(safeMag);

    // Angle-weighted energy
    let energy = V.C(0);
    let totalAngle = V.C(0);
    for (let i = 0; i < normals.length; i++) {
      const dot = Vec3.dot(normals[i], planeNormalNormalized);
      const dotSquared = V.mul(dot, dot);
      const weightedDist = V.mul(angles[i], dotSquared);
      energy = V.add(energy, weightedDist);
      totalAngle = V.add(totalAngle, angles[i]);
    }

    // Normalize by total angle (tessellation invariance)
    return V.div(energy, totalAngle);
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

EnergyRegistry.register(DifferentiablePlaneAlignment);
