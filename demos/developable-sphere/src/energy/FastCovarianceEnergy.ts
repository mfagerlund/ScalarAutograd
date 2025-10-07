import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './utils/EnergyRegistry';

/**
 * Fast Covariance Energy (Grid) - CUSTOM IMPLEMENTATION
 *
 * Fast approximation to E^λ using icosahedral sampling.
 * Instead of computing eigenvalues, we test 12 fixed directions
 * and pick the minimum.
 *
 * - Deterministic (same 12 directions always)
 * - Fast (~12k FLOPs where k = valence)
 * - Compilable (static graph structure)
 */
export class FastCovarianceGrid {
  static readonly name = 'Fast Covariance (Grid)';
  static readonly className = 'FastCovarianceGrid';
  static readonly description = 'Custom: E^λ approx via icosahedral sampling';
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
      residuals.push(this.computeVertexEnergy(i, mesh));
    }

    return residuals;
  }

  static computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);
    if (star.length === 3) return V.C(0);

    const normalsValue: Vec3[] = [];
    const weights: Value[] = [];

    for (const faceIdx of star) {
      const normalValue = mesh.getFaceNormal(faceIdx).normalized;
      const weight = mesh.getInteriorAngle(faceIdx, vertexIdx);

      normalsValue.push(normalValue);
      weights.push(weight);
    }

    const directions = this.getIcosahedralDirections();

    let minLambda: Value | null = null;
    for (const u of directions) {
      let lambda = V.C(0);
      for (let k = 0; k < normalsValue.length; k++) {
        const N = normalsValue[k];
        const dotProd = V.add(
          V.add(V.mul(N.x, u[0]), V.mul(N.y, u[1])),
          V.mul(N.z, u[2])
        );
        lambda = V.add(lambda, V.mul(weights[k], V.mul(dotProd, dotProd)));
      }

      if (minLambda === null) {
        minLambda = lambda;
      } else {
        minLambda = V.min(minLambda, lambda);
      }
    }

    return minLambda === null ? V.C(0) : V.max(minLambda, V.C(0));
  }

  private static getIcosahedralDirections(): number[][] {
    const phi = (1 + Math.sqrt(5)) / 2;
    const invNorm = 1 / Math.sqrt(1 + phi * phi);

    return [
      [1, phi, 0].map(x => x * invNorm),
      [-1, phi, 0].map(x => x * invNorm),
      [1, -phi, 0].map(x => x * invNorm),
      [-1, -phi, 0].map(x => x * invNorm),
      [0, 1, phi].map(x => x * invNorm),
      [0, -1, phi].map(x => x * invNorm),
      [0, 1, -phi].map(x => x * invNorm),
      [0, -1, -phi].map(x => x * invNorm),
      [phi, 0, 1].map(x => x * invNorm),
      [-phi, 0, 1].map(x => x * invNorm),
      [phi, 0, -1].map(x => x * invNorm),
      [-phi, 0, -1].map(x => x * invNorm),
    ];
  }

}

EnergyRegistry.register(FastCovarianceGrid);
