import { Value } from './Value';
import { V } from './V';

/**
 * 3x3 symmetric matrix utilities for automatic differentiation.
 * All operations are differentiable.
 * @public
 */
export class Matrix3x3 {
  /**
   * Compute the smallest eigenvalue of a 3x3 symmetric matrix using the characteristic polynomial.
   * Uses the analytic solution for cubic equations.
   *
   * For a symmetric matrix, all eigenvalues are real.
   * The characteristic polynomial is: det(A - λI) = 0
   * Which gives: -λ³ + tr(A)λ² - (...)λ + det(A) = 0
   *
   * @param c00 - Matrix element (0,0)
   * @param c01 - Matrix element (0,1)
   * @param c02 - Matrix element (0,2)
   * @param c11 - Matrix element (1,1)
   * @param c12 - Matrix element (1,2)
   * @param c22 - Matrix element (2,2)
   * @returns Smallest eigenvalue
   */
  static smallestEigenvalue(
    c00: Value,
    c01: Value,
    c02: Value,
    c11: Value,
    c12: Value,
    c22: Value
  ): Value {
    // For a 3x3 symmetric matrix, we use Cardano's formula
    // Following https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices

    // Trace
    const tr = V.add(V.add(c00, c11), c22);

    // Shift matrix by tr/3 * I to improve numerical stability
    const shift = V.div(tr, 3);
    const b00 = V.sub(c00, shift);
    const b11 = V.sub(c11, shift);
    const b22 = V.sub(c22, shift);

    // q = (1/2) * trace(B²) where B = C - (tr/3)*I
    const q = V.mul(
      V.C(0.5),
      V.add(
        V.add(V.square(b00), V.square(b11)),
        V.add(V.add(V.square(b22), V.mul(2, V.square(c01))),
          V.add(V.mul(2, V.square(c02)), V.mul(2, V.square(c12))))
      )
    );

    // p = sqrt(q/3), with epsilon for stability
    const p = V.sqrt(V.div(V.add(q, V.C(1e-20)), 3));

    // Compute det(B / p) - need to be careful about division by zero
    const invP = V.reciprocal(V.add(p, V.C(1e-12)), 1e-12);
    const d00 = V.mul(b00, invP);
    const d01 = V.mul(c01, invP);
    const d02 = V.mul(c02, invP);
    const d11 = V.mul(b11, invP);
    const d12 = V.mul(c12, invP);
    const d22 = V.mul(b22, invP);

    // det(D) where D = B/p
    const det = V.add(
      V.add(
        V.mul(d00, V.sub(V.mul(d11, d22), V.square(d12))),
        V.mul(d01, V.sub(V.mul(d12, d02), V.mul(d01, d22)))
      ),
      V.mul(d02, V.sub(V.mul(d01, d12), V.mul(d11, d02)))
    );

    // r = det/2, clamped to [-1, 1] for acos
    const r = V.clamp(V.div(det, 2), -0.99999, 0.99999);

    // The eigenvalues are: shift + 2p*cos(θ/3 + 2πk/3) for k=0,1,2
    // where θ = acos(r)
    const theta = V.acos(r);

    // For smallest eigenvalue: use θ/3 + 2π/3 (k=1)
    const twoThirdsPI = V.C(2.0943951023931953); // 2π/3
    const angle1 = V.add(V.div(theta, 3), twoThirdsPI);
    const eig1 = V.add(shift, V.mul(V.mul(2, p), V.cos(angle1)));

    return eig1;
  }

  /**
   * Compute smallest eigenvalue with custom analytical gradients.
   * More efficient than autodiff through cubic solver - avoids huge computation graph.
   *
   * Uses analytical gradient formula for symmetric matrices:
   * ∂λ/∂C_ij = v_i * v_j where v is the eigenvector
   *
   * @param c00 - Matrix element (0,0)
   * @param c01 - Matrix element (0,1)
   * @param c02 - Matrix element (0,2)
   * @param c11 - Matrix element (1,1)
   * @param c12 - Matrix element (1,2)
   * @param c22 - Matrix element (2,2)
   * @returns Smallest eigenvalue with custom gradients
   */
  static smallestEigenvalueCustomGrad(
    c00: Value,
    c01: Value,
    c02: Value,
    c11: Value,
    c12: Value,
    c22: Value
  ): Value {
    // Forward pass: compute eigenvalue using .data (no autodiff graph)
    const lambda = this.smallestEigenvalue(
      V.C(c00.data),
      V.C(c01.data),
      V.C(c02.data),
      V.C(c11.data),
      V.C(c12.data),
      V.C(c22.data)
    ).data;

    // Compute corresponding eigenvector
    const eigenvector = this.computeEigenvector(
      c00.data, c01.data, c02.data,
      c11.data, c12.data, c22.data,
      lambda
    );

    // Create Value with custom backward pass
    return Value.makeNary(
      lambda,
      [c00, c01, c02, c11, c12, c22],
      (out) => () => {
        const {x, y, z} = eigenvector;
        // Symmetric matrix: ∂λ/∂C_ij = v_i * v_j
        // Off-diagonal elements need factor of 2 (appear in both C_ij and C_ji)
        c00.grad += out.grad * x * x;
        c01.grad += out.grad * 2 * x * y;
        c02.grad += out.grad * 2 * x * z;
        c11.grad += out.grad * y * y;
        c12.grad += out.grad * 2 * y * z;
        c22.grad += out.grad * z * z;
      },
      'eigenvalue_custom',
      'eigenvalue_custom'
    );
  }

  /**
   * Compute eigenvector for given eigenvalue of 3x3 symmetric matrix.
   * Uses null space method: find v such that (C - λI)v = 0
   *
   * @private
   */
  private static computeEigenvector(
    c00: number, c01: number, c02: number,
    c11: number, c12: number, c22: number,
    lambda: number
  ): {x: number, y: number, z: number} {
    // Matrix (C - λI)
    const a00 = c00 - lambda;
    const a11 = c11 - lambda;
    const a22 = c22 - lambda;

    // Find null space using cross product method
    // Take two rows and compute their cross product
    const row0 = {x: a00, y: c01, z: c02};
    const row1 = {x: c01, y: a11, z: c12};
    const row2 = {x: c02, y: c12, z: a22};

    // Try cross products of different row pairs to avoid degeneracy
    const candidates = [
      this.cross(row0, row1),
      this.cross(row0, row2),
      this.cross(row1, row2)
    ];

    // Pick the cross product with largest magnitude (most stable)
    let bestVec = candidates[0];
    let bestMag = this.magnitude(bestVec);

    for (const vec of candidates) {
      const mag = this.magnitude(vec);
      if (mag > bestMag) {
        bestVec = vec;
        bestMag = mag;
      }
    }

    // Normalize
    if (bestMag < 1e-12) {
      // Degenerate case: return arbitrary unit vector
      return {x: 1, y: 0, z: 0};
    }

    return {
      x: bestVec.x / bestMag,
      y: bestVec.y / bestMag,
      z: bestVec.z / bestMag
    };
  }

  private static cross(
    a: {x: number, y: number, z: number},
    b: {x: number, y: number, z: number}
  ): {x: number, y: number, z: number} {
    return {
      x: a.y * b.z - a.z * b.y,
      y: a.z * b.x - a.x * b.z,
      z: a.x * b.y - a.y * b.x
    };
  }

  private static magnitude(v: {x: number, y: number, z: number}): number {
    return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  }
}
