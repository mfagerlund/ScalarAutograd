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

    // For smallest eigenvalue: use θ/3 + 4π/3
    const twoThirdsPI = V.C(2.0943951023931953); // 2π/3
    const angle2 = V.add(V.div(theta, 3), V.mul(2, twoThirdsPI));
    const eig2 = V.add(shift, V.mul(V.mul(2, p), V.cos(angle2)));

    return eig2;
  }
}
