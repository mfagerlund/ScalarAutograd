/**
 * Performs Cholesky decomposition of a positive definite matrix A = LL^T.
 * @param A - Positive definite matrix to decompose
 * @returns Lower triangular matrix L
 * @throws Error if matrix is not positive definite
 * @public
 */
export function choleskyDecomposition(A: number[][]): number[][] {
  const n = A.length;
  const L: number[][] = Array(n)
    .fill(0)
    .map(() => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += L[i][k] * L[j][k];
      }

      if (i === j) {
        const val = A[i][i] - sum;
        if (val <= 0) {
          throw new Error(
            `Matrix is not positive definite at diagonal element ${i}`
          );
        }
        L[i][j] = Math.sqrt(val);
      } else {
        L[i][j] = (A[i][j] - sum) / L[j][j];
      }
    }
  }

  return L;
}

/**
 * Solves Ly = b for y using forward substitution (L is lower triangular).
 * @param L - Lower triangular matrix
 * @param b - Right-hand side vector
 * @returns Solution vector y
 * @public
 */
export function forwardSubstitution(L: number[][], b: number[]): number[] {
  const n = L.length;
  const y = Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) {
      sum += L[i][j] * y[j];
    }
    y[i] = (b[i] - sum) / L[i][i];
  }

  return y;
}

/**
 * Solves L^T x = y for x using back substitution (L^T is upper triangular).
 * @param L - Lower triangular matrix (uses its transpose)
 * @param y - Right-hand side vector
 * @returns Solution vector x
 * @public
 */
export function backSubstitution(L: number[][], y: number[]): number[] {
  const n = L.length;
  const x = Array(n).fill(0);

  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) {
      sum += L[j][i] * x[j];
    }
    x[i] = (y[i] - sum) / L[i][i];
  }

  return x;
}

/**
 * Solves Ax = b using Cholesky decomposition where A is positive definite.
 * @param A - Positive definite matrix
 * @param b - Right-hand side vector
 * @returns Solution vector x
 * @public
 */
export function choleskySolve(A: number[][], b: number[]): number[] {
  const L = choleskyDecomposition(A);
  const y = forwardSubstitution(L, b);
  const x = backSubstitution(L, y);
  return x;
}

/**
 * Computes J^T J (Jacobian transpose times Jacobian).
 * Used in Gauss-Newton and Levenberg-Marquardt methods.
 * @param J - Jacobian matrix (m residuals × n parameters)
 * @returns J^T J matrix (n × n)
 * @public
 */
export function computeJtJ(J: number[][]): number[][] {
  const m = J.length;
  const n = J[0].length;
  const JtJ = Array(n)
    .fill(0)
    .map(() => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < m; k++) {
        sum += J[k][i] * J[k][j];
      }
      JtJ[i][j] = sum;
    }
  }

  return JtJ;
}

/**
 * Computes J^T r (Jacobian transpose times residual vector).
 * Used to compute the gradient in least squares problems.
 * @param J - Jacobian matrix (m residuals × n parameters)
 * @param r - Residual vector (m elements)
 * @returns J^T r vector (n elements)
 * @public
 */
export function computeJtr(J: number[][], r: number[]): number[] {
  const m = J.length;
  const n = J[0].length;
  const Jtr = Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let k = 0; k < m; k++) {
      sum += J[k][i] * r[k];
    }
    Jtr[i] = sum;
  }

  return Jtr;
}
