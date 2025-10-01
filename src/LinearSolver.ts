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

/**
 * Computes QR decomposition of matrix A using Householder reflections.
 * @param A - Input matrix (m × n)
 * @returns Object with Q (m × m) and R (m × n) matrices
 * @public
 */
export function qrDecomposition(A: number[][]): { Q: number[][]; R: number[][] } {
  const m = A.length;
  const n = A[0].length;

  const R = A.map(row => [...row]);
  const Q = Array(m).fill(0).map((_, i) =>
    Array(m).fill(0).map((_, j) => i === j ? 1 : 0)
  );

  for (let k = 0; k < Math.min(m - 1, n); k++) {
    let norm = 0;
    for (let i = k; i < m; i++) {
      norm += R[i][k] * R[i][k];
    }
    norm = Math.sqrt(norm);

    if (norm < 1e-14) continue;

    const s = R[k][k] >= 0 ? -1 : 1;
    const u1 = R[k][k] - s * norm;
    const v = Array(m).fill(0);
    v[k] = 1;
    for (let i = k + 1; i < m; i++) {
      v[i] = R[i][k] / u1;
    }

    const tau = -s * u1 / norm;

    for (let j = k; j < n; j++) {
      let sum = R[k][j];
      for (let i = k + 1; i < m; i++) {
        sum += v[i] * R[i][j];
      }
      sum *= tau;

      R[k][j] -= sum;
      for (let i = k + 1; i < m; i++) {
        R[i][j] -= sum * v[i];
      }
    }

    for (let j = 0; j < m; j++) {
      let sum = Q[k][j];
      for (let i = k + 1; i < m; i++) {
        sum += v[i] * Q[i][j];
      }
      sum *= tau;

      Q[k][j] -= sum;
      for (let i = k + 1; i < m; i++) {
        Q[i][j] -= sum * v[i];
      }
    }
  }

  return { Q, R };
}

/**
 * Solves least squares problem min ||Ax - b|| using QR decomposition.
 * Handles rank-deficient matrices by truncating small singular values.
 * @param A - Coefficient matrix (m × n)
 * @param b - Right-hand side vector (m elements)
 * @param epsilon - Singular value threshold (default 1e-10)
 * @returns Solution vector x (n elements)
 * @public
 */
export function qrSolve(A: number[][], b: number[], epsilon = 1e-10): number[] {
  const m = A.length;
  const n = A[0].length;

  const { Q, R } = qrDecomposition(A);

  const Qtb = Array(m).fill(0);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      Qtb[i] += Q[j][i] * b[j];
    }
  }

  const x = Array(n).fill(0);
  for (let i = Math.min(m, n) - 1; i >= 0; i--) {
    if (Math.abs(R[i][i]) < epsilon) {
      x[i] = 0;
      continue;
    }

    let sum = Qtb[i];
    for (let j = i + 1; j < n; j++) {
      sum -= R[i][j] * x[j];
    }
    x[i] = sum / R[i][i];
  }

  return x;
}
