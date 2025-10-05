/**
 * External helper functions for eigenvalue computation in compiled code.
 * These are called from generated JavaScript to avoid inlining huge formulas.
 */

/**
 * Compute smallest eigenvalue of 3x3 symmetric matrix.
 * Uses Cardano's formula for cubic equation.
 */
export function computeSmallestEigenvalue(
  c00: number, c01: number, c02: number,
  c11: number, c12: number, c22: number
): number {
  const tr = c00 + c11 + c22;
  const shift = tr / 3;
  const b00 = c00 - shift;
  const b11 = c11 - shift;
  const b22 = c22 - shift;
  const q = 0.5 * (b00*b00 + b11*b11 + b22*b22 + 2*c01*c01 + 2*c02*c02 + 2*c12*c12);
  const p = Math.sqrt((q + 1e-20) / 3);
  const invP = 1 / (p + 1e-12);
  const d00 = b00 * invP;
  const d01 = c01 * invP;
  const d02 = c02 * invP;
  const d11 = b11 * invP;
  const d12 = c12 * invP;
  const d22 = b22 * invP;
  const det = d00 * (d11*d22 - d12*d12) + d01 * (d12*d02 - d01*d22) + d02 * (d01*d12 - d11*d02);
  const r = Math.max(-0.99999, Math.min(det / 2, 0.99999));
  const theta = Math.acos(r);
  const angle2 = theta / 3 + 2 * 2.0943951023931953;
  return shift + 2 * p * Math.cos(angle2);
}

/**
 * Apply eigenvalue gradients using analytical formula.
 * Computes eigenvector and updates input gradients with ∂λ/∂C_ij = v_i * v_j
 */
export function applyEigenvalueGradients(
  outGrad: number,
  c00: number, c01: number, c02: number,
  c11: number, c12: number, c22: number,
  g00: number, g01: number, g02: number,
  g11: number, g12: number, g22: number
): [number, number, number, number, number, number] {
  const lambda = computeSmallestEigenvalue(c00, c01, c02, c11, c12, c22);

  const a00 = c00 - lambda;
  const a11 = c11 - lambda;
  const a22 = c22 - lambda;

  const row0_x = a00, row0_y = c01, row0_z = c02;
  const row1_x = c01, row1_y = a11, row1_z = c12;
  const row2_x = c02, row2_y = c12, row2_z = a22;

  const cross0_x = row0_y * row1_z - row0_z * row1_y;
  const cross0_y = row0_z * row1_x - row0_x * row1_z;
  const cross0_z = row0_x * row1_y - row0_y * row1_x;
  const cross1_x = row0_y * row2_z - row0_z * row2_y;
  const cross1_y = row0_z * row2_x - row0_x * row2_z;
  const cross1_z = row0_x * row2_y - row0_y * row2_x;
  const cross2_x = row1_y * row2_z - row1_z * row2_y;
  const cross2_y = row1_z * row2_x - row1_x * row2_z;
  const cross2_z = row1_x * row2_y - row1_y * row2_x;

  const mag0 = Math.sqrt(cross0_x*cross0_x + cross0_y*cross0_y + cross0_z*cross0_z);
  const mag1 = Math.sqrt(cross1_x*cross1_x + cross1_y*cross1_y + cross1_z*cross1_z);
  const mag2 = Math.sqrt(cross2_x*cross2_x + cross2_y*cross2_y + cross2_z*cross2_z);

  let bestMag = mag0;
  let vx = cross0_x, vy = cross0_y, vz = cross0_z;
  if (mag1 > bestMag) { bestMag = mag1; vx = cross1_x; vy = cross1_y; vz = cross1_z; }
  if (mag2 > bestMag) { bestMag = mag2; vx = cross2_x; vy = cross2_y; vz = cross2_z; }

  if (bestMag < 1e-12) { vx = 1; vy = 0; vz = 0; }
  else { vx /= bestMag; vy /= bestMag; vz /= bestMag; }

  return [
    g00 + outGrad * vx * vx,
    g01 + outGrad * 2 * vx * vy,
    g02 + outGrad * 2 * vx * vz,
    g11 + outGrad * vy * vy,
    g12 + outGrad * 2 * vy * vz,
    g22 + outGrad * vz * vz
  ];
}

// Make available globally for compiled code
if (typeof window !== 'undefined') {
  (window as any).__eigenHelpers = {
    computeSmallestEigenvalue,
    applyEigenvalueGradients
  };
}
