/**
 * Example: Gradient of θ = atan2(u×v, u·v)
 *
 * This computes the angle between two 2D vectors using atan2.
 * Shows how to compute symbolic gradients for vector operations.
 */

import { parse, computeGradients, simplify, generateGradientCode } from '../src/index';

console.log('=== Gradient of θ = atan2(u×v, u·v) ===\n');

// For 2D vectors u=(ux, uy) and v=(vx, vy):
// Cross product (scalar): u×v = ux*vy - uy*vx
// Dot product: u·v = ux*vx + uy*vy
// Angle: θ = atan2(u×v, u·v)

const input = `
  cross = ux * vy - uy * vx
  dot = ux * vx + uy * vy
  output = atan2(cross, dot)
`;

const program = parse(input);
const gradients = computeGradients(program, ['ux', 'uy', 'vx', 'vy']);

// Simplify all gradients
const simplified = new Map();
for (const [param, expr] of gradients.entries()) {
  simplified.set(param, simplify(expr));
}

const code = generateGradientCode(program, simplified, {
  includeMath: true,
  includeForward: true
});

console.log(code);

console.log('\n=== Verification ===\n');

// Test with specific values: u = (1, 0), v = (0, 1)
// This gives θ = atan2(1, 0) = π/2
console.log('Test vectors:');
console.log('  u = (1, 0)');
console.log('  v = (0, 1)');
console.log('  cross = 1*1 - 0*0 = 1');
console.log('  dot = 1*0 + 0*1 = 0');
console.log('  θ = atan2(1, 0) = π/2 ≈ 1.571 radians');

console.log('\n=== Mathematical Interpretation ===\n');
console.log('The gradient ∂θ/∂u tells us how the angle changes when we move vector u.');
console.log('The gradient ∂θ/∂v tells us how the angle changes when we move vector v.');
console.log('');
console.log('This is useful for:');
console.log('  - Robotics: optimizing joint angles');
console.log('  - Computer vision: aligning orientations');
console.log('  - Physics: angular momentum calculations');
