/**
 * DEBUG: Eigenvalue gradient computation
 *
 * Test if Matrix3x3.smallestEigenvalueCustomGrad is computing correct gradients
 */

import { V, Value, Matrix3x3 } from 'scalar-autograd';

console.log('Testing eigenvalue gradient computation...');
console.log('');

console.log('STEP 1: Create simple test matrix');
const c00 = V.W(1.0);
const c01 = V.W(0.2);
const c02 = V.W(0.1);
const c11 = V.W(0.8);
const c12 = V.W(0.15);
const c22 = V.W(0.6);

console.log('Matrix:');
console.log(`  [${c00.data.toFixed(3)}, ${c01.data.toFixed(3)}, ${c02.data.toFixed(3)}]`);
console.log(`  [${c01.data.toFixed(3)}, ${c11.data.toFixed(3)}, ${c12.data.toFixed(3)}]`);
console.log(`  [${c02.data.toFixed(3)}, ${c12.data.toFixed(3)}, ${c22.data.toFixed(3)}]`);
console.log('');

console.log('STEP 2: Compute eigenvalue with custom gradients');
const lambda = Matrix3x3.smallestEigenvalueCustomGrad(c00, c01, c02, c11, c12, c22);
console.log(`  Smallest eigenvalue: ${lambda.data.toExponential(6)}`);
console.log('');

console.log('STEP 3: Backward pass');
lambda.backward();
console.log(`  ∂λ/∂c00 = ${c00.grad.toExponential(6)}`);
console.log(`  ∂λ/∂c01 = ${c01.grad.toExponential(6)}`);
console.log(`  ∂λ/∂c02 = ${c02.grad.toExponential(6)}`);
console.log(`  ∂λ/∂c11 = ${c11.grad.toExponential(6)}`);
console.log(`  ∂λ/∂c12 = ${c12.grad.toExponential(6)}`);
console.log(`  ∂λ/∂c22 = ${c22.grad.toExponential(6)}`);
console.log('');

const gradSum = Math.abs(c00.grad) + Math.abs(c01.grad) + Math.abs(c02.grad) +
                Math.abs(c11.grad) + Math.abs(c12.grad) + Math.abs(c22.grad);

if (gradSum === 0) {
  console.log('❌ ERROR: All gradients are zero!');
  console.log('The custom gradient implementation is not working.');
} else {
  console.log('✓ Gradients computed successfully');
}
console.log('');

console.log('STEP 4: Test gradient through max(lambda, 0)');
Value.resetAllGrads();

const c00_2 = V.W(1.0);
const c01_2 = V.W(0.2);
const c02_2 = V.W(0.1);
const c11_2 = V.W(0.8);
const c12_2 = V.W(0.15);
const c22_2 = V.W(0.6);

const lambda2 = Matrix3x3.smallestEigenvalueCustomGrad(c00_2, c01_2, c02_2, c11_2, c12_2, c22_2);
const clampedLambda = V.max(lambda2, V.C(0));

console.log(`  lambda = ${lambda2.data.toExponential(6)}`);
console.log(`  max(lambda, 0) = ${clampedLambda.data.toExponential(6)}`);

clampedLambda.backward();

console.log(`  ∂max/∂c00 = ${c00_2.grad.toExponential(6)}`);
console.log(`  ∂max/∂c01 = ${c01_2.grad.toExponential(6)}`);

const gradSum2 = Math.abs(c00_2.grad) + Math.abs(c01_2.grad) + Math.abs(c02_2.grad) +
                 Math.abs(c11_2.grad) + Math.abs(c12_2.grad) + Math.abs(c22_2.grad);

if (gradSum2 === 0) {
  console.log('❌ ERROR: Gradients lost through max() operation!');
} else {
  console.log('✓ Gradients propagate through max()');
}
