/**
 * Example: Symbolic Gradient Generation
 *
 * This example demonstrates how to use ScalarAutograd's symbolic differentiation
 * to generate analytical gradient formulas from mathematical expressions.
 */

import { parse, computeGradients, simplify, generateGradientCode, generateGradientFunction } from '../src/index';

// Example 1: Simple quadratic function
console.log('=== Example 1: Simple Quadratic ===\n');

const input1 = `
  output = x * x + y * y
`;

const program1 = parse(input1);
const gradients1 = computeGradients(program1, ['x', 'y']);

// Simplify expressions
const simplified1 = new Map();
for (const [param, expr] of gradients1.entries()) {
  simplified1.set(param, simplify(expr));
}

const code1 = generateGradientCode(program1, simplified1);
console.log(code1);

// Example 2: Trigonometric function
console.log('\n=== Example 2: Trigonometric Function ===\n');

const input2 = `
  z = x * x
  output = sin(z)
`;

const program2 = parse(input2);
const gradients2 = computeGradients(program2, ['x']);

const simplified2 = new Map();
for (const [param, expr] of gradients2.entries()) {
  simplified2.set(param, simplify(expr));
}

const code2 = generateGradientCode(program2, simplified2);
console.log(code2);

// Example 3: Generate as function
console.log('\n=== Example 3: Gradient Function ===\n');

const input3 = `
  output = x * x + 2 * x + 1
`;

const program3 = parse(input3);
const gradients3 = computeGradients(program3, ['x']);

const simplified3 = new Map();
for (const [param, expr] of gradients3.entries()) {
  simplified3.set(param, simplify(expr));
}

const func = generateGradientFunction(program3, simplified3, 'quadraticGradient', ['x']);
console.log(func);

// Example 4: Product rule
console.log('\n=== Example 4: Product Rule ===\n');

const input4 = `
  output = x * sin(x)
`;

const program4 = parse(input4);
const gradients4 = computeGradients(program4, ['x']);

const simplified4 = new Map();
for (const [param, expr] of gradients4.entries()) {
  simplified4.set(param, simplify(expr));
}

const code4 = generateGradientCode(program4, simplified4);
console.log(code4);

// Example 5: Compare with numerical autodiff
console.log('\n=== Example 5: Verification with Numerical Autodiff ===\n');

import { V } from '../src/index';

// Symbolic: d/dx(x^2) = 2x
const input5 = 'output = x * x';
const program5 = parse(input5);
const gradients5 = computeGradients(program5, ['x']);
const grad_symbolic = simplify(gradients5.get('x')!);

console.log('Symbolic gradient for x^2:');
console.log(`  ∂f/∂x = ${grad_symbolic.toString()}`);
console.log('');

// Numerical: compute gradient at x=3
const x = V.W(3);
const output = V.square(x);
output.backward();

console.log('Numerical verification at x=3:');
console.log(`  Symbolic formula: 2*x = 2*3 = 6`);
console.log(`  Numerical grad: ${x.grad}`);
console.log(`  Match: ${x.grad === 6 ? '✓' : '✗'}`);
