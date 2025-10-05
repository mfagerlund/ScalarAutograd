/**
 * Test indirect kernel compilation correctness.
 * Validates that compiled kernels produce identical results to graph backward pass.
 */

import { V } from "../../src/V";
import { ValueRegistry } from "../../src/ValueRegistry";
import { compileIndirectKernel, extractInputIndices } from "../../src/compileIndirectKernel";
import { Value } from "../../src/Value";
import { testLog } from '../testUtils';

/**
 * Helper: Compare compiled kernel Jacobian with graph backward gradients
 */
function validateKernelMatchesGraph(
  residual: Value,
  params: Value[],
  testName: string
) {
  testLog(`\n=== ${testName} ===`);

  // Setup registry and compile kernel
  const registry = new ValueRegistry();
  params.forEach(p => registry.register(p));

  const kernel = compileIndirectKernel(residual, params, registry);
  const indices = extractInputIndices(residual, registry);

  testLog(`Graph inputs: ${indices.length}`);
  testLog(`Parameters: ${params.length}`);

  // Build gradient indices mapping
  const paramIds = params.map(p => registry.getId(p));
  const gradientIndices = indices.map(idx => {
    const paramIndex = paramIds.indexOf(idx);
    return paramIndex >= 0 ? paramIndex : -1;
  });

  // Get Jacobian row from compiled kernel
  const allValues = registry.getDataArray();
  const jacobianRow = new Array(params.length).fill(0);
  const kernelValue = kernel(allValues, indices, gradientIndices, jacobianRow);

  testLog(`Kernel value: ${kernelValue}`);
  testLog(`Kernel Jacobian: [${jacobianRow.join(', ')}]`);

  // Get gradients from graph backward
  Value.zeroGradTree(residual);
  params.forEach(p => p.grad = 0);
  residual.backward();

  const graphGradients = params.map(p => p.grad);
  testLog(`Graph gradients: [${graphGradients.join(', ')}]`);

  // Validate
  expect(kernelValue).toBeCloseTo(residual.data, 10);

  for (let i = 0; i < params.length; i++) {
    expect(jacobianRow[i]).toBeCloseTo(graphGradients[i], 10);
  }

  testLog('✓ Kernel matches graph');
}

describe('Indirect Kernel Correctness', () => {
  it('should match graph backward for simple addition', () => {
    const a = V.W(3.0, 'a');
    const b = V.W(5.0, 'b');
    a.paramName = 'a';
    b.paramName = 'b';

    const residual = V.add(a, b);

    validateKernelMatchesGraph(residual, [a, b], 'Simple Addition');
  });

  it('should match graph backward for multiplication', () => {
    const x = V.W(2.0, 'x');
    const y = V.W(4.0, 'y');
    x.paramName = 'x';
    y.paramName = 'y';

    const residual = V.mul(x, y);

    validateKernelMatchesGraph(residual, [x, y], 'Multiplication');
  });

  it('should match graph backward for composite expression', () => {
    const a = V.W(2.0, 'a');
    const b = V.W(3.0, 'b');
    const c = V.W(1.0, 'c');
    a.paramName = 'a';
    b.paramName = 'b';
    c.paramName = 'c';

    // (a + b) * c
    const residual = V.mul(V.add(a, b), c);

    validateKernelMatchesGraph(residual, [a, b, c], '(a+b)*c');
  });

  it('should match graph backward for exponential', () => {
    const a = V.W(2.0, 'a');
    const b = V.W(0.5, 'b');
    a.paramName = 'a';
    b.paramName = 'b';

    // a * exp(b)
    const residual = V.mul(a, V.exp(b));

    validateKernelMatchesGraph(residual, [a, b], 'a*exp(b)');
  });

  it('should match graph backward for trig functions', () => {
    const theta = V.W(Math.PI / 4, 'theta');
    theta.paramName = 'theta';

    const residual = V.sin(theta);

    validateKernelMatchesGraph(residual, [theta], 'sin(theta)');
  });

  it('should match graph backward for curve fitting residual', () => {
    const a = V.W(2.0, 'a');
    const b = V.W(0.5, 'b');
    a.paramName = 'a';
    b.paramName = 'b';

    const x = 3.0;
    const yTarget = 10.0;

    // Residual: a * exp(b * x) - yTarget
    const pred = V.mul(a, V.exp(V.mul(b, V.C(x))));
    const residual = V.sub(pred, V.C(yTarget));

    validateKernelMatchesGraph(residual, [a, b], 'Curve Fitting Residual');
  });

  it('should match graph backward for distance constraint', () => {
    const x1 = V.W(0.0, 'x1');
    const y1 = V.W(0.0, 'y1');
    const x2 = V.W(3.0, 'x2');
    const y2 = V.W(4.0, 'y2');
    [x1, y1, x2, y2].forEach(v => v.paramName = v.label);

    const dx = V.sub(x2, x1);
    const dy = V.sub(y2, y1);
    const distSq = V.add(V.mul(dx, dx), V.mul(dy, dy));
    const dist = V.sqrt(distSq);
    const residual = V.sub(dist, V.C(5.0)); // Target distance = 5

    validateKernelMatchesGraph(
      residual,
      [x1, y1, x2, y2],
      'Distance Constraint'
    );
  });

  it('should handle constants correctly', () => {
    const x = V.W(3.0, 'x');
    x.paramName = 'x';

    const residual = V.sub(V.mul(x, V.C(2.0)), V.C(5.0)); // 2*x - 5

    validateKernelMatchesGraph(residual, [x], 'Constants');
  });

  it('should handle unused parameters', () => {
    const a = V.W(3.0, 'a');
    const b = V.W(5.0, 'b');
    a.paramName = 'a';
    b.paramName = 'b';

    // Residual only uses 'a'
    const residual = V.mul(a, V.C(2.0));

    validateKernelMatchesGraph(residual, [a, b], 'Unused Parameter');
    // Should have zero gradient for b
  });

  it('should match for complex nested expression', () => {
    const w1 = V.W(1.5, 'w1');
    const w2 = V.W(2.5, 'w2');
    const w3 = V.W(0.5, 'w3');
    [w1, w2, w3].forEach(v => v.paramName = v.label);

    // ((w1 + w2) * w3) - (w1 * w2)
    const left = V.mul(V.add(w1, w2), w3);
    const right = V.mul(w1, w2);
    const residual = V.sub(left, right);

    validateKernelMatchesGraph(residual, [w1, w2, w3], 'Complex Nested');
  });
});
