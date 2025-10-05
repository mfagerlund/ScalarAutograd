/**
 * Test that demonstrates the duplicate branch bug:
 * When the same computation branch is used multiple times,
 * gradients must be accumulated from ALL uses.
 */

import { V, Value, CompiledResiduals } from '../src';
import { testLog } from './testUtils';

describe('Duplicate Branch Bug', () => {
  it('should accumulate gradients when same branch used twice', () => {
    testLog('\n=== DUPLICATE BRANCH TEST ===\n');

    const params = [V.W(2.0), V.W(3.0)];

    let compilationGraph: Value | null = null;

    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      const x = p[0];
      const y = p[1];

      // Compute the same branch twice
      const branch = V.mul(x, y);  // x * y

      // Use the branch TWICE in the result
      const result = V.add(branch, branch);  // (x*y) + (x*y) = 2xy

      compilationGraph = result;
      return [result];
    });

    params.forEach(p => p.grad = 0);
    compilationGraph!.backward();
    const graphGrads = params.map(p => p.grad);

    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    testLog('Graph gradients:', graphGrads);
    testLog('Compiled gradients:', compiledGrads);

    // Analytical: f(x,y) = 2xy
    // df/dx = 2y = 2*3 = 6
    // df/dy = 2x = 2*2 = 4
    testLog('Expected gradients: [6, 4]');

    const maxDiff = Math.max(...params.map((_, i) => Math.abs(graphGrads[i] - compiledGrads[i])));
    testLog(`Max gradient diff: ${maxDiff.toExponential(6)}`);

    if (maxDiff < 1e-10) {
      testLog('✅ PASS');
    } else {
      testLog('❌ FAIL - Compiler only visited branch once!');
    }

    expect(maxDiff).toBeLessThan(1e-10);
  });

  it('should handle branch used in multiple residuals', () => {
    testLog('\n=== MULTIPLE RESIDUALS WITH SHARED BRANCH ===\n');

    const params = [V.W(2.0), V.W(3.0)];

    let compilationGraphs: Value[] = [];

    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      const x = p[0];
      const y = p[1];

      // Shared computation
      const product = V.mul(x, y);  // x * y

      // Use in TWO separate residuals
      const residual1 = V.mul(product, product);  // (xy)^2
      const residual2 = V.add(product, V.C(1));   // xy + 1

      compilationGraphs = [residual1, residual2];
      return [residual1, residual2];
    });

    // Create sum of residuals (same as compiled does)
    const sumResidual = V.add(compilationGraphs[0], compilationGraphs[1]);

    params.forEach(p => p.grad = 0);
    sumResidual.backward();
    const graphGrads = params.map(p => p.grad);

    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    testLog('Graph gradients:', graphGrads);
    testLog('Compiled gradients:', compiledGrads);

    // Analytical: f1(x,y) = (xy)^2, f2(x,y) = xy + 1
    // df1/dx = 2xy*y = 2*2*3*3 = 36
    // df2/dx = y = 3
    // Total df/dx = 36 + 3 = 39

    // df1/dy = 2xy*x = 2*2*2*3 = 24
    // df2/dy = x = 2
    // Total df/dy = 24 + 2 = 26
    testLog('Expected gradients: [39, 26]');

    const maxDiff = Math.max(...params.map((_, i) => Math.abs(graphGrads[i] - compiledGrads[i])));
    testLog(`Max gradient diff: ${maxDiff.toExponential(6)}`);

    if (maxDiff < 1e-10) {
      testLog('✅ PASS');
    } else {
      testLog('❌ FAIL - Shared branch only visited once!');
    }

    expect(maxDiff).toBeLessThan(1e-10);
  });
});
