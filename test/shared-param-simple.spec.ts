/**
 * Simplest possible test of shared parameter bug
 */

import { V, Value, CompiledResiduals } from '../src';
import { testLog } from './testUtils';

describe('Shared Parameter Bug', () => {
  it('should handle parameter used twice in graph', () => {
    testLog('\n=== SHARED PARAMETER TEST ===\n');

    // Single parameter
    const params = [V.W(2.0), V.W(3.0)];

    let compilationGraph: Value | null = null;

    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      const x = p[0];  // Same value object
      const y = p[1];

      // Use x twice in different branches
      const branch1 = V.mul(x, y);     // x * y
      const branch2 = V.mul(x, x);     // x * x
      const result = V.add(branch1, branch2);  // x*y + x*x

      compilationGraph = result;
      return [result];
    });

    params.forEach(p => p.grad = 0);
    compilationGraph!.backward();
    const graphGrads = params.map(p => p.grad);

    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    testLog('Graph gradients:', graphGrads);
    testLog('Compiled gradients:', compiledGrads);

    const maxDiff = Math.max(...params.map((_, i) => Math.abs(graphGrads[i] - compiledGrads[i])));
    testLog(`Max gradient diff: ${maxDiff.toExponential(6)}`);

    // Analytical: f(x,y) = xy + x^2
    // df/dx = y + 2x = 3 + 2*2 = 7
    // df/dy = x = 2
    testLog(`Expected gradients: [7, 2]`);

    if (maxDiff < 1e-10) {
      testLog('✅ PASS');
    } else {
      testLog('❌ FAIL');
    }

    expect(maxDiff).toBeLessThan(1e-10);
  });
});
