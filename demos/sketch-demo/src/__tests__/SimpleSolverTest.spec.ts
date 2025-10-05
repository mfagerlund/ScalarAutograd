/**
 * Simple debug test for solver
 */

import { describe, it, expect } from 'vitest';
import { Value, V } from '../../../../src';
import { nonlinearLeastSquares } from '../../../../src/NonlinearLeastSquares';
import { testLog } from '../../../../test/testUtils';

describe('Simple Solver Test', () => {
  it('should solve x^2 = 4', () => {
    const x = V.W(1); // Start at 1

    const residualFn = (vars: Value[]) => {
      const [xVar] = vars;
      // Residual: x^2 - 4 should be 0
      const residual = V.sub(V.mul(xVar, xVar), V.C(4));
      return [residual];
    };

    const result = nonlinearLeastSquares([x], residualFn, {
      costTolerance: 1e-6,
      maxIterations: 100,
    });

    testLog('Simple test result:', result);
    testLog('x =', x.data, 'should be ±2');

    expect(result.success).toBe(true);
    expect(Math.abs(x.data)).toBeCloseTo(2, 4);
  });
});
