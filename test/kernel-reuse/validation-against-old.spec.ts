/**
 * Validate new kernel reuse implementation against old direct compilation
 */

import { V } from "../../src/V";
import { CompiledResiduals } from "../../src/CompiledResiduals";
import { compileResidualJacobian } from "../../src/jit-compile-value";

describe('Kernel Reuse Validation', () => {
  it('should produce identical results to old implementation', () => {
    const x1 = V.W(0, 'x1');
    const y1 = V.W(0, 'y1');
    const x2 = V.W(3, 'x2');
    const y2 = V.W(4, 'y2');
    const params = [x1, y1, x2, y2];

    const residualFn = (p: V.Value[]) => {
      const dx = V.sub(p[2], p[0]);
      const dy = V.sub(p[3], p[1]);
      const distSq = V.add(V.mul(dx, dx), V.mul(dy, dy));
      const dist = V.sqrt(distSq);
      return [V.sub(dist, V.C(5))];
    };

    // New implementation (kernel reuse)
    const compiledNew = CompiledResiduals.compile(params, residualFn);
    const resultNew = compiledNew.evaluate(params);

    // Old implementation (direct)
    params.forEach((p, i) => { if (!p.paramName) p.paramName = `p${i}`; });
    const residualValues = residualFn(params);
    const compiledOld = residualValues.map((r, i) =>
      compileResidualJacobian(r, params, i)
    );

    const paramValues = params.map(p => p.data);
    const JOld = [[0, 0, 0, 0]];
    const valueOld = compiledOld[0](paramValues, JOld[0]);

    console.log('\nOld implementation:');
    console.log('  Value:', valueOld);
    console.log('  Jacobian:', JOld[0]);

    console.log('\nNew implementation (kernel reuse):');
    console.log('  Value:', resultNew.residuals[0]);
    console.log('  Jacobian:', resultNew.J[0]);
    console.log('  Kernels compiled:', compiledNew.kernelCount);

    // Results must be identical
    expect(resultNew.residuals[0]).toBeCloseTo(valueOld, 12);
    expect(resultNew.J[0][0]).toBeCloseTo(JOld[0][0], 12);
    expect(resultNew.J[0][1]).toBeCloseTo(JOld[0][1], 12);
    expect(resultNew.J[0][2]).toBeCloseTo(JOld[0][2], 12);
    expect(resultNew.J[0][3]).toBeCloseTo(JOld[0][3], 12);
  });

  it('should match old implementation with multiple residuals', () => {
    const a = V.W(1, 'a');
    const b = V.W(2, 'b');
    const c = V.W(3, 'c');
    const params = [a, b, c];

    const residualFn = (p: V.Value[]) => [
      V.add(p[0], p[1]),
      V.mul(p[1], p[2]),
      V.sub(V.mul(p[0], p[2]), V.C(5))
    ];

    // New
    const compiledNew = CompiledResiduals.compile(params, residualFn);
    const resultNew = compiledNew.evaluate(params);

    // Old
    params.forEach((p, i) => { if (!p.paramName) p.paramName = `p${i}`; });
    const residualValues = residualFn(params);
    const compiledOld = residualValues.map((r, i) =>
      compileResidualJacobian(r, params, i)
    );

    const paramValues = params.map(p => p.data);
    const JOld = Array(3).fill(0).map(() => [0, 0, 0]);
    const residualsOld = compiledOld.map((fn, i) => fn(paramValues, JOld[i]));

    console.log('\nMultiple residuals comparison:');
    console.log('Old:', residualsOld);
    console.log('New:', resultNew.residuals);
    console.log('Kernel reuse factor:', compiledNew.kernelReuseFactor.toFixed(1) + 'x');

    // All residuals must match
    for (let i = 0; i < 3; i++) {
      expect(resultNew.residuals[i]).toBeCloseTo(residualsOld[i], 12);
      for (let j = 0; j < 3; j++) {
        expect(resultNew.J[i][j]).toBeCloseTo(JOld[i][j], 12);
      }
    }
  });

  it('should match with 50 identical residuals', () => {
    const numResiduals = 50;
    const params: V.Value[] = [];
    for (let i = 0; i < numResiduals * 2; i++) {
      params.push(V.W(Math.random(), `p${i}`));
    }

    const residualFn = (p: V.Value[]) => {
      const res: V.Value[] = [];
      for (let i = 0; i < numResiduals; i++) {
        res.push(V.add(p[i * 2], p[i * 2 + 1]));
      }
      return res;
    };

    // New
    const compiledNew = CompiledResiduals.compile(params, residualFn);
    const resultNew = compiledNew.evaluate(params);

    // Old
    params.forEach((p, i) => { if (!p.paramName) p.paramName = `p${i}`; });
    const residualValues = residualFn(params);
    const compiledOld = residualValues.map((r, i) =>
      compileResidualJacobian(r, params, i)
    );

    const paramValues = params.map(p => p.data);
    const JOld = Array(numResiduals).fill(0).map(() => Array(numResiduals * 2).fill(0));
    const residualsOld = compiledOld.map((fn, i) => fn(paramValues, JOld[i]));

    console.log('\n50 identical residuals:');
    console.log('  Old kernels:', numResiduals);
    console.log('  New kernels:', compiledNew.kernelCount);
    console.log('  Kernel savings:', (100 * (1 - compiledNew.kernelCount / numResiduals)).toFixed(1) + '%');

    // All must match
    for (let i = 0; i < numResiduals; i++) {
      expect(resultNew.residuals[i]).toBeCloseTo(residualsOld[i], 12);
    }
  });
});
