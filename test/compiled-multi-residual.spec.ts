/**
 * Test compiled gradients for multiple residuals.
 * This tests the case where we compile multiple functions and accumulate their gradients.
 */

import { V } from "../src/V";
import { Value } from "../src/Value";
import { CompiledFunctions } from "../src/CompiledFunctions";
import { testLog } from './testUtils';

/**
 * Helper: Compare compiled multi-residual gradients with graph backward
 */
function validateMultiResidualGradients(
  buildResiduals: (params: Value[]) => Value[],
  paramValues: number[],
  testName: string,
  tolerance = 1e-10
) {
  testLog(`\n=== ${testName} ===`);

  // Create parameters
  const params = paramValues.map((val, i) => {
    const p = V.W(val, `p${i}`);
    p.paramName = `p${i}`;
    return p;
  });

  // Build residuals
  const residuals = buildResiduals(params);
  testLog(`Number of residuals: ${residuals.length}`);

  // Compute graph gradients (sum of all residuals)
  params.forEach(p => p.grad = 0);
  const sumResiduals = V.sum(residuals);
  sumResiduals.backward();
  const graphGradients = params.map(p => p.grad);
  const graphValue = sumResiduals.data;

  testLog(`Graph sum: ${graphValue}`);
  testLog(`Graph gradients: [${graphGradients.map(g => g.toExponential(6)).join(', ')}]`);

  // Compile and compute compiled gradients
  const compiled = CompiledFunctions.compile(params, (p) => buildResiduals(p));
  const { value: compiledValue, gradient: compiledGradients } = compiled.evaluateSumWithGradient(params);

  testLog(`Compiled sum: ${compiledValue}`);
  testLog(`Compiled gradients: [${compiledGradients.map(g => g.toExponential(6)).join(', ')}]`);

  // Also check individual residuals via Jacobian
  const { values: residualValues, jacobian } = compiled.evaluateJacobian(params);
  testLog(`\nIndividual residuals:`);
  for (let i = 0; i < residualValues.length; i++) {
    testLog(`  r[${i}] = ${residualValues[i].toExponential(6)}`);
  }

  // Validate sum matches
  expect(compiledValue).toBeCloseTo(graphValue, 10);

  // Validate gradients match
  for (let i = 0; i < params.length; i++) {
    const diff = Math.abs(compiledGradients[i] - graphGradients[i]);
    if (diff > tolerance) {
      testLog(`❌ GRADIENT MISMATCH at param ${i}:`);
      testLog(`   Graph:    ${graphGradients[i].toExponential(15)}`);
      testLog(`   Compiled: ${compiledGradients[i].toExponential(15)}`);
      testLog(`   Diff:     ${diff.toExponential(6)}`);

      // Check individual Jacobian rows
      testLog(`\n   Jacobian contributions:`);
      for (let j = 0; j < residuals.length; j++) {
        testLog(`     r[${j}]: ${jacobian[j][i].toExponential(6)}`);
      }
      const jacobianSum = jacobian.reduce((sum, row) => sum + row[i], 0);
      testLog(`   Jacobian sum: ${jacobianSum.toExponential(15)}`);
    }
    expect(compiledGradients[i]).toBeCloseTo(graphGradients[i], 10);
  }

  testLog('✓ Gradients match');
}

describe('Compiled Multi-Residual - Gradient Correctness', () => {
  it('two independent residuals', () => {
    validateMultiResidualGradients(
      ([a, b]) => [
        V.mul(a, V.C(2.0)),  // r0 = 2*a
        V.mul(b, V.C(3.0)),  // r1 = 3*b
      ],
      [4.0, 5.0],
      'Two independent residuals'
    );
  });

  it('two residuals sharing parameter', () => {
    validateMultiResidualGradients(
      ([a, b]) => [
        V.add(a, b),      // r0 = a + b
        V.mul(a, b),      // r1 = a * b
      ],
      [3.0, 4.0],
      'Two residuals sharing parameters'
    );
  });

  it('three residuals with complex expressions', () => {
    validateMultiResidualGradients(
      ([x, y, z]) => [
        V.add(V.square(x), V.C(1.0)),           // r0 = x^2 + 1
        V.sub(V.mul(y, V.C(2.0)), V.C(3.0)),    // r1 = 2*y - 3
        V.mul(V.sin(z), V.C(0.5)),              // r2 = 0.5*sin(z)
      ],
      [2.0, 3.0, Math.PI / 4],
      'Three residuals with complex expressions'
    );
  });

  it('residuals with shared subexpressions', () => {
    validateMultiResidualGradients(
      ([a, b]) => {
        const ab = V.mul(a, b);  // Shared subexpression
        return [
          V.add(ab, a),  // r0 = a*b + a
          V.sub(ab, b),  // r1 = a*b - b
        ];
      },
      [2.0, 3.0],
      'Residuals with shared subexpressions'
    );
  });

  it('distance residuals (like IK)', () => {
    validateMultiResidualGradients(
      ([x1, y1, x2, y2]) => {
        const dx = V.sub(x2, x1);
        const dy = V.sub(y2, y1);
        const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
        return [
          V.sub(dist, V.C(5.0)),  // r0 = distance - 5
          V.sub(x1, V.C(0.0)),    // r1 = x1 - 0 (anchor)
          V.sub(y1, V.C(0.0)),    // r2 = y1 - 0 (anchor)
        ];
      },
      [0.0, 0.0, 3.0, 4.0],
      'Distance residuals (IK-like)'
    );
  });

  it('many residuals (10)', () => {
    validateMultiResidualGradients(
      (params) => {
        const residuals: Value[] = [];
        for (let i = 0; i < params.length; i++) {
          residuals.push(V.square(V.sub(params[i], V.C(i + 1))));
        }
        return residuals;
      },
      [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
      'Many residuals (10)'
    );
  });

  it('normalized vector per-component residuals', () => {
    validateMultiResidualGradients(
      ([x, y, z]) => {
        const mag = V.sqrt(V.add(V.add(V.square(x), V.square(y)), V.square(z)));
        const nx = V.div(x, mag);
        const ny = V.div(y, mag);
        const nz = V.div(z, mag);

        // Target: (1,0,0)
        return [
          V.sub(nx, V.C(1.0)),
          V.sub(ny, V.C(0.0)),
          V.sub(nz, V.C(0.0)),
        ];
      },
      [1.0, 2.0, 2.0],
      'Normalized vector residuals'
    );
  });

  it('cross product components as residuals', () => {
    validateMultiResidualGradients(
      ([ax, ay, az, bx, by, bz]) => {
        // Cross product: a × b
        const cx = V.sub(V.mul(ay, bz), V.mul(az, by));
        const cy = V.sub(V.mul(az, bx), V.mul(ax, bz));
        const cz = V.sub(V.mul(ax, by), V.mul(ay, bx));

        return [cx, cy, cz];
      },
      [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      'Cross product components'
    );
  });

  it('mixed complexity residuals', () => {
    validateMultiResidualGradients(
      ([a, b, c]) => [
        V.add(a, V.C(1.0)),                      // Simple
        V.mul(V.sin(b), V.cos(b)),               // Moderate
        V.div(V.exp(c), V.add(V.C(1.0), V.exp(c))),  // Complex (sigmoid-like)
      ],
      [1.0, Math.PI / 4, 0.5],
      'Mixed complexity residuals'
    );
  });

  it('residuals with constants and parameters mixed', () => {
    validateMultiResidualGradients(
      ([p1, p2]) => {
        const c1 = V.C(2.0);
        const c2 = V.C(3.0);
        return [
          V.add(V.mul(p1, c1), V.mul(p2, c2)),  // r0 = 2*p1 + 3*p2
          V.mul(V.add(p1, c1), V.add(p2, c2)),  // r1 = (p1+2)*(p2+3)
        ];
      },
      [5.0, 7.0],
      'Residuals with constants mixed'
    );
  });
});
