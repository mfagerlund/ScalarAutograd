/**
 * Comprehensive test suite for all operators: compiled vs graph backward.
 * Tests that every operator produces identical gradients in compiled and uncompiled mode.
 */

import { V } from "../src/V";
import { Value } from "../src/Value";
import { CompiledFunctions } from "../src/CompiledFunctions";
import { testLog } from './testUtils';

/**
 * Helper: Compare compiled kernel gradient with graph backward gradient
 */
function validateOperatorGradients(
  buildExpr: (params: Value[]) => Value,
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

  // Build expression
  const expr = buildExpr(params);

  // Compute graph gradients
  params.forEach(p => p.grad = 0);
  expr.backward();
  const graphGradients = params.map(p => p.grad);
  const graphValue = expr.data;

  testLog(`Graph value: ${graphValue}`);
  testLog(`Graph gradients: [${graphGradients.map(g => g.toExponential(6)).join(', ')}]`);

  // Compile and compute compiled gradients
  const compiled = CompiledFunctions.compile(params, (p) => [buildExpr(p)]);
  const { value: compiledValue, gradient: compiledGradients } = compiled.evaluateGradient(params);

  testLog(`Compiled value: ${compiledValue}`);
  testLog(`Compiled gradients: [${compiledGradients.map(g => g.toExponential(6)).join(', ')}]`);

  // Validate value matches
  expect(compiledValue).toBeCloseTo(graphValue, 10);

  // Validate gradients match
  for (let i = 0; i < params.length; i++) {
    const diff = Math.abs(compiledGradients[i] - graphGradients[i]);
    if (diff > tolerance) {
      testLog(`❌ GRADIENT MISMATCH at param ${i}:`);
      testLog(`   Graph:    ${graphGradients[i].toExponential(15)}`);
      testLog(`   Compiled: ${compiledGradients[i].toExponential(15)}`);
      testLog(`   Diff:     ${diff.toExponential(6)}`);
    }
    expect(compiledGradients[i]).toBeCloseTo(graphGradients[i], 10);
  }

  testLog('✓ Gradients match');
}

describe('Compiled Operators - Gradient Correctness', () => {
  describe('Arithmetic Operations', () => {
    it('add', () => {
      validateOperatorGradients(
        ([a, b]) => V.add(a, b),
        [3.0, 5.0],
        'add(a, b)'
      );
    });

    it('sub', () => {
      validateOperatorGradients(
        ([a, b]) => V.sub(a, b),
        [7.0, 3.0],
        'sub(a, b)'
      );
    });

    it('mul', () => {
      validateOperatorGradients(
        ([a, b]) => V.mul(a, b),
        [4.0, 6.0],
        'mul(a, b)'
      );
    });

    it('div', () => {
      validateOperatorGradients(
        ([a, b]) => V.div(a, b),
        [12.0, 4.0],
        'div(a, b)'
      );
    });

    it('pow (numeric exponent)', () => {
      validateOperatorGradients(
        ([a]) => V.pow(a, 3),
        [2.0],
        'pow(a, 3)'
      );
    });

    it('powValue', () => {
      validateOperatorGradients(
        ([a, b]) => V.powValue(a, b),
        [2.0, 3.0],
        'powValue(a, b)'
      );
    });

    it('abs (positive)', () => {
      validateOperatorGradients(
        ([a]) => V.abs(a),
        [5.0],
        'abs(positive)'
      );
    });

    it('abs (negative)', () => {
      validateOperatorGradients(
        ([a]) => V.abs(a),
        [-5.0],
        'abs(negative)'
      );
    });

    it('square', () => {
      validateOperatorGradients(
        ([a]) => V.square(a),
        [4.0],
        'square(a)'
      );
    });

    it('sqrt', () => {
      validateOperatorGradients(
        ([a]) => V.sqrt(a),
        [9.0],
        'sqrt(a)'
      );
    });
  });

  describe('Transcendental Functions', () => {
    it('exp', () => {
      validateOperatorGradients(
        ([a]) => V.exp(a),
        [1.5],
        'exp(a)'
      );
    });

    it('log', () => {
      validateOperatorGradients(
        ([a]) => V.log(a),
        [5.0],
        'log(a)'
      );
    });
  });

  describe('Trigonometric Functions', () => {
    it('sin', () => {
      validateOperatorGradients(
        ([a]) => V.sin(a),
        [Math.PI / 4],
        'sin(a)'
      );
    });

    it('cos', () => {
      validateOperatorGradients(
        ([a]) => V.cos(a),
        [Math.PI / 3],
        'cos(a)'
      );
    });

    it('tan', () => {
      validateOperatorGradients(
        ([a]) => V.tan(a),
        [Math.PI / 6],
        'tan(a)'
      );
    });

    it('asin', () => {
      validateOperatorGradients(
        ([a]) => V.asin(a),
        [0.5],
        'asin(a)'
      );
    });

    it('acos', () => {
      validateOperatorGradients(
        ([a]) => V.acos(a),
        [0.5],
        'acos(a)'
      );
    });

    it('atan', () => {
      validateOperatorGradients(
        ([a]) => V.atan(a),
        [1.0],
        'atan(a)'
      );
    });

    // it('atan2', () => {
    //   validateOperatorGradients(
    //     ([y, x]) => V.atan2(y, x),
    //     [3.0, 4.0],
    //     'atan2(y, x)'
    //   );
    // });
  });

  describe('Activation Functions', () => {
    it('relu (positive)', () => {
      validateOperatorGradients(
        ([a]) => V.relu(a),
        [5.0],
        'relu(positive)'
      );
    });

    it('relu (negative)', () => {
      validateOperatorGradients(
        ([a]) => V.relu(a),
        [-5.0],
        'relu(negative)'
      );
    });

    it('sigmoid', () => {
      validateOperatorGradients(
        ([a]) => V.sigmoid(a),
        [2.0],
        'sigmoid(a)'
      );
    });

    it('tanh', () => {
      validateOperatorGradients(
        ([a]) => V.tanh(a),
        [1.5],
        'tanh(a)'
      );
    });

    it('softplus', () => {
      validateOperatorGradients(
        ([a]) => V.softplus(a),
        [2.0],
        'softplus(a)'
      );
    });
  });

  describe('Comparison Operations', () => {
    it('min', () => {
      validateOperatorGradients(
        ([a, b]) => V.min(a, b),
        [3.0, 5.0],
        'min(a, b) - a smaller'
      );
    });

    it('min (reversed)', () => {
      validateOperatorGradients(
        ([a, b]) => V.min(a, b),
        [7.0, 4.0],
        'min(a, b) - b smaller'
      );
    });

    it('max', () => {
      validateOperatorGradients(
        ([a, b]) => V.max(a, b),
        [3.0, 5.0],
        'max(a, b) - b larger'
      );
    });

    it('max (reversed)', () => {
      validateOperatorGradients(
        ([a, b]) => V.max(a, b),
        [7.0, 4.0],
        'max(a, b) - a larger'
      );
    });
  });

  describe('Composite Expressions', () => {
    it('(a + b) * c', () => {
      validateOperatorGradients(
        ([a, b, c]) => V.mul(V.add(a, b), c),
        [2.0, 3.0, 4.0],
        '(a+b)*c'
      );
    });

    it('a * exp(b * x) - y', () => {
      validateOperatorGradients(
        ([a, b]) => V.sub(V.mul(a, V.exp(V.mul(b, V.C(3.0)))), V.C(10.0)),
        [2.0, 0.5],
        'a*exp(b*x) - y'
      );
    });

    it('sqrt((x2-x1)^2 + (y2-y1)^2)', () => {
      validateOperatorGradients(
        ([x1, y1, x2, y2]) => {
          const dx = V.sub(x2, x1);
          const dy = V.sub(y2, y1);
          return V.sqrt(V.add(V.square(dx), V.square(dy)));
        },
        [0.0, 0.0, 3.0, 4.0],
        'Euclidean distance'
      );
    });

    it('sin(a) * cos(b) + exp(c)', () => {
      validateOperatorGradients(
        ([a, b, c]) => V.add(V.mul(V.sin(a), V.cos(b)), V.exp(c)),
        [Math.PI / 4, Math.PI / 3, 0.5],
        'sin(a)*cos(b) + exp(c)'
      );
    });

    it('normalized vector', () => {
      validateOperatorGradients(
        ([x, y, z]) => {
          const mag = V.sqrt(V.add(V.add(V.square(x), V.square(y)), V.square(z)));
          return V.div(x, mag); // Just test x component
        },
        [1.0, 2.0, 2.0],
        'normalized vector (x component)'
      );
    });

    it('dot product with cross product', () => {
      validateOperatorGradients(
        ([ax, ay, az, bx, by, bz]) => {
          // Cross product z-component: ax*by - ay*bx
          const crossZ = V.sub(V.mul(ax, by), V.mul(ay, bx));
          // Dot product: ax*bx + ay*by + az*bz
          const dot = V.add(V.add(V.mul(ax, bx), V.mul(ay, by)), V.mul(az, bz));
          // Combine
          return V.add(crossZ, dot);
        },
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'cross and dot combination'
      );
    });
  });

  describe('Edge Cases', () => {
    it('very small values', () => {
      validateOperatorGradients(
        ([a, b]) => V.add(V.mul(a, a), V.mul(b, b)),
        [1e-8, 1e-8],
        'very small values'
      );
    });

    it('mixed scales', () => {
      validateOperatorGradients(
        ([a, b]) => V.add(V.mul(a, V.C(1e6)), V.mul(b, V.C(1e-6))),
        [1e-3, 1e3],
        'mixed scales'
      );
    });

    it('deep nesting', () => {
      validateOperatorGradients(
        ([a]) => {
          let result = a;
          for (let i = 0; i < 10; i++) {
            result = V.add(result, V.mul(V.C(0.1), result));
          }
          return result;
        },
        [2.0],
        'deep nesting (10 levels)'
      );
    });
  });
});
