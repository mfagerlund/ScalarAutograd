/**
 * Test CompiledFunctions API: evaluateGradient() and evaluateJacobian()
 */

import { V } from "../src/V";
import { CompiledFunctions } from "../src/CompiledFunctions";

describe('CompiledFunctions API', () => {
  it('evaluateGradient() should work for single objective', () => {
    const x = V.W(3);
    const y = V.W(4);
    const params = [x, y];

    // Objective: f(x,y) = x^2 + 2*y^2
    const compiled = CompiledFunctions.compile(params, (p) => [
      V.add(V.square(p[0]), V.mul(V.C(2), V.square(p[1])))
    ]);

    const { value, gradient } = compiled.evaluateGradient(params);

    // f(3,4) = 9 + 2*16 = 41
    expect(value).toBeCloseTo(41);

    // ∂f/∂x = 2x = 6
    // ∂f/∂y = 4y = 16
    expect(gradient[0]).toBeCloseTo(6);
    expect(gradient[1]).toBeCloseTo(16);
  });

  it('evaluateJacobian() should work for multiple functions', () => {
    const x = V.W(2);
    const y = V.W(3);
    const params = [x, y];

    // Three functions:
    // f1 = x + y
    // f2 = x * y
    // f3 = x^2
    const compiled = CompiledFunctions.compile(params, (p) => [
      V.add(p[0], p[1]),
      V.mul(p[0], p[1]),
      V.square(p[0])
    ]);

    const { values, jacobian } = compiled.evaluateJacobian(params);

    // Values
    expect(values[0]).toBeCloseTo(5);  // 2+3
    expect(values[1]).toBeCloseTo(6);  // 2*3
    expect(values[2]).toBeCloseTo(4);  // 2^2

    // Jacobian
    // f1: ∂(x+y)/∂x=1, ∂(x+y)/∂y=1
    expect(jacobian[0][0]).toBeCloseTo(1);
    expect(jacobian[0][1]).toBeCloseTo(1);

    // f2: ∂(xy)/∂x=y=3, ∂(xy)/∂y=x=2
    expect(jacobian[1][0]).toBeCloseTo(3);
    expect(jacobian[1][1]).toBeCloseTo(2);

    // f3: ∂(x^2)/∂x=2x=4, ∂(x^2)/∂y=0
    expect(jacobian[2][0]).toBeCloseTo(4);
    expect(jacobian[2][1]).toBeCloseTo(0);
  });

  it('evaluate() backward compatibility should still work', () => {
    const x = V.W(1);
    const y = V.W(2);
    const params = [x, y];

    const compiled = CompiledFunctions.compile(params, (p) => [
      V.sub(p[0], V.C(5)),
      V.sub(p[1], V.C(3))
    ]);

    const { residuals, J, cost } = compiled.evaluate(params);

    expect(residuals[0]).toBeCloseTo(-4);  // 1-5
    expect(residuals[1]).toBeCloseTo(-1);  // 2-3
    expect(cost).toBeCloseTo(17);  // (-4)^2 + (-1)^2

    // Check Jacobian - each residual depends only on its own param
    // ∂(x-5)/∂x = 1, ∂(x-5)/∂y = 0
    // ∂(y-3)/∂x = 0, ∂(y-3)/∂y = 1
    expect(J[0][0]).toBeCloseTo(1);
    expect(J[0][1]).toBeCloseTo(0);
    expect(J[1][0]).toBeCloseTo(0);
    expect(J[1][1]).toBeCloseTo(1);
  });

  it('evaluateJacobian() should not compute cost (no squaring)', () => {
    const x = V.W(3);
    const params = [x];

    const compiled = CompiledFunctions.compile(params, (p) => [
      V.sub(p[0], V.C(1))
    ]);

    const { values } = compiled.evaluateJacobian(params);

    // Should return raw value (2), not squared (4)
    expect(values[0]).toBeCloseTo(2);
  });
});
