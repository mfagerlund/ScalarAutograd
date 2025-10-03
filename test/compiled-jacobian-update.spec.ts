/**
 * Test new in-place Jacobian update API
 */

import { V } from "../src/V";
import { compileResidualJacobian } from "../src/jit-compile-value";

describe("Compiled Jacobian In-Place Update", () => {
  it("should update Jacobian matrix in-place", () => {
    const x = V.W(0, "x");
    x.paramName = "x";

    const residual = V.sub(x, V.C(5));

    // Compile with row index 0
    const compiled = compileResidualJacobian(residual, [x], 0);

    console.log("\n=== Generated function ===");
    console.log(compiled.toString());

    // Create Jacobian matrix
    const J = [[0]];
    const paramValues = [3];

    // Call compiled function
    const value = compiled(paramValues, J);

    console.log("\nResult:");
    console.log("  value:", value);
    console.log("  J:", J);

    expect(value).toBe(-2);
    expect(J[0][0]).toBe(1);
  });

  it("should work with multiple parameters", () => {
    const a = V.W(2.0, "a");
    const b = V.W(0.5, "b");
    a.paramName = "a";
    b.paramName = "b";

    const x = 3.0;
    const pred = V.mul(a, V.exp(V.mul(b, V.C(x))));
    const residual = V.sub(pred, V.C(10));

    // Compile with row index 5
    const compiled = compileResidualJacobian(residual, [a, b], 5);

    // Create Jacobian matrix (10 rows × 2 cols)
    const J = Array(10).fill(0).map(() => [0, 0]);
    const paramValues = [2.0, 0.5];

    const value = compiled(paramValues, J);

    console.log("\nMulti-param result:");
    console.log("  value:", value);
    console.log("  J[5]:", J[5]);
    console.log("  Other rows unchanged:", J[0], J[9]);

    // Expected: pred = 2.0 * exp(0.5 * 3) = 2 * exp(1.5) ≈ 8.96
    // residual ≈ 8.96 - 10 = -1.04
    expect(value).toBeCloseTo(2.0 * Math.exp(1.5) - 10, 4);

    // Jacobian row 5 should be updated
    expect(J[5][0]).toBeCloseTo(Math.exp(1.5), 4);  // d/da
    expect(J[5][1]).toBeCloseTo(2.0 * 3.0 * Math.exp(1.5), 4);  // d/db

    // Other rows should be unchanged
    expect(J[0]).toEqual([0, 0]);
    expect(J[9]).toEqual([0, 0]);
  });

  it("should compile 100 residuals efficiently", () => {
    const a = V.W(2.0, "a");
    const b = V.W(0.5, "b");
    a.paramName = "a";
    b.paramName = "b";

    const numResiduals = 100;
    const compiledFunctions = [];

    // Compile 100 residuals, each with its own row index
    for (let i = 0; i < numResiduals; i++) {
      const x = i / 10;
      const pred = V.mul(a, V.exp(V.mul(b, V.C(x))));
      const residual = V.sub(pred, V.C(10));

      compiledFunctions.push(compileResidualJacobian(residual, [a, b], i));
    }

    // Evaluate all residuals
    const J = Array(numResiduals).fill(0).map(() => [0, 0]);
    const residuals: number[] = [];
    const paramValues = [2.0, 0.5];

    const start = performance.now();
    for (let i = 0; i < numResiduals; i++) {
      residuals[i] = compiledFunctions[i](paramValues, J);
    }
    const time = performance.now() - start;

    console.log("\n=== Performance ===");
    console.log(`100 residuals evaluated in ${time.toFixed(2)}ms`);
    console.log(`Per residual: ${(time / numResiduals).toFixed(4)}ms`);
    console.log(`First residual: ${residuals[0].toFixed(4)}`);
    console.log(`Last residual: ${residuals[99].toFixed(4)}`);
    console.log(`J[0]: [${J[0].map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`J[99]: [${J[99].map(v => v.toFixed(4)).join(', ')}]`);

    expect(residuals).toHaveLength(numResiduals);
    expect(J).toHaveLength(numResiduals);
  });
});
