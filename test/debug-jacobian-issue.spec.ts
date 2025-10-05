/**
 * Debug why Jacobian rows are identical for different param usage
 */

import { V } from "../src/V";
import { CompiledFunctions } from "../src/CompiledFunctions";
import { canonicalizeGraphNoSort } from "../src/GraphCanonicalizerNoSort";
import { testLog } from "./testUtils";

describe('Debug Jacobian Issue', () => {
  it('should show canonical strings for different param usage', () => {
    const x = V.W(1, 'x');
    const y = V.W(2, 'y');
    const params = [x, y];

    // Build two residuals
    const r1 = V.sub(x, V.C(5));  // x - 5
    const r2 = V.sub(y, V.C(3));  // y - 3

    // Check their canonical strings
    const { canon: canon1 } = canonicalizeGraphNoSort(r1, params);
    const { canon: canon2 } = canonicalizeGraphNoSort(r2, params);

    testLog('r1 (x-5) canonical:', canon1);
    testLog('r2 (y-3) canonical:', canon2);
    testLog('Are they equal?', canon1 === canon2);

    // Now compile both together
    const compiled = CompiledFunctions.compile(params, (p) => [
      V.sub(p[0], V.C(5)),
      V.sub(p[1], V.C(3))
    ]);

    testLog('\nCompiled info:');
    testLog('Number of kernels:', compiled.kernelCount);
    testLog('Number of functions:', compiled.numFunctions);

    const { values, jacobian } = compiled.evaluateJacobian(params);

    testLog('\nEvaluation:');
    testLog('Values:', values);
    testLog('Jacobian:', jacobian);

    // What we expect:
    // J[0] = [1, 0] because r1 = x - 5, so ∂r1/∂x=1, ∂r1/∂y=0
    // J[1] = [0, 1] because r2 = y - 3, so ∂r2/∂x=0, ∂r2/∂y=1
  });

  it('should check what params each residual actually uses', () => {
    const x = V.W(1, 'x');
    const y = V.W(2, 'y');

    const r1 = V.sub(x, V.C(5));
    const r2 = V.sub(y, V.C(3));

    // Manually check which params are in each graph
    function findParams(node: V.Value, allParams: Set<V.Value>): V.Value[] {
      const found: V.Value[] = [];
      const visited = new Set<V.Value>();

      function traverse(n: V.Value) {
        if (visited.has(n)) return;
        visited.add(n);

        if (allParams.has(n)) {
          found.push(n);
        }

        const prev = (n as any).prev as V.Value[];
        for (const child of prev) {
          traverse(child);
        }
      }

      traverse(node);
      return found;
    }

    const paramSet = new Set([x, y]);
    const r1Params = findParams(r1, paramSet);
    const r2Params = findParams(r2, paramSet);

    testLog('r1 uses params:', r1Params.map(p => p.label));
    testLog('r2 uses params:', r2Params.map(p => p.label));
  });
});
