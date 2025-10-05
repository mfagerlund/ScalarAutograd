/**
 * Test L-BFGS with compiled gradient evaluation
 */

import { V } from "../src/V";
import { CompiledFunctions } from "../src/CompiledFunctions";
import { lbfgs } from "../src/LBFGS";
import { testLog } from "./testUtils";

describe('L-BFGS with CompiledFunctions', () => {
  it.skip('should optimize Rosenbrock with compiled gradient', () => {
    const x = V.W(-1.2, 'x');
    const y = V.W(1.0, 'y');
    const params = [x, y];

    // Rosenbrock: (1-x)² + 100(y-x²)²
    const compiled = CompiledFunctions.compile(params, (p) => {
      const [px, py] = p;
      const term1 = V.square(V.sub(V.C(1), px));
      const term2 = V.mul(V.C(100), V.square(V.sub(py, V.square(px))));
      return [term1, term2];
    });

    testLog('\nRosenbrock with compiled gradients:');
    testLog(`Kernels: ${compiled.kernelCount}`);
    testLog(`Functions: ${compiled.numFunctions}`);

    const result = lbfgs(params, compiled, {
      maxIterations: 200,
      gradientTolerance: 1e-5,
      verbose: false
    });

    testLog(`Final: x=${x.data.toFixed(6)}, y=${y.data.toFixed(6)}`);
    testLog(`Iterations: ${result.iterations}`);
    testLog(`Convergence: ${result.convergenceReason}`);

    // Should converge near (1, 1) - Rosenbrock is hard
    expect(x.data).toBeCloseTo(1, 2);
    expect(y.data).toBeCloseTo(1, 2);
  });

  it('should handle many identical residuals with kernel reuse', () => {
    const N = 100;
    const params = Array.from({ length: N }, (_, i) =>
      V.W(Math.random() * 2 - 1, `p${i}`)
    );

    // N residuals: (p[i] - target[i])²
    const targets = Array.from({ length: N }, (_, i) => i * 0.01);

    const compiled = CompiledFunctions.compile(params, (p) =>
      p.map((param, i) => V.square(V.sub(param, V.C(targets[i]))))
    );

    testLog(`\n${N} residuals with kernel reuse:`);
    testLog(`Kernels: ${compiled.kernelCount}`);
    testLog(`Reuse factor: ${compiled.kernelReuseFactor.toFixed(1)}x`);

    // Should compile to 1 kernel
    expect(compiled.kernelCount).toBe(1);
    expect(compiled.kernelReuseFactor).toBeCloseTo(N, 1);

    const initialCost = compiled.evaluateSumWithGradient(params).value;

    const result = lbfgs(params, compiled, {
      maxIterations: 50,
      verbose: false
    });

    const finalCost = compiled.evaluateSumWithGradient(params).value;

    testLog(`Cost: ${initialCost.toFixed(6)} → ${finalCost.toFixed(6)}`);
    testLog(`Iterations: ${result.iterations}`);

    // Should converge to targets
    expect(finalCost).toBeLessThan(1e-10);
    params.forEach((p, i) => {
      expect(p.data).toBeCloseTo(targets[i], 6);
    });
  });

  it('should optimize distance constraints with kernel reuse', () => {
    // 4 points forming a square
    const points = [
      V.W(0.1, 'x0'), V.W(0.1, 'y0'),
      V.W(0.9, 'x1'), V.W(0.1, 'y1'),
      V.W(0.9, 'x2'), V.W(0.9, 'y2'),
      V.W(0.1, 'x3'), V.W(0.9, 'y3')
    ];

    // 4 edge constraints + 2 diagonal constraints
    const compiled = CompiledFunctions.compile(points, (p) => {
      const distance = (i: number, j: number, target: number) => {
        const dx = V.sub(p[i * 2], p[j * 2]);
        const dy = V.sub(p[i * 2 + 1], p[j * 2 + 1]);
        const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
        return V.sub(dist, V.C(target));
      };

      return [
        distance(0, 1, 1.0), // edge
        distance(1, 2, 1.0), // edge
        distance(2, 3, 1.0), // edge
        distance(3, 0, 1.0), // edge
        distance(0, 2, Math.sqrt(2)), // diagonal
        distance(1, 3, Math.sqrt(2))  // diagonal
      ];
    });

    testLog('\nDistance constraints:');
    testLog(`Kernels: ${compiled.kernelCount}`);
    testLog(`Constraints: ${compiled.numFunctions}`);

    const initialCost = compiled.evaluateSumWithGradient(points).value;

    const result = lbfgs(points, compiled, {
      maxIterations: 100,
      verbose: false
    });

    const finalCost = compiled.evaluateSumWithGradient(points).value;

    testLog(`Cost: ${initialCost.toFixed(6)} → ${finalCost.toFixed(10)}`);
    testLog(`Iterations: ${result.iterations}`);

    // Should satisfy all constraints
    expect(finalCost).toBeLessThan(1e-8);
  });

  it('should match uncompiled L-BFGS results', () => {
    const x1 = V.W(0.5, 'x');
    const y1 = V.W(0.5, 'y');

    const x2 = V.W(0.5, 'x');
    const y2 = V.W(0.5, 'y');

    // Uncompiled version
    const result1 = lbfgs([x1, y1], (params) => {
      const [px, py] = params;
      return V.add(
        V.square(V.sub(px, V.C(2))),
        V.square(V.sub(py, V.C(3)))
      );
    }, {
      maxIterations: 50,
      verbose: false
    });

    // Compiled version
    const compiled = CompiledFunctions.compile([x2, y2], (p) => [
      V.square(V.sub(p[0], V.C(2))),
      V.square(V.sub(p[1], V.C(3)))
    ]);

    const result2 = lbfgs([x2, y2], compiled, {
      maxIterations: 50,
      verbose: false
    });

    testLog('\nUncompiled vs Compiled:');
    testLog(`Uncompiled: (${x1.data.toFixed(6)}, ${y1.data.toFixed(6)}) in ${result1.iterations} iters`);
    testLog(`Compiled:   (${x2.data.toFixed(6)}, ${y2.data.toFixed(6)}) in ${result2.iterations} iters`);

    // Should produce identical results
    expect(x1.data).toBeCloseTo(x2.data, 8);
    expect(y1.data).toBeCloseTo(y2.data, 8);
  });
});
