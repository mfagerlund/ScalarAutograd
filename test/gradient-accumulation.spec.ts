/**
 * Test gradient accumulation in compiled kernels
 */

import { V } from "../src/V";
import { CompiledFunctions } from "../src/CompiledFunctions";

describe('Gradient Accumulation', () => {
  it('should accumulate gradients across multiple residuals', () => {
    const x = V.W(2.0, 'x');
    const y = V.W(3.0, 'y');

    // Two residuals: (x - 5) and (y - 3)
    // These should share the same kernel
    const compiled = CompiledFunctions.compile([x, y], (params) => {
      const [px, py] = params;
      return [
        V.sub(px, V.C(5)), // residual 1: x - 5 = -3
        V.sub(py, V.C(3))  // residual 2: y - 3 = 0
      ];
    });

    // Verify kernel reuse
    expect(compiled.kernelCount).toBe(1);
    expect(compiled.numFunctions).toBe(2);

    // Evaluate sum with gradient
    const { value, gradient } = compiled.evaluateSumWithGradient([x, y]);

    // Sum: (x-5) + (y-3) = (2-5) + (3-3) = -3 + 0 = -3
    expect(value).toBeCloseTo(-3, 10);

    // Gradients:
    // d/dx (x-5 + y-3) = 1
    // d/dy (x-5 + y-3) = 1
    expect(gradient[0]).toBeCloseTo(1, 10);
    expect(gradient[1]).toBeCloseTo(1, 10);
  });

  it('should accumulate gradients from structurally identical residuals', () => {
    const params = [V.W(1.0, 'a'), V.W(2.0, 'b'), V.W(3.0, 'c')];

    // Three identical structures: (a-1)², (b-2)², (c-3)²
    const compiled = CompiledFunctions.compile(params, (p) => [
      V.square(V.sub(p[0], V.C(1))),
      V.square(V.sub(p[1], V.C(2))),
      V.square(V.sub(p[2], V.C(3)))
    ]);

    // Should share 1 kernel
    expect(compiled.kernelCount).toBe(1);
    expect(compiled.kernelReuseFactor).toBeCloseTo(3, 10);

    const { value, gradient } = compiled.evaluateSumWithGradient(params);

    // Sum: (1-1)² + (2-2)² + (3-3)² = 0
    expect(value).toBeCloseTo(0, 10);

    // Gradients: d/da (a-1)² = 2(a-1) = 0, same for b,c
    expect(gradient[0]).toBeCloseTo(0, 10);
    expect(gradient[1]).toBeCloseTo(0, 10);
    expect(gradient[2]).toBeCloseTo(0, 10);
  });

  it('should handle distance constraints with kernel reuse', () => {
    // 3 points forming a triangle with distance constraints
    const x1 = V.W(0.0, 'x1');
    const y1 = V.W(0.0, 'y1');
    const x2 = V.W(3.0, 'x2');
    const y2 = V.W(4.0, 'y2');
    const x3 = V.W(6.0, 'x3');
    const y3 = V.W(0.0, 'y3');

    const params = [x1, y1, x2, y2, x3, y3];

    // 3 distance constraints (all should share the same kernel)
    const compiled = CompiledFunctions.compile(params, (p) => {
      const distanceSquared = (i1: number, i2: number, target: number) => {
        const dx = V.sub(p[i1], p[i2]);
        const dy = V.sub(p[i1 + 1], p[i2 + 1]);
        const distSq = V.add(V.square(dx), V.square(dy));
        return V.sub(distSq, V.C(target * target));
      };

      return [
        distanceSquared(0, 2, 5), // d(p1,p2) - 5
        distanceSquared(2, 4, 6), // d(p2,p3) - 6
        distanceSquared(4, 0, 6)  // d(p3,p1) - 6
      ];
    });

    // 2 kernels: one for target=5, one for target=6
    expect(compiled.kernelCount).toBe(2);
    expect(compiled.numFunctions).toBe(3);

    const { value, gradient } = compiled.evaluateSumWithGradient(params);

    // Current distances:
    // p1-p2: sqrt(3²+4²) = 5 ✓
    // p2-p3: sqrt(3²+4²) = 5 (target 6, error=-11)
    // p3-p1: sqrt(6²+0²) = 6 ✓

    // Sum of squared residuals
    expect(value).toBeCloseTo(0 - 11 + 0, 10);

    // Gradients should be accumulated from all 3 constraints
    expect(gradient.length).toBe(6);
  });

  it('should match evaluateGradient for single function', () => {
    const x = V.W(2.0, 'x');
    const y = V.W(3.0, 'y');

    const compiled = CompiledFunctions.compile([x, y], (params) => {
      const [px, py] = params;
      return [V.add(V.square(px), V.square(py))]; // x² + y²
    });

    const result1 = compiled.evaluateGradient([x, y]);
    const result2 = compiled.evaluateSumWithGradient([x, y]);

    expect(result1.value).toBeCloseTo(result2.value, 10);
    expect(result1.gradient[0]).toBeCloseTo(result2.gradient[0], 10);
    expect(result1.gradient[1]).toBeCloseTo(result2.gradient[1], 10);
  });

  it('should work with many identical residuals', () => {
    const N = 100;
    const params = Array.from({ length: N }, (_, i) => V.W(i * 0.1, `p${i}`));

    // N identical structures: (p[i] - i*0.1)²
    const compiled = CompiledFunctions.compile(params, (p) =>
      p.map((param, i) => V.square(V.sub(param, V.C(i * 0.1))))
    );

    // Should have massive kernel reuse
    expect(compiled.kernelCount).toBe(1);
    expect(compiled.kernelReuseFactor).toBeCloseTo(N, 10);

    const { value, gradient } = compiled.evaluateSumWithGradient(params);

    // All at optimal values: sum should be 0
    expect(value).toBeCloseTo(0, 8);

    // All gradients should be 0
    gradient.forEach(g => expect(g).toBeCloseTo(0, 8));
  });
});
