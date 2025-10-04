/**
 * Benchmark L-BFGS performance: compiled vs uncompiled
 */

import { V } from "../src/V";
import { CompiledFunctions } from "../src/CompiledFunctions";
import { lbfgs } from "../src/LBFGS";
import { Value } from "../src/Value";

describe('L-BFGS Compiled Performance', () => {
  it('should benchmark compiled vs uncompiled gradient evaluation', () => {
    const N = 500; // 500 residuals
    const iterations = 20;

    console.log(`\n=== L-BFGS Performance: ${N} residuals, ${iterations} iterations ===\n`);

    // Uncompiled version
    const params1 = Array.from({ length: N }, (_, i) => V.W(Math.random(), `p${i}`));
    const targets = Array.from({ length: N }, (_, i) => i * 0.01);

    const start1 = performance.now();
    const result1 = lbfgs(params1, (p) => {
      const residuals = p.map((param, i) => V.square(V.sub(param, V.C(targets[i]))));
      return V.sum(residuals);
    }, {
      maxIterations: iterations,
      verbose: false
    });
    const time1 = performance.now() - start1;

    console.log('Uncompiled:');
    console.log(`  Time: ${time1.toFixed(2)}ms`);
    console.log(`  Iterations: ${result1.iterations}`);
    console.log(`  Final cost: ${result1.finalCost.toExponential(3)}`);

    // Compiled version
    const params2 = Array.from({ length: N }, (_, i) => V.W(Math.random(), `p${i}`));

    const compiled = CompiledFunctions.compile(params2, (p) =>
      p.map((param, i) => V.square(V.sub(param, V.C(targets[i]))))
    );

    console.log(`\nCompiled:`);
    console.log(`  Kernels: ${compiled.kernelCount}`);
    console.log(`  Reuse factor: ${compiled.kernelReuseFactor.toFixed(1)}x`);

    const start2 = performance.now();
    const result2 = lbfgs(params2, compiled, {
      maxIterations: iterations,
      verbose: false
    });
    const time2 = performance.now() - start2;

    console.log(`  Time: ${time2.toFixed(2)}ms`);
    console.log(`  Iterations: ${result2.iterations}`);
    console.log(`  Final cost: ${result2.finalCost.toExponential(3)}`);

    const speedup = time1 / time2;
    console.log(`\n**Speedup: ${speedup.toFixed(2)}x**`);

    expect(speedup).toBeGreaterThan(1.5);
  });

  it('should benchmark with distance constraints', () => {
    const N = 50; // 50 points = 100 params
    const iterations = 30;

    console.log(`\n=== Distance Constraints: ${N} points, ${iterations} iterations ===\n`);

    // Create N points in a circle, optimize to target distances
    const createPoints = () => Array.from({ length: N * 2 }, (_, i) => {
      const angle = (i / 2) * 2 * Math.PI / N;
      return V.W(i % 2 === 0 ? Math.cos(angle) + Math.random() * 0.1 : Math.sin(angle) + Math.random() * 0.1, `p${i}`);
    });

    const targetDist = 2 * Math.PI / N; // Arc length for unit circle

    // Uncompiled
    const params1 = createPoints();

    const start1 = performance.now();
    const result1 = lbfgs(params1, (p) => {
      const residuals: Value[] = [];
      for (let i = 0; i < N; i++) {
        const j = (i + 1) % N;
        const dx = V.sub(p[i * 2], p[j * 2]);
        const dy = V.sub(p[i * 2 + 1], p[j * 2 + 1]);
        const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
        residuals.push(V.square(V.sub(dist, V.C(targetDist))));
      }
      return V.sum(residuals);
    }, {
      maxIterations: iterations,
      verbose: false
    });
    const time1 = performance.now() - start1;

    console.log('Uncompiled:');
    console.log(`  Time: ${time1.toFixed(2)}ms`);
    console.log(`  Iterations: ${result1.iterations}`);
    console.log(`  Final cost: ${result1.finalCost.toExponential(3)}`);

    // Compiled
    const params2 = createPoints();

    const compiled = CompiledFunctions.compile(params2, (p) => {
      const residuals: Value[] = [];
      for (let i = 0; i < N; i++) {
        const j = (i + 1) % N;
        const dx = V.sub(p[i * 2], p[j * 2]);
        const dy = V.sub(p[i * 2 + 1], p[j * 2 + 1]);
        const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
        residuals.push(V.square(V.sub(dist, V.C(targetDist))));
      }
      return residuals;
    });

    console.log(`\nCompiled:`);
    console.log(`  Kernels: ${compiled.kernelCount}`);
    console.log(`  Constraints: ${compiled.numFunctions}`);
    console.log(`  Reuse factor: ${compiled.kernelReuseFactor.toFixed(1)}x`);

    const start2 = performance.now();
    const result2 = lbfgs(params2, compiled, {
      maxIterations: iterations,
      verbose: false
    });
    const time2 = performance.now() - start2;

    console.log(`  Time: ${time2.toFixed(2)}ms`);
    console.log(`  Iterations: ${result2.iterations}`);
    console.log(`  Final cost: ${result2.finalCost.toExponential(3)}`);

    const speedup = time1 / time2;
    console.log(`\n**Speedup: ${speedup.toFixed(2)}x**`);

    expect(speedup).toBeGreaterThan(1.2);
  });

  it('should show scaling with problem size', () => {
    const sizes = [50, 100, 200, 500];
    const iterations = 10;

    console.log(`\n=== Scaling Analysis (${iterations} iterations) ===\n`);
    console.log('| Size | Uncompiled | Compiled | Speedup | Reuse |');
    console.log('|------|------------|----------|---------|-------|');

    for (const N of sizes) {
      const targets = Array.from({ length: N }, (_, i) => i * 0.01);

      // Uncompiled
      const params1 = Array.from({ length: N }, (_, i) => V.W(Math.random(), `p${i}`));
      const start1 = performance.now();
      lbfgs(params1, (p) => {
        const residuals = p.map((param, i) => V.square(V.sub(param, V.C(targets[i]))));
        return V.sum(residuals);
      }, { maxIterations: iterations, verbose: false });
      const time1 = performance.now() - start1;

      // Compiled
      const params2 = Array.from({ length: N }, (_, i) => V.W(Math.random(), `p${i}`));
      const compiled = CompiledFunctions.compile(params2, (p) =>
        p.map((param, i) => V.square(V.sub(param, V.C(targets[i]))))
      );
      const start2 = performance.now();
      lbfgs(params2, compiled, { maxIterations: iterations, verbose: false });
      const time2 = performance.now() - start2;

      const speedup = time1 / time2;
      console.log(`| ${N.toString().padEnd(4)} | ${time1.toFixed(1).padEnd(10)} | ${time2.toFixed(1).padEnd(8)} | ${speedup.toFixed(2)}x${' '.repeat(4)} | ${compiled.kernelReuseFactor.toFixed(0)}x${' '.repeat(4)} |`);
    }

    console.log('');
  });
});
