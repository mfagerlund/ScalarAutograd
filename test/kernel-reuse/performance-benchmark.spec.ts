/**
 * Performance benchmarks: Old vs New implementation
 */

import { V } from "../../src/V";
import { CompiledResiduals } from "../../src/CompiledResiduals";
import { compileResidualJacobian } from "../../src/jit-compile-value";

describe('Performance Benchmarks', () => {
  it('should benchmark compilation time: 100 distance constraints', () => {
    const numConstraints = 100;
    const points: V.Value[] = [];

    for (let i = 0; i < numConstraints * 4; i++) {
      points.push(V.W(Math.random(), `p${i}`));
    }

    const residualFn = (params: V.Value[]) => {
      const res: V.Value[] = [];
      for (let i = 0; i < numConstraints; i++) {
        const idx = i * 4;
        const x1 = params[idx];
        const y1 = params[idx + 1];
        const x2 = params[idx + 2];
        const y2 = params[idx + 3];

        const dx = V.sub(x2, x1);
        const dy = V.sub(y2, y1);
        const distSq = V.add(V.mul(dx, dx), V.mul(dy, dy));
        const dist = V.sqrt(distSq);

        res.push(V.sub(dist, V.C(1.0)));
      }
      return res;
    };

    // Old implementation
    let startOld = performance.now();
    points.forEach((p, i) => { if (!p.paramName) p.paramName = `p${i}`; });
    const residualValues = residualFn(points);
    const compiledOld = residualValues.map((r, i) =>
      compileResidualJacobian(r, points, i)
    );
    const timeOld = performance.now() - startOld;

    // New implementation
    const startNew = performance.now();
    const compiledNew = CompiledResiduals.compile(points, residualFn);
    const timeNew = performance.now() - startNew;

    console.log('\n=== Compilation Time: 100 Distance Constraints ===');
    console.log(`Old (unique kernels):  ${timeOld.toFixed(2)}ms`);
    console.log(`New (kernel reuse):    ${timeNew.toFixed(2)}ms`);
    console.log(`Speedup:               ${(timeOld / timeNew).toFixed(2)}x`);
    console.log(`Kernels:               ${compiledOld.length} → ${compiledNew.kernelCount}`);
  });

  it('should benchmark evaluation time: 100 distance constraints', () => {
    const numConstraints = 100;
    const points: V.Value[] = [];

    for (let i = 0; i < numConstraints * 4; i++) {
      points.push(V.W(Math.random(), `p${i}`));
    }

    const residualFn = (params: V.Value[]) => {
      const res: V.Value[] = [];
      for (let i = 0; i < numConstraints; i++) {
        const idx = i * 4;
        const x1 = params[idx];
        const y1 = params[idx + 1];
        const x2 = params[idx + 2];
        const y2 = params[idx + 3];

        const dx = V.sub(x2, x1);
        const dy = V.sub(y2, y1);
        const distSq = V.add(V.mul(dx, dx), V.mul(dy, dy));
        const dist = V.sqrt(distSq);

        res.push(V.sub(dist, V.C(1.0)));
      }
      return res;
    };

    // Compile both
    points.forEach((p, i) => { if (!p.paramName) p.paramName = `p${i}`; });
    const residualValues = residualFn(points);
    const compiledOld = residualValues.map((r, i) =>
      compileResidualJacobian(r, points, i)
    );

    const compiledNew = CompiledResiduals.compile(points, residualFn);

    const iterations = 1000;

    // Benchmark old
    const paramValues = points.map(p => p.data);
    const JOld = Array(numConstraints).fill(0).map(() => Array(points.length).fill(0));

    const startOld = performance.now();
    for (let iter = 0; iter < iterations; iter++) {
      for (let i = 0; i < numConstraints; i++) {
        compiledOld[i](paramValues, JOld[i]);
      }
    }
    const timeOld = performance.now() - startOld;

    // Benchmark new
    const startNew = performance.now();
    for (let iter = 0; iter < iterations; iter++) {
      compiledNew.evaluate(points);
    }
    const timeNew = performance.now() - startNew;

    console.log('\n=== Evaluation Time: 100 Distance Constraints (1000 iterations) ===');
    console.log(`Old:      ${timeOld.toFixed(2)}ms (${(timeOld / iterations).toFixed(3)}ms per iteration)`);
    console.log(`New:      ${timeNew.toFixed(2)}ms (${(timeNew / iterations).toFixed(3)}ms per iteration)`);
    console.log(`Speedup:  ${(timeOld / timeNew).toFixed(2)}x`);
  });

  it('should benchmark memory usage estimate', () => {
    const numConstraints = 1000;
    const points: V.Value[] = [];

    for (let i = 0; i < numConstraints * 4; i++) {
      points.push(V.W(Math.random(), `p${i}`));
    }

    const residualFn = (params: V.Value[]) => {
      const res: V.Value[] = [];
      for (let i = 0; i < numConstraints; i++) {
        const idx = i * 4;
        const x1 = params[idx];
        const y1 = params[idx + 1];
        const x2 = params[idx + 2];
        const y2 = params[idx + 3];

        const dx = V.sub(x2, x1);
        const dy = V.sub(y2, y1);
        const distSq = V.add(V.mul(dx, dx), V.mul(dy, dy));
        const dist = V.sqrt(distSq);

        res.push(V.sub(dist, V.C(1.0)));
      }
      return res;
    };

    const compiledNew = CompiledResiduals.compile(points, residualFn);

    console.log('\n=== Memory Usage Estimate: 1000 Distance Constraints ===');
    console.log(`Old approach:`);
    console.log(`  Unique kernels:    1000`);
    console.log(`  Est. size:         ~500KB (500 bytes per kernel)`);
    console.log(`\nNew approach:`);
    console.log(`  Unique kernels:    ${compiledNew.kernelCount}`);
    console.log(`  Est. size:         ~${(compiledNew.kernelCount * 500 / 1024).toFixed(1)}KB`);
    console.log(`  Savings:           ${(100 * (1 - compiledNew.kernelCount / 1000)).toFixed(1)}%`);
    console.log(`  Reuse factor:      ${compiledNew.kernelReuseFactor.toFixed(0)}x`);
  });

  it('should benchmark sketch solver scenario: mixed constraints', () => {
    // Realistic sketch: 20 distance, 15 parallel, 10 perpendicular constraints
    const points: V.Value[] = [];
    for (let i = 0; i < 200; i++) {  // Enough points
      points.push(V.W(Math.random() * 100, `p${i}`));
    }

    const residualFn = (params: V.Value[]) => {
      const res: V.Value[] = [];

      // 20 distance constraints (each uses 4 params)
      for (let i = 0; i < 20; i++) {
        const idx = i * 4;
        const dx = V.sub(params[idx + 2], params[idx]);
        const dy = V.sub(params[idx + 3], params[idx + 1]);
        const dist = V.sqrt(V.add(V.mul(dx, dx), V.mul(dy, dy)));
        res.push(V.sub(dist, V.C(10)));
      }

      // 15 parallel constraints
      for (let i = 0; i < 15; i++) {
        const idx = 80 + i * 4;
        const dx1 = V.sub(params[idx + 1], params[idx]);
        const dx2 = V.sub(params[idx + 3], params[idx + 2]);
        res.push(V.sub(dx1, dx2));
      }

      // 10 perpendicular constraints
      for (let i = 0; i < 10; i++) {
        const idx = 140 + i * 4;
        const dx1 = V.sub(params[idx + 1], params[idx]);
        const dx2 = V.sub(params[idx + 3], params[idx + 2]);
        res.push(V.mul(dx1, dx2));
      }

      return res;
    };

    const compiled = CompiledResiduals.compile(points, residualFn);

    console.log('\n=== Sketch Solver: Mixed Constraints ===');
    console.log(`Total residuals:   ${compiled.numResiduals}`);
    console.log(`Unique kernels:    ${compiled.kernelCount}`);
    console.log(`Reuse factor:      ${compiled.kernelReuseFactor.toFixed(1)}x`);
    console.log(`Expected kernels:  3 (distance, parallel, perpendicular)`);

    expect(compiled.kernelCount).toBeLessThanOrEqual(3);
  });
});
