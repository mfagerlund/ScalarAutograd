import { describe, it } from 'vitest';
import { V, Value } from '../src/Value';
import { GraphBuilder } from '../src/GraphBuilder';
import { canonicalizeGraphHash } from '../src/GraphHashCanonicalizer';
import { canonicalizeGraphNoSort } from '../src/GraphCanonicalizerNoSort';

describe('GraphBuilder Performance Benchmarks', () => {
  function createSimpleGraph(params: Value[]): Value {
    const [a, b, c] = params;
    const d = V.add(a, b);
    const e = V.mul(d, c);
    const f = V.exp(e);
    return f;
  }

  function createMediumGraph(params: Value[]): Value {
    let result = params[0];
    for (let i = 1; i < params.length; i++) {
      result = V.add(result, V.mul(params[i], V.C(2)));
      result = V.sin(result);
    }
    return result;
  }

  function createComplexGraph(params: Value[]): Value {
    let sum = V.C(0);
    for (let i = 0; i < params.length - 1; i++) {
      const dx = V.sub(params[i + 1], params[i]);
      const dist = V.sqrt(V.add(V.mul(dx, dx), V.C(1e-12)));
      const energy = V.mul(dist, dist);
      sum = V.add(sum, energy);
    }
    return sum;
  }

  function createHingeEnergyLike(vertexParams: Value[], neighborParams: Value[][]): Value {
    let sum = V.C(0);

    for (const neighbors of neighborParams) {
      let covar_00 = V.C(0);
      let covar_01 = V.C(0);
      let covar_11 = V.C(0);

      for (const neighbor of neighbors) {
        const diff_x = V.sub(neighbor, vertexParams[0]);
        const diff_y = V.sub(neighbor, vertexParams[1]);

        const mag = V.sqrt(V.add(V.mul(diff_x, diff_x), V.mul(diff_y, diff_y)));
        const norm_x = V.div(diff_x, V.max(mag, V.C(1e-12)));
        const norm_y = V.div(diff_y, V.max(mag, V.C(1e-12)));

        covar_00 = V.add(covar_00, V.mul(norm_x, norm_x));
        covar_01 = V.add(covar_01, V.mul(norm_x, norm_y));
        covar_11 = V.add(covar_11, V.mul(norm_y, norm_y));
      }

      const trace = V.add(covar_00, covar_11);
      const det = V.sub(V.mul(covar_00, covar_11), V.mul(covar_01, covar_01));
      const discriminant = V.sub(V.mul(trace, trace), V.mul(V.C(4), det));
      const lambda = V.div(V.sub(trace, V.sqrt(V.max(discriminant, V.C(0)))), 2);

      sum = V.add(sum, V.max(lambda, V.C(0)));
    }

    return sum;
  }

  it('Benchmark: Simple graph (3 params, 4 ops)', () => {
    const iterations = 10000;
    const params = [V.W(1), V.W(2), V.W(3)];

    console.log('\n--- Simple Graph (3 params, 4 ops) ---');

    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
      const output = createSimpleGraph(params);
    }
    const t1 = performance.now();
    console.log(`Graph build only: ${((t1 - t0) / iterations).toFixed(3)}ms per iteration`);

    const t2 = performance.now();
    for (let i = 0; i < iterations; i++) {
      const output = createSimpleGraph(params);
      canonicalizeGraphHash(output, params);
    }
    const t3 = performance.now();
    console.log(`Old approach (build + hash): ${((t3 - t2) / iterations).toFixed(3)}ms per iteration`);

    const t4 = performance.now();
    for (let i = 0; i < iterations; i++) {
      const builder = new GraphBuilder(params);
      const { output, signature } = builder.build(() => createSimpleGraph(params));
    }
    const t5 = performance.now();
    console.log(`New approach (GraphBuilder): ${((t5 - t4) / iterations).toFixed(3)}ms per iteration`);

    const speedup = (t3 - t2) / (t5 - t4);
    console.log(`Speedup: ${speedup.toFixed(2)}x`);
  });

  it('Benchmark: Medium graph (10 params, ~30 ops)', () => {
    const iterations = 5000;
    const params = Array.from({ length: 10 }, (_, i) => V.W(i));

    console.log('\n--- Medium Graph (10 params, ~30 ops) ---');

    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
      const output = createMediumGraph(params);
    }
    const t1 = performance.now();
    console.log(`Graph build only: ${((t1 - t0) / iterations).toFixed(3)}ms per iteration`);

    const t2 = performance.now();
    for (let i = 0; i < iterations; i++) {
      const output = createMediumGraph(params);
      canonicalizeGraphHash(output, params);
    }
    const t3 = performance.now();
    console.log(`Old approach (build + hash): ${((t3 - t2) / iterations).toFixed(3)}ms per iteration`);

    const t4 = performance.now();
    for (let i = 0; i < iterations; i++) {
      const builder = new GraphBuilder(params);
      const { output, signature } = builder.build(() => createMediumGraph(params));
    }
    const t5 = performance.now();
    console.log(`New approach (GraphBuilder): ${((t5 - t4) / iterations).toFixed(3)}ms per iteration`);

    const speedup = (t3 - t2) / (t5 - t4);
    console.log(`Speedup: ${speedup.toFixed(2)}x`);
  });

  it('Benchmark: Complex graph (20 params, ~100 ops)', () => {
    const iterations = 1000;
    const params = Array.from({ length: 20 }, (_, i) => V.W(i));

    console.log('\n--- Complex Graph (20 params, ~100 ops) ---');

    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
      const output = createComplexGraph(params);
    }
    const t1 = performance.now();
    console.log(`Graph build only: ${((t1 - t0) / iterations).toFixed(3)}ms per iteration`);

    const t2 = performance.now();
    for (let i = 0; i < iterations; i++) {
      const output = createComplexGraph(params);
      canonicalizeGraphHash(output, params);
    }
    const t3 = performance.now();
    console.log(`Old approach (build + hash): ${((t3 - t2) / iterations).toFixed(3)}ms per iteration`);

    const t4 = performance.now();
    for (let i = 0; i < iterations; i++) {
      const builder = new GraphBuilder(params);
      const { output, signature } = builder.build(() => createComplexGraph(params));
    }
    const t5 = performance.now();
    console.log(`New approach (GraphBuilder): ${((t5 - t4) / iterations).toFixed(3)}ms per iteration`);

    const speedup = (t3 - t2) / (t5 - t4);
    console.log(`Speedup: ${speedup.toFixed(2)}x`);
  });

  it('Benchmark: Hinge energy-like graph (6 params, ~50 ops)', () => {
    const iterations = 1000;
    const vertexParams = [V.W(0), V.W(0)];
    const neighborParams = [
      [V.W(1), V.W(0)],
      [V.W(0.5), V.W(0.866)],
      [V.W(-0.5), V.W(0.866)],
    ];
    const allParams = [...vertexParams, ...neighborParams.flat()];

    console.log('\n--- Hinge Energy-like Graph (8 params, ~50 ops) ---');

    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
      const output = createHingeEnergyLike(vertexParams, neighborParams);
    }
    const t1 = performance.now();
    console.log(`Graph build only: ${((t1 - t0) / iterations).toFixed(3)}ms per iteration`);

    const t2 = performance.now();
    for (let i = 0; i < iterations; i++) {
      const output = createHingeEnergyLike(vertexParams, neighborParams);
      canonicalizeGraphHash(output, allParams);
    }
    const t3 = performance.now();
    console.log(`Old approach (build + hash): ${((t3 - t2) / iterations).toFixed(3)}ms per iteration`);

    const t4 = performance.now();
    for (let i = 0; i < iterations; i++) {
      const builder = new GraphBuilder(allParams);
      const { output, signature } = builder.build(() =>
        createHingeEnergyLike(vertexParams, neighborParams)
      );
    }
    const t5 = performance.now();
    console.log(`New approach (GraphBuilder): ${((t5 - t4) / iterations).toFixed(3)}ms per iteration`);

    const speedup = (t3 - t2) / (t5 - t4);
    console.log(`Speedup: ${speedup.toFixed(2)}x`);
  });

  it('Benchmark: Hash collision rate test', () => {
    console.log('\n--- Hash Collision Test ---');

    const signatures = new Set<string>();
    const iterations = 1000;

    for (let i = 0; i < iterations; i++) {
      const numParams = 3 + (i % 10);
      const params = Array.from({ length: numParams }, (_, j) => V.W(j));

      const builder = new GraphBuilder(params);
      const { signature } = builder.build(() => {
        let result = params[0];
        for (let j = 1; j < params.length; j++) {
          if (i % 3 === 0) {
            result = V.add(result, V.mul(params[j], V.C(2)));
          } else if (i % 3 === 1) {
            result = V.sub(result, V.div(params[j], V.C(2)));
          } else {
            result = V.mul(result, V.sin(params[j]));
          }
        }
        return result;
      });

      signatures.add(signature.hash);
    }

    console.log(`Generated ${iterations} graphs`);
    console.log(`Unique signatures: ${signatures.size}`);
    console.log(`Collision rate: ${(((iterations - signatures.size) / iterations) * 100).toFixed(2)}%`);
  });

  it('Benchmark: Signature stability across rebuilds', () => {
    console.log('\n--- Signature Stability Test ---');

    const params = [V.W(1), V.W(2), V.W(3)];
    const iterations = 100;
    const signatures: string[] = [];

    for (let i = 0; i < iterations; i++) {
      const builder = new GraphBuilder(params);
      const { signature } = builder.build(() => createSimpleGraph(params));
      signatures.push(signature.hash);
    }

    const uniqueSignatures = new Set(signatures);
    console.log(`Rebuilt ${iterations} times`);
    console.log(`Unique signatures: ${uniqueSignatures.size} (should be 1)`);
    console.log(`Stable: ${uniqueSignatures.size === 1 ? 'YES' : 'NO'}`);

    if (uniqueSignatures.size > 1) {
      console.log('WARNING: Signatures are not stable!');
      console.log('First 5 signatures:', signatures.slice(0, 5));
    }
  });
});
