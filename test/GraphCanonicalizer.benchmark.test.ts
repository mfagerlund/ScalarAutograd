import { describe, it } from 'vitest';
import { V } from '../src/Value';
import { canonicalizeGraphHash } from '../src/GraphHashCanonicalizer';
import { canonicalizeGraphNoSort } from '../src/GraphCanonicalizerNoSort';

describe('Canonicalizer Performance Comparison', () => {
  it('should compare performance on complex graphs', () => {
    // Build a VERY complex graph with many nested additions (worst case for string sorting)
    function buildComplexGraph(params: any[]) {
      // Create many terms that will all be added together
      // This creates a flat add() with many children - lots of sorting!
      const terms: any[] = [];

      for (let i = 0; i < params.length; i += 3) {
        if (i + 2 >= params.length) break;

        const p1 = params[i];
        const p2 = params[i + 1];
        const p3 = params[i + 2];

        // Create deeply nested additions - worst case for string canonicalizer
        const dx = V.sub(p2, p1);
        const dy = V.sub(p3, p2);
        const dz = V.sub(p1, p3);

        // Many addition terms (will be flattened and sorted)
        const t1 = V.add(V.square(dx), V.square(dy));
        const t2 = V.add(V.square(dy), V.square(dz));
        const t3 = V.add(V.square(dz), V.square(dx));
        const t4 = V.add(V.abs(dx), V.abs(dy));
        const t5 = V.add(V.abs(dy), V.abs(dz));
        const t6 = V.add(V.abs(dz), V.abs(dx));

        // More additions (nested)
        const sum1 = V.add(V.add(t1, t2), V.add(t3, t4));
        const sum2 = V.add(V.add(t5, t6), V.add(t1, t3));
        const sum3 = V.add(V.add(t2, t4), V.add(t5, t6));

        terms.push(sum1, sum2, sum3);

        // Add multiplication terms too
        const m1 = V.mul(V.mul(p1, p2), p3);
        const m2 = V.mul(V.mul(p2, p3), p1);
        terms.push(V.add(m1, m2));
      }

      // Combine all terms with addition (creates huge flat addition node)
      let result = terms[0] || V.C(0);
      for (let i = 1; i < terms.length; i++) {
        result = V.add(result, terms[i]);
      }

      return result;
    }

    const testSizes = [21, 42, 63, 84, 126, 168]; // Up to ~1000 nodes

    console.log('\n' + '='.repeat(70));
    console.log('Canonicalizer Performance Comparison');
    console.log('='.repeat(70));

    for (const numParams of testSizes) {
      const params = Array.from({ length: numParams }, () => V.W(Math.random()));
      const graph = buildComplexGraph(params);

      console.log(`\nGraph with ${numParams} parameters:`);

      // Warmup
      canonicalizeGraphHash(graph, params);
      canonicalizeGraphNoSort(graph, params);

      // Benchmark no-sort (ID-based)
      const noSortTimes: number[] = [];
      for (let i = 0; i < 5; i++) {
        const t0 = performance.now();
        canonicalizeGraphNoSort(graph, params);
        noSortTimes.push(performance.now() - t0);
      }

      // Benchmark hash-based
      const hashTimes: number[] = [];
      for (let i = 0; i < 5; i++) {
        const t0 = performance.now();
        canonicalizeGraphHash(graph, params);
        hashTimes.push(performance.now() - t0);
      }

      const avgNoSort = noSortTimes.reduce((a, b) => a + b, 0) / noSortTimes.length;
      const avgHash = hashTimes.reduce((a, b) => a + b, 0) / hashTimes.length;

      console.log(`  ID-based (no-sort): ${avgNoSort.toFixed(2)}ms`);
      console.log(`  Hash-based:         ${avgHash.toFixed(2)}ms  (${(avgNoSort / avgHash).toFixed(2)}x)`);

      // Compare canonical outputs
      const noSortResult = canonicalizeGraphNoSort(graph, params);
      const hashResult = canonicalizeGraphHash(graph, params);
      console.log(`  No-sort canon:  ${noSortResult.canon.substring(0, 80)}...`);
      console.log(`  Hash canon:     ${hashResult.canon.substring(0, 80)}...`);
    }

    console.log('\n' + '='.repeat(70));
  });
});
