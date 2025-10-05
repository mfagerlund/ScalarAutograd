// Quick test script to compare string vs hash canonicalizers
const { V } = require('./dist/Value');
const { canonicalizeGraph } = require('./dist/GraphCanonicalizer');
const { canonicalizeGraphHash } = require('./dist/GraphHashCanonicalizer');

// Build a complex graph similar to the developable sphere energy
function buildComplexGraph(params) {
  // Simulate a complex energy computation with many operations
  let result = V.C(0);

  for (let i = 0; i < params.length; i += 3) {
    if (i + 2 >= params.length) break;

    const p1 = params[i];
    const p2 = params[i + 1];
    const p3 = params[i + 2];

    // Lots of nested commutative operations
    const dx = V.sub(p2, p1);
    const dy = V.sub(p3, p2);
    const dist1 = V.add(V.square(dx), V.square(dy));
    const dist2 = V.mul(V.abs(dx), V.abs(dy));
    const term = V.add(V.div(dist1, V.add(dist2, V.C(1e-6))), V.C(1));

    result = V.add(result, V.square(term));
  }

  return result;
}

// Test with different numbers of parameters
const testSizes = [21, 42, 63]; // Similar to developable sphere with different subdivisions

console.log('Hash-based Canonicalizer Performance Test\n' + '='.repeat(50));

for (const numParams of testSizes) {
  const params = Array.from({ length: numParams }, (_, i) => V.W(Math.random()));
  const graph = buildComplexGraph(params);

  console.log(`\nTest with ${numParams} parameters:`);

  // String-based canonicalization
  const t0 = performance.now();
  const stringResult = canonicalizeGraph(graph, params);
  const stringTime = performance.now() - t0;

  // Hash-based canonicalization
  const t1 = performance.now();
  const hashResult = canonicalizeGraphHash(graph, params);
  const hashTime = performance.now() - t1;

  console.log(`  String-based: ${stringTime.toFixed(2)}ms`);
  console.log(`  Hash-based:   ${hashTime.toFixed(2)}ms`);
  console.log(`  Speedup:      ${(stringTime / hashTime).toFixed(2)}x`);
  console.log(`  String canon: ${stringResult.canon.substring(0, 80)}...`);
  console.log(`  Hash canon:   ${hashResult.canon.substring(0, 80)}...`);
}

// Test that reordering produces same hash
console.log('\n' + '='.repeat(50));
console.log('Commutativity Test:\n');

const a = V.W(1);
const b = V.W(2);
const c = V.W(3);

const expr1 = V.add(V.add(a, b), c);
const expr2 = V.add(c, V.add(b, a));

const canon1 = canonicalizeGraphHash(expr1, [a, b, c]);
const canon2 = canonicalizeGraphHash(expr2, [a, b, c]);

console.log('  add(add(a,b),c):', canon1.canon);
console.log('  add(c,add(b,a)):', canon2.canon);
console.log('  Match:', canon1.canon === canon2.canon ? 'YES ✓' : 'NO ✗');
