/**
 * MINIMAL REPRODUCER for the compilation gradient bug.
 *
 * Bug: When the same parameters are used in two different subgraphs
 * that are later combined via multiplication, compiled gradients are wrong.
 */

import { V, Value, CompiledResiduals } from '../src';

describe('Minimal Shared Parameter Bug', () => {
  it('should give same gradients for shared params in multiply', () => {
    console.log('\n=== MINIMAL SHARED PARAMETER BUG ===\n');

    // Create 3 parameters
    const params = [V.W(1.0), V.W(2.0), V.W(3.0)];

    let compilationGraph: Value | null = null;

    // Compile
    const compiled = CompiledResiduals.compile(params, (p) => {
      // Subgraph 1: compute vector from p[0] and p[1], normalize it
      const v0 = V.add(p[0], p[1]);  // 1 + 2 = 3
      const v1 = V.sub(p[1], V.C(0.5));  // 2 - 0.5 = 1.5
      const mag1 = V.sqrt(V.add(V.square(v0), V.square(v1)));
      const norm0 = V.div(v0, mag1);  // normalized component

      // Subgraph 2: compute vector from p[1] and p[2], normalize it (p[1] is SHARED!)
      const u0 = V.sub(p[2], p[1]);  // 3 - 2 = 1
      const u1 = V.add(p[1], V.C(1.0));  // 2 + 1 = 3
      const mag2 = V.sqrt(V.add(V.square(u0), V.square(u1)));
      const norm1 = V.div(u0, mag2);  // normalized component

      // Compute "angle-like" term from p[0] and p[2]
      const angle = V.mul(p[0], p[2]);  // 1 * 3 = 3

      // Combine: angle * (norm0 * norm1)
      // This mimics the pattern: angle * (dist0 * dist1) from the failing case
      const dotProduct = V.mul(norm0, norm1);
      const result = V.mul(angle, dotProduct);

      compilationGraph = result;
      return [result];
    });

    // Graph backward
    console.log(`Forward value: ${compilationGraph!.data.toExponential(6)}\n`);
    params.forEach(p => p.grad = 0);
    compilationGraph!.backward();
    const graphGrads = params.map(p => p.grad);

    console.log('Graph gradients:', graphGrads.map(g => g.toExponential(6)));

    // Compiled backward
    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    console.log('Compiled gradients:', compiledGrads.map(g => g.toExponential(6)));

    // Compare
    const maxDiff = Math.max(...params.map((_, i) => Math.abs(graphGrads[i] - compiledGrads[i])));
    console.log(`\nMax gradient diff: ${maxDiff.toExponential(6)}`);

    for (let i = 0; i < params.length; i++) {
      console.log(`  p[${i}]: graph=${graphGrads[i].toExponential(10)}, compiled=${compiledGrads[i].toExponential(10)}, diff=${Math.abs(graphGrads[i] - compiledGrads[i]).toExponential(6)}`);
    }

    if (maxDiff < 1e-10) {
      console.log('\n✅ PASS - Gradients match!');
    } else {
      console.log('\n❌ FAIL - Gradients differ! BUG REPRODUCED!');
    }

    expect(maxDiff).toBeLessThan(1e-10);
  });

  it('should handle multiple residuals using same parameters', () => {
    console.log('\n=== MULTIPLE RESIDUALS TEST ===\n');

    // Create 9 parameters (3 vertices x 3 components)
    const params = [
      V.W(1.0), V.W(0.0), V.W(0.0),  // vertex 0
      V.W(0.0), V.W(1.0), V.W(0.0),  // vertex 1
      V.W(0.0), V.W(0.0), V.W(1.0),  // vertex 2
    ];

    const compilationGraphs: Value[] = [];

    // Compile - create 2 residuals, each using different subsets of vertices
    const compiled = CompiledResiduals.compile(params, (p) => {
      // Build 3 Vec3 from params
      const v0x = p[0], v0y = p[1], v0z = p[2];
      const v1x = p[3], v1y = p[4], v1z = p[5];
      const v2x = p[6], v2y = p[7], v2z = p[8];

      const residuals: Value[] = [];

      // Residual 0: uses v0 and v1
      const edge01x = V.sub(v1x, v0x);
      const edge01y = V.sub(v1y, v0y);
      const edge01z = V.sub(v1z, v0z);
      const len01 = V.sqrt(V.add(V.add(V.square(edge01x), V.square(edge01y)), V.square(edge01z)));
      const norm01x = V.div(edge01x, len01);

      // Residual 1: uses v1 and v2 (v1 is SHARED with residual 0!)
      const edge12x = V.sub(v2x, v1x);
      const edge12y = V.sub(v2y, v1y);
      const edge12z = V.sub(v2z, v1z);
      const len12 = V.sqrt(V.add(V.add(V.square(edge12x), V.square(edge12y)), V.square(edge12z)));
      const norm12x = V.div(edge12x, len12);

      // Combine them
      const r0 = V.mul(norm01x, norm01x);  // squared normalized component
      const r1 = V.mul(norm12x, norm12x);  // squared normalized component

      residuals.push(r0);
      residuals.push(r1);

      compilationGraphs.push(r0, r1);
      return residuals;
    });

    // Graph backward - sum all residuals
    params.forEach(p => p.grad = 0);
    const graphSum = V.add(compilationGraphs[0], compilationGraphs[1]);
    graphSum.backward();
    const graphGrads = params.map(p => p.grad);

    console.log('Graph gradients:', graphGrads.slice(0, 6).map(g => g.toExponential(6)));

    // Compiled backward
    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    console.log('Compiled gradients:', compiledGrads.slice(0, 6).map(g => g.toExponential(6)));

    // Compare
    const maxDiff = Math.max(...params.map((_, i) => Math.abs(graphGrads[i] - compiledGrads[i])));
    console.log(`\nMax gradient diff: ${maxDiff.toExponential(6)}`);

    if (maxDiff < 1e-10) {
      console.log('✅ PASS - Gradients match!');
    } else {
      console.log('❌ FAIL - Gradients differ! BUG REPRODUCED!');
    }

    expect(maxDiff).toBeLessThan(1e-10);
  });
});
