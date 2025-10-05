/**
 * Try to reproduce the mesh energy bug without using the mesh.
 * Pattern from minimal failing case:
 * - 3 face normals (n0, n1, n2) computed via cross products
 * - 2 interior angles (a0, a1)
 * - planeNormal from cross(n0,n1) + cross(n0,n2), normalized
 * - energy = a0 * dot(n0, planeNormal)^2 + a1 * dot(n1, planeNormal)^2
 */

import { V, Value, Vec3, CompiledResiduals } from '../src';
import { testLog } from './testUtils';

describe('Mesh Pattern Bug', () => {
  it.skip('should handle cross product and dot product pattern', () => {
    testLog('\n=== MESH PATTERN TEST ===\n');

    // 9 parameters for 3 vertices (non-planar for different normals)
    const params = [
      V.W(0.0), V.W(0.0), V.W(0.0),  // v0
      V.W(1.0), V.W(0.0), V.W(0.1),  // v1
      V.W(0.5), V.W(0.866), V.W(-0.05),  // v2
    ];

    let compilationGraph: Value | null = null;

    const compiled = CompiledResiduals.compile(params, (p) => {
      // Build Vec3 vertices
      const v0 = new Vec3(p[0], p[1], p[2]);
      const v1 = new Vec3(p[3], p[4], p[5]);
      const v2 = new Vec3(p[6], p[7], p[8]);

      // Compute "face normals" via cross products
      const edge0 = v1.sub(v0);
      const edge1 = v2.sub(v0);
      const n0 = Vec3.cross(edge0, edge1).normalized;

      const edge2 = v2.sub(v1);
      const edge3 = v0.sub(v1);
      const n1 = Vec3.cross(edge2, edge3).normalized;

      const edge4 = v0.sub(v2);
      const edge5 = v1.sub(v2);
      const n2 = Vec3.cross(edge4, edge5).normalized;

      // Compute angles (simplified - just use dot products for now)
      const a0 = V.C(1.5);  // constant for now
      const a1 = V.C(2.0);

      // Build planeNormal from cross products
      const cross01 = Vec3.cross(n0, n1);
      const cross02 = Vec3.cross(n0, n2);
      const planeNormal = cross01.add(cross02).normalized;

      // Compute distances (dot products)
      const dist0 = Vec3.dot(n0, planeNormal);
      const dist1 = Vec3.dot(n1, planeNormal);

      // Energy terms
      const term0 = V.mul(a0, V.mul(dist0, dist0));
      const term1 = V.mul(a1, V.mul(dist1, dist1));

      const result = V.add(term0, term1);

      compilationGraph = result;
      return [result];
    });

    // Graph backward
    testLog(`Forward value: ${compilationGraph!.data.toExponential(6)}\n`);
    params.forEach(p => p.grad = 0);
    compilationGraph!.backward();
    const graphGrads = params.map(p => p.grad);

    testLog('Graph gradients (first 6):', graphGrads.slice(0, 6).map(g => g.toExponential(6)));

    // Compiled backward
    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    testLog('Compiled gradients (first 6):', compiledGrads.slice(0, 6).map(g => g.toExponential(6)));

    // Compare
    const maxDiff = Math.max(...params.map((_, i) => Math.abs(graphGrads[i] - compiledGrads[i])));
    testLog(`\nMax gradient diff: ${maxDiff.toExponential(6)}`);

    for (let i = 0; i < Math.min(6, params.length); i++) {
      const diff = Math.abs(graphGrads[i] - compiledGrads[i]);
      testLog(`  p[${i}]: graph=${graphGrads[i].toExponential(10)}, compiled=${compiledGrads[i].toExponential(10)}, diff=${diff.toExponential(6)}`);
    }

    if (maxDiff < 1e-10) {
      testLog('\n✅ PASS - Gradients match!');
    } else {
      testLog('\n❌ FAIL - Gradients differ! BUG REPRODUCED!');
    }

    expect(maxDiff).toBeLessThan(1e-10);
  });
});
