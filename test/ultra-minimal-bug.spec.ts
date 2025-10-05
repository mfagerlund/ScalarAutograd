/**
 * Ultra-minimal reproducer - keep removing until it passes
 */

import { V, Value, Vec3, CompiledResiduals } from '../src';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';

describe('Ultra Minimal Bug', () => {
  it('TWO normals', () => {
    console.log('\n=== TWO NORMALS ===\n');

    const mesh = IcoSphere.generate(0, 1.0);
    const params: Value[] = [];
    for (const v of mesh.vertices) {
      params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
    }

    const paramsToMesh = (p: Value[]) => {
      for (let i = 0; i < mesh.vertices.length; i++) {
        mesh.vertices[i].x = p[3 * i];
        mesh.vertices[i].y = p[3 * i + 1];
        mesh.vertices[i].z = p[3 * i + 2];
      }
    };

    let compilationGraph: Value | null = null;

    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      paramsToMesh(p);

      const star = mesh.getVertexStar(0);
      const n0 = mesh.getFaceNormal(star[0]).normalized;
      const n1 = mesh.getFaceNormal(star[1]).normalized;

      // Just dot product of the two normals, squared
      const dist = Vec3.dot(n0, n1);
      const result = V.mul(dist, dist);

      compilationGraph = result;
      return [result];
    });

    params.forEach(p => p.grad = 0);
    compilationGraph!.backward();
    const graphGrads = params.map(p => p.grad);

    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    const maxDiff = Math.max(...params.map((_, i) => Math.abs(graphGrads[i] - compiledGrads[i])));
    console.log(`Max gradient diff: ${maxDiff.toExponential(6)}`);

    if (maxDiff < 1e-10) {
      console.log('✅ PASS');
    } else {
      console.log('❌ FAIL');
    }

    expect(maxDiff).toBeLessThan(1e-10);
  });

  it('MINIMAL BUG: Compute normals from triangle vertices', () => {
    console.log('\n=== MINIMAL BUG: NORMALS FROM VERTICES ===\n');

    const mesh = IcoSphere.generate(0, 1.0);
    const params: Value[] = [];
    for (const v of mesh.vertices) {
      params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
    }

    const paramsToMesh = (p: Value[]) => {
      for (let i = 0; i < mesh.vertices.length; i++) {
        mesh.vertices[i].x = p[3 * i];
        mesh.vertices[i].y = p[3 * i + 1];
        mesh.vertices[i].z = p[3 * i + 2];
      }
    };

    let compilationGraph: Value | null = null;

    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      paramsToMesh(p);

      const star = mesh.getVertexStar(0);
      console.log(`Vertex 0 star faces: [${star}]`);

      // Check which vertices are in each face
      const face0 = mesh.faces[star[0]];
      const face1 = mesh.faces[star[1]];
      const face2 = mesh.faces[star[2]];
      console.log(`Face ${star[0]} vertices: [${face0.a}, ${face0.b}, ${face0.c}]`);
      console.log(`Face ${star[1]} vertices: [${face1.a}, ${face1.b}, ${face1.c}]`);
      console.log(`Face ${star[2]} vertices: [${face2.a}, ${face2.b}, ${face2.c}]`);

      // Check if p[0] is shared
      console.log(`\nParameter p[0] shared?`);
      console.log(`  mesh.vertices[0].x === p[0]: ${mesh.vertices[0].x === p[0]}`);

      const n0 = mesh.getFaceNormal(star[0]).normalized;
      const n1 = mesh.getFaceNormal(star[1]).normalized;
      const n2 = mesh.getFaceNormal(star[2]).normalized;

      const a0 = mesh.getInteriorAngle(star[0], 0);
      const a1 = mesh.getInteriorAngle(star[1], 0);

      // PlaneNormal from TWO cross products
      const cross01 = Vec3.cross(n0, n1);
      const cross02 = Vec3.cross(n0, n2);
      const planeNormal = cross01.add(cross02).normalized;

      // Dot products
      const dist0 = Vec3.dot(n0, planeNormal);
      const dist1 = Vec3.dot(n1, planeNormal);

      // Energy
      const term0 = V.mul(a0, V.mul(dist0, dist0));
      const term1 = V.mul(a1, V.mul(dist1, dist1));
      const result = V.add(term0, term1);

      compilationGraph = result;
      return [result];
    });

    params.forEach(p => p.grad = 0);
    compilationGraph!.backward();
    const graphGrads = params.map(p => p.grad);

    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    const maxDiff = Math.max(...params.map((_, i) => Math.abs(graphGrads[i] - compiledGrads[i])));
    console.log(`Max gradient diff: ${maxDiff.toExponential(6)}`);

    console.log('First 6 gradients:');
    for (let i = 0; i < Math.min(6, params.length); i++) {
      const diff = Math.abs(graphGrads[i] - compiledGrads[i]);
      console.log(`  p[${i}]: graph=${graphGrads[i].toExponential(10)}, compiled=${compiledGrads[i].toExponential(10)}, diff=${diff.toExponential(6)}`);
    }

    if (maxDiff < 1e-10) {
      console.log('✅ PASS');
    } else {
      console.log('❌ FAIL - BUG REPRODUCED!');
    }

    expect(maxDiff).toBeLessThan(1e-10);
  });
});
