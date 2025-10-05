/**
 * Test to verify compiled and non-compiled produce identical results
 * for developable sphere energies
 */
import { describe, it, expect } from 'vitest';
import { V, Vec3, lbfgs, CompiledFunctions, Value } from '../src';
import { testLog } from './testUtils';

// Minimal mesh for testing
interface SimpleMesh {
  vertices: Vec3[];
  faces: { vertices: [number, number, number] }[];
  getVertexStar(idx: number): number[];
  getFaceNormal(idx: number): Vec3;
  setVertexPosition(idx: number, pos: Vec3): void;
}

class TestMesh implements SimpleMesh {
  vertices: Vec3[];
  faces: { vertices: [number, number, number] }[];

  constructor(vertices: Vec3[], faces: { vertices: [number, number, number] }[]) {
    this.vertices = vertices;
    this.faces = faces;
  }

  getVertexStar(idx: number): number[] {
    const star: number[] = [];
    for (let i = 0; i < this.faces.length; i++) {
      if (this.faces[i].vertices.includes(idx)) {
        star.push(i);
      }
    }
    return star;
  }

  getFaceNormal(idx: number): Vec3 {
    const face = this.faces[idx];
    const v0 = this.vertices[face.vertices[0]];
    const v1 = this.vertices[face.vertices[1]];
    const v2 = this.vertices[face.vertices[2]];
    const e1 = v1.sub(v0);
    const e2 = v2.sub(v0);
    return Vec3.cross(e1, e2).normalized;
  }

  setVertexPosition(idx: number, pos: Vec3): void {
    this.vertices[idx] = pos;
  }
}

// Variance energy (same as in sweep test)
class VarianceEnergy {
  static compute(mesh: SimpleMesh): Value {
    const energies: Value[] = [];
    for (let i = 0; i < mesh.vertices.length; i++) {
      energies.push(this.computeVertexEnergy(i, mesh));
    }
    return V.sum(energies);
  }

  static computeResiduals(mesh: SimpleMesh): Value[] {
    const residuals: Value[] = [];
    for (let i = 0; i < mesh.vertices.length; i++) {
      residuals.push(this.computeVertexEnergy(i, mesh));
    }
    return residuals;
  }

  private static computeVertexEnergy(vertexIdx: number, mesh: SimpleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);

    const mid = Math.floor(star.length / 2);
    const region1 = star.slice(0, mid);
    const region2 = star.slice(mid);

    const energy1 = this.computeRegionEnergy(region1, mesh);
    const energy2 = this.computeRegionEnergy(region2, mesh);

    return V.add(energy1, energy2);
  }

  private static computeRegionEnergy(region: number[], mesh: SimpleMesh): Value {
    const n = region.length;
    if (n === 0) return V.C(0);

    let avgNormal = Vec3.zero();
    for (const faceIdx of region) {
      const normal = mesh.getFaceNormal(faceIdx);
      avgNormal = avgNormal.add(normal);
    }
    avgNormal = avgNormal.div(n);

    let deviation = V.C(0);
    for (const faceIdx of region) {
      const normal = mesh.getFaceNormal(faceIdx);
      const diff = normal.sub(avgNormal);
      deviation = V.add(deviation, diff.sqrMagnitude);
    }

    const normalizationFactor = n * n;
    return V.div(deviation, normalizationFactor);
  }
}

describe('Developable Energy Comparison', () => {
  it.concurrent('should produce identical results for variance energy with compiled vs non-compiled', async () => {
    // Create a simple icosphere (12 vertices, 20 faces)
    const t = (1.0 + Math.sqrt(5.0)) / 2.0;
    const vertices = [
      Vec3.W(-1, t, 0), Vec3.W(1, t, 0), Vec3.W(-1, -t, 0), Vec3.W(1, -t, 0),
      Vec3.W(0, -1, t), Vec3.W(0, 1, t), Vec3.W(0, -1, -t), Vec3.W(0, 1, -t),
      Vec3.W(t, 0, -1), Vec3.W(t, 0, 1), Vec3.W(-t, 0, -1), Vec3.W(-t, 0, 1),
    ].map(v => {
      const len = Math.sqrt(v.x.data ** 2 + v.y.data ** 2 + v.z.data ** 2);
      return Vec3.W(v.x.data / len, v.y.data / len, v.z.data / len);
    });

    const faces: { vertices: [number, number, number] }[] = [
      {vertices: [0, 11, 5]}, {vertices: [0, 5, 1]}, {vertices: [0, 1, 7]}, {vertices: [0, 7, 10]}, {vertices: [0, 10, 11]},
      {vertices: [1, 5, 9]}, {vertices: [5, 11, 4]}, {vertices: [11, 10, 2]}, {vertices: [10, 7, 6]}, {vertices: [7, 1, 8]},
      {vertices: [3, 9, 4]}, {vertices: [3, 4, 2]}, {vertices: [3, 2, 6]}, {vertices: [3, 6, 8]}, {vertices: [3, 8, 9]},
      {vertices: [4, 9, 5]}, {vertices: [2, 4, 11]}, {vertices: [6, 2, 10]}, {vertices: [8, 6, 7]}, {vertices: [9, 8, 1]},
    ];

    // Test 1: Non-compiled
    const mesh1 = new TestMesh([...vertices], faces);
    const params1: Value[] = [];
    for (const v of mesh1.vertices) {
      params1.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
    }

    const result1 = await lbfgs(params1, (p) => {
      for (let i = 0; i < mesh1.vertices.length; i++) {
        mesh1.setVertexPosition(i, new Vec3(p[3*i], p[3*i+1], p[3*i+2]));
      }
      return VarianceEnergy.compute(mesh1);
    }, { maxIterations: 20, verbose: false });

    // Test 2: Compiled
    const mesh2 = new TestMesh([...vertices], faces);
    const params2: Value[] = [];
    for (const v of mesh2.vertices) {
      params2.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
    }

    const compiled = CompiledFunctions.compile(params2, (p) => {
      for (let i = 0; i < mesh2.vertices.length; i++) {
        mesh2.setVertexPosition(i, new Vec3(p[3*i], p[3*i+1], p[3*i+2]));
      }
      return VarianceEnergy.computeResiduals(mesh2);
    });

    const result2 = await lbfgs(params2, compiled, { maxIterations: 20, verbose: false });

    testLog('Non-compiled:');
    testLog('  Energy:', result1.finalCost);
    testLog('  Iterations:', result1.iterations);
    testLog('  Convergence:', result1.convergenceReason);

    testLog('\nCompiled:');
    testLog('  Energy:', result2.finalCost);
    testLog('  Iterations:', result2.iterations);
    testLog('  Convergence:', result2.convergenceReason);
    testLog('  Kernels:', compiled.kernelCount, 'Reuse:', compiled.kernelReuseFactor.toFixed(1) + 'x');

    testLog('\nDifference:');
    testLog('  Energy diff:', Math.abs(result1.finalCost - result2.finalCost));
    testLog('  Max param diff:', Math.max(...params1.map((p, i) => Math.abs(p.data - params2[i].data))));

    // Check results are close
    expect(Math.abs(result1.finalCost - result2.finalCost)).toBeLessThan(1e-6);
    expect(Math.max(...params1.map((p, i) => Math.abs(p.data - params2[i].data)))).toBeLessThan(1e-6);
  });
});
