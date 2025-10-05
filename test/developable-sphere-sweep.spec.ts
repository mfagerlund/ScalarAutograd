/**
 * Parameter sweep for developable sphere optimization
 *
 * This is a long-running test (minutes to hours) that's skipped by default.
 * Run with: npm test -- --grep "Parameter sweep"
 * Or force run: FORCE_SWEEP=1 npm test -- partition-sweep
 */

import { describe, it } from 'vitest';
import { V } from '../src/V';
import { Value, Vec3 } from '../src/Value';
import { lbfgs } from '../src/LBFGS';
import { CompiledFunctions } from '../src/CompiledFunctions';
import { testLog } from './testUtils';

// Skip this test by default unless explicitly requested
const shouldRun = process.env.FORCE_SWEEP === '1';
const testFn = shouldRun ? it : it.skip;

interface TriangleMesh {
  vertices: Vec3[];
  faces: { vertices: [number, number, number] }[];

  getVertexStar(idx: number): number[];
  getFaceNormal(idx: number): Vec3;
  clone(): TriangleMesh;
  setVertexPosition(idx: number, pos: Vec3): void;
}

class SimpleMesh implements TriangleMesh {
  vertices: Vec3[];
  faces: { vertices: [number, number, number] }[];

  constructor(vertices: Vec3[], faces: { vertices: [number, number, number] }[]) {
    this.vertices = vertices;
    this.faces = faces;
  }

  getVertexStar(idx: number): number[] {
    const star: number[] = [];
    for (let i = 0; i < this.faces.length; i++) {
      const face = this.faces[i];
      if (face.vertices.includes(idx)) {
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

  clone(): TriangleMesh {
    return new SimpleMesh(
      this.vertices.map(v => v.clone()),
      this.faces.map(f => ({ vertices: [...f.vertices] as [number, number, number] }))
    );
  }

  setVertexPosition(idx: number, pos: Vec3): void {
    this.vertices[idx] = pos;
  }

  static fromIcoSphere(subdivisions: number, radius: number): SimpleMesh {
    // Manual icosphere generation with subdivision
    const t = (1.0 + Math.sqrt(5.0)) / 2.0;
    let vertices: Vec3[] = [
      Vec3.W(-1, t, 0), Vec3.W(1, t, 0), Vec3.W(-1, -t, 0), Vec3.W(1, -t, 0),
      Vec3.W(0, -1, t), Vec3.W(0, 1, t), Vec3.W(0, -1, -t), Vec3.W(0, 1, -t),
      Vec3.W(t, 0, -1), Vec3.W(t, 0, 1), Vec3.W(-t, 0, -1), Vec3.W(-t, 0, 1),
    ].map(v => {
      const len = Math.sqrt(v.x.data * v.x.data + v.y.data * v.y.data + v.z.data * v.z.data);
      return Vec3.W(v.x.data / len, v.y.data / len, v.z.data / len);
    });

    let faces: { vertices: [number, number, number] }[] = [
      {vertices: [0, 11, 5]}, {vertices: [0, 5, 1]}, {vertices: [0, 1, 7]}, {vertices: [0, 7, 10]}, {vertices: [0, 10, 11]},
      {vertices: [1, 5, 9]}, {vertices: [5, 11, 4]}, {vertices: [11, 10, 2]}, {vertices: [10, 7, 6]}, {vertices: [7, 1, 8]},
      {vertices: [3, 9, 4]}, {vertices: [3, 4, 2]}, {vertices: [3, 2, 6]}, {vertices: [3, 6, 8]}, {vertices: [3, 8, 9]},
      {vertices: [4, 9, 5]}, {vertices: [2, 4, 11]}, {vertices: [6, 2, 10]}, {vertices: [8, 6, 7]}, {vertices: [9, 8, 1]},
    ];

    // Subdivide
    for (let i = 0; i < subdivisions; i++) {
      const edgeCache = new Map<string, number>();
      const newFaces: { vertices: [number, number, number] }[] = [];

      const getMidpoint = (v1: number, v2: number): number => {
        const key = v1 < v2 ? `${v1}-${v2}` : `${v2}-${v1}`;
        if (edgeCache.has(key)) return edgeCache.get(key)!;

        const a = vertices[v1];
        const b = vertices[v2];
        const mx = (a.x.data + b.x.data) / 2;
        const my = (a.y.data + b.y.data) / 2;
        const mz = (a.z.data + b.z.data) / 2;
        const len = Math.sqrt(mx * mx + my * my + mz * mz);

        const newIdx = vertices.length;
        vertices.push(Vec3.W(mx / len, my / len, mz / len));
        edgeCache.set(key, newIdx);
        return newIdx;
      };

      for (const face of faces) {
        const v1 = face.vertices[0];
        const v2 = face.vertices[1];
        const v3 = face.vertices[2];
        const a = getMidpoint(v1, v2);
        const b = getMidpoint(v2, v3);
        const c = getMidpoint(v3, v1);

        newFaces.push({vertices: [v1, a, c]});
        newFaces.push({vertices: [v2, b, a]});
        newFaces.push({vertices: [v3, c, b]});
        newFaces.push({vertices: [a, b, c]});
      }

      faces = newFaces;
    }

    // Scale to radius
    vertices = vertices.map(v =>
      Vec3.W(v.x.data * radius, v.y.data * radius, v.z.data * radius)
    );

    return new SimpleMesh(vertices, faces);
  }
}

// Energy interface - pluggable!
interface EnergyFunction {
  name: string;
  compute(mesh: TriangleMesh): Value;
  computeResiduals(mesh: TriangleMesh): Value[];
  classifyVertices(mesh: TriangleMesh, threshold?: number): { hingeVertices: number[]; seamVertices: number[] };
}

// Original: Variance-based energy
class VarianceEnergy implements EnergyFunction {
  name = 'Variance';

  compute(mesh: TriangleMesh): Value {
    const energies: Value[] = [];
    for (let i = 0; i < mesh.vertices.length; i++) {
      const vertexEnergy = this.computeVertexEnergy(i, mesh);
      energies.push(vertexEnergy);
    }
    return V.sum(energies);
  }

  computeResiduals(mesh: TriangleMesh): Value[] {
    const residuals: Value[] = [];
    for (let i = 0; i < mesh.vertices.length; i++) {
      residuals.push(this.computeVertexEnergy(i, mesh));
    }
    return residuals;
  }

  private computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);

    const mid = Math.floor(star.length / 2);
    const region1 = star.slice(0, mid);
    const region2 = star.slice(mid);

    const energy1 = this.computeRegionEnergy(region1, mesh);
    const energy2 = this.computeRegionEnergy(region2, mesh);

    return V.add(energy1, energy2);
  }

  private computeRegionEnergy(region: number[], mesh: TriangleMesh): Value {
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

  classifyVertices(
    mesh: TriangleMesh,
    hingeThreshold: number = 1e-3
  ): { hingeVertices: number[]; seamVertices: number[] } {
    const hingeVertices: number[] = [];
    const seamVertices: number[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      const energy = this.computeVertexEnergy(i, mesh).data;
      if (energy < hingeThreshold) {
        hingeVertices.push(i);
      } else {
        seamVertices.push(i);
      }
    }

    return { hingeVertices, seamVertices };
  }
}

// Alternative: Angle Defect energy
// Bounding box on sphere: measure spread of normals
class BoundingBoxEnergy implements EnergyFunction {
  name = 'BoundingBox';

  compute(mesh: TriangleMesh): Value {
    const energies: Value[] = [];
    for (let i = 0; i < mesh.vertices.length; i++) {
      const vertexEnergy = this.computeVertexEnergy(i, mesh);
      energies.push(vertexEnergy);
    }
    return V.sum(energies);
  }

  computeResiduals(mesh: TriangleMesh): Value[] {
    const residuals: Value[] = [];
    for (let i = 0; i < mesh.vertices.length; i++) {
      residuals.push(this.computeVertexEnergy(i, mesh));
    }
    return residuals;
  }

  private computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);

    // Split into two regions
    const mid = Math.floor(star.length / 2);
    const region1 = star.slice(0, mid);
    const region2 = star.slice(mid);

    const energy1 = this.computeRegionBoundingBox(region1, mesh);
    const energy2 = this.computeRegionBoundingBox(region2, mesh);

    return V.add(energy1, energy2);
  }

  private computeRegionBoundingBox(region: number[], mesh: TriangleMesh): Value {
    if (region.length === 0) return V.C(0);

    // Get all normals in this region
    const normals: Vec3[] = [];
    for (const faceIdx of region) {
      normals.push(mesh.getFaceNormal(faceIdx).normalized);
    }

    // Find the pair of normals with maximum separation (longest axis)
    let maxDist = V.C(0);
    let axis1 = normals[0];
    let axis2 = normals[0];

    for (let i = 0; i < normals.length; i++) {
      for (let j = i + 1; j < normals.length; j++) {
        const diff = normals[i].sub(normals[j]);
        const dist = diff.sqrMagnitude;
        const isMax = V.gt(dist, maxDist);
        maxDist = V.ifThenElse(isMax, dist, maxDist);
        // Can't conditionally assign vectors in autograd, so we'll use a simpler approach
      }
    }

    // Simpler approach: compute spread along all pairwise directions
    // and use max distance * perpendicular max distance
    let maxSpread1 = V.C(0);

    for (let i = 0; i < normals.length; i++) {
      for (let j = i + 1; j < normals.length; j++) {
        const dist = normals[i].sub(normals[j]).magnitude;
        maxSpread1 = V.max(maxSpread1, dist);
      }
    }

    // For perpendicular spread, we approximate by looking at variance in orthogonal direction
    // This is a simplified metric: longest axis * average perpendicular distance
    const avgNormal = normals.reduce((sum, n) => sum.add(n), Vec3.zero()).div(normals.length);

    let perpSpread = V.C(0);
    for (const n of normals) {
      const proj = Vec3.dot(n.sub(avgNormal), avgNormal.normalized);
      const perp = n.sub(avgNormal.normalized.mul(proj));
      perpSpread = V.add(perpSpread, perp.magnitude);
    }
    perpSpread = V.div(perpSpread, normals.length);

    // Bounding box area = longest axis * perpendicular spread
    return V.mul(maxSpread1, perpSpread);
  }

  classifyVertices(
    mesh: TriangleMesh,
    hingeThreshold: number = 0.3
  ): { hingeVertices: number[]; seamVertices: number[] } {
    const hingeVertices: number[] = [];
    const seamVertices: number[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      const energy = this.computeVertexEnergy(i, mesh).data;
      if (energy < hingeThreshold) {
        hingeVertices.push(i);
      } else {
        seamVertices.push(i);
      }
    }

    return { hingeVertices, seamVertices };
  }
}

class AngleDefectEnergy implements EnergyFunction {
  name = 'AngleDefect';

  compute(mesh: TriangleMesh): Value {
    let total = V.C(0);
    for (let i = 0; i < mesh.vertices.length; i++) {
      const star = mesh.getVertexStar(i);
      if (star.length === 0) continue;

      // Sum angles at this vertex
      let angleSum = V.C(0);
      for (const faceIdx of star) {
        const angle = this.computeAngleAtVertex(i, faceIdx, mesh);
        angleSum = V.add(angleSum, angle);
      }

      // Defect from 2π (flat would be exactly 2π)
      const defect = V.sub(angleSum, V.C(2 * Math.PI));
      const energy = V.mul(defect, defect);
      total = V.add(total, energy);
    }
    return total;
  }

  computeResiduals(mesh: TriangleMesh): Value[] {
    const residuals: Value[] = [];
    for (let i = 0; i < mesh.vertices.length; i++) {
      const star = mesh.getVertexStar(i);
      if (star.length === 0) {
        residuals.push(V.C(0));
        continue;
      }

      // Sum angles at this vertex
      let angleSum = V.C(0);
      for (const faceIdx of star) {
        const angle = this.computeAngleAtVertex(i, faceIdx, mesh);
        angleSum = V.add(angleSum, angle);
      }

      // Defect from 2π (flat would be exactly 2π)
      const defect = V.sub(angleSum, V.C(2 * Math.PI));
      const energy = V.mul(defect, defect);
      residuals.push(energy);
    }
    return residuals;
  }

  private computeAngleAtVertex(vertexIdx: number, faceIdx: number, mesh: TriangleMesh): Value {
    const face = mesh.faces[faceIdx];
    const [v0, v1, v2] = face.vertices;

    // Find which position in the triangle this vertex is
    let localIdx = -1;
    if (v0 === vertexIdx) localIdx = 0;
    else if (v1 === vertexIdx) localIdx = 1;
    else if (v2 === vertexIdx) localIdx = 2;

    if (localIdx === -1) return V.C(0);

    // Get the three vertices in order
    const verts = [v0, v1, v2];
    const curr = mesh.vertices[verts[localIdx]];
    const prev = mesh.vertices[verts[(localIdx + 2) % 3]];
    const next = mesh.vertices[verts[(localIdx + 1) % 3]];

    // Vectors from current to neighbors
    const e1 = prev.sub(curr);
    const e2 = next.sub(curr);

    // Angle using dot product: cos(θ) = (e1·e2) / (|e1||e2|)
    const dot = Vec3.dot(e1, e2);
    const len1 = e1.magnitude;
    const len2 = e2.magnitude;
    const cosAngle = V.div(dot, V.add(V.mul(len1, len2), V.C(1e-8)));

    // acos(clamp(cosAngle, -1, 1))
    const clamped = V.clamp(cosAngle, -1, 1);
    return V.acos(clamped);
  }

  classifyVertices(
    mesh: TriangleMesh,
    hingeThreshold: number = 1e-4
  ): { hingeVertices: number[]; seamVertices: number[] } {
    const hingeVertices: number[] = [];
    const seamVertices: number[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      const star = mesh.getVertexStar(i);
      if (star.length === 0) continue;

      let angleSum = 0;  // Use number, not Value
      for (const faceIdx of star) {
        const angle = this.computeAngleAtVertex(i, faceIdx, mesh);
        angleSum += angle.data;  // Extract numerical value immediately
      }

      const defect = Math.abs(angleSum - 2 * Math.PI);
      if (defect < hingeThreshold) {
        hingeVertices.push(i);
      } else {
        seamVertices.push(i);
      }
    }

    return { hingeVertices, seamVertices };
  }
}

// Wrapper for backward compatibility
class DevelopableEnergy {
  static compute(mesh: TriangleMesh): Value {
    return new VarianceEnergy().compute(mesh);
  }

  static classifyVertices(mesh: TriangleMesh, threshold?: number) {
    return new VarianceEnergy().classifyVertices(mesh, threshold);
  }
}

interface OptimizationConfig {
  subdivisions: number;
  maxIterations: number;
  gradientTolerance: number;
  chunkSize: number;
  energyFunction: EnergyFunction;
}

interface OptimizationResult {
  config: OptimizationConfig;
  developableRatio: number;
  finalEnergy: number;
  iterations: number;
  functionEvals: number;
  convergenceReason: string;
  gradientNorm?: number;
  timeMs: number;
  vertices: number;
  hingeVertices: number;
  seamVertices: number;
}

async function runOptimization(config: OptimizationConfig): Promise<OptimizationResult> {
  const { subdivisions, maxIterations, gradientTolerance, energyFunction } = config;

  testLog(`\nTesting: ${energyFunction.name}, subdiv=${subdivisions}, maxIter=${maxIterations}, gradTol=${gradientTolerance}`);

  const startTime = performance.now();

  // Create sphere
  const sphere = SimpleMesh.fromIcoSphere(subdivisions, 1.0);

  // Convert to params
  const params: Value[] = [];
  for (const v of sphere.vertices) {
    params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
  }

  // Compile residuals for all energy types
  testLog(`  Compiling residuals...`);
  const compiled = CompiledFunctions.compile(params, (p: Value[]) => {
    // Update mesh with parameters
    const numVertices = sphere.vertices.length;
    for (let i = 0; i < numVertices; i++) {
      const x = p[3 * i];
      const y = p[3 * i + 1];
      const z = p[3 * i + 2];
      sphere.setVertexPosition(i, new Vec3(x, y, z));
    }
    return energyFunction.computeResiduals(sphere);
  });

  testLog(`  Compiled: ${compiled.kernelCount} kernels, ${compiled.kernelReuseFactor.toFixed(1)}x reuse`);

  // IMPORTANT: Restore mesh to initial state after compilation
  // (compilation modifies the mesh during graph building)
  for (let i = 0; i < sphere.vertices.length; i++) {
    const x = params[3 * i];
    const y = params[3 * i + 1];
    const z = params[3 * i + 2];
    sphere.setVertexPosition(i, new Vec3(x, y, z));
  }

  testLog(`  Starting L-BFGS optimization...`);
  // Run optimization
  const result = lbfgs(params, compiled, {
    maxIterations,
    gradientTolerance,
    verbose: false,
  });
  testLog(`  L-BFGS finished: ${result.convergenceReason}`);

  // Update final mesh
  const numVertices = sphere.vertices.length;
  for (let i = 0; i < numVertices; i++) {
    const x = params[3 * i];
    const y = params[3 * i + 1];
    const z = params[3 * i + 2];
    sphere.setVertexPosition(i, new Vec3(x, y, z));
  }

  const endTime = performance.now();
  const timeMs = endTime - startTime;

  // Compute final metrics
  const classification = energyFunction.classifyVertices(sphere);
  const developableRatio = classification.hingeVertices.length / sphere.vertices.length;

  const stats: OptimizationResult = {
    config,
    developableRatio,
    finalEnergy: result.finalCost,
    iterations: result.iterations,
    functionEvals: result.functionEvaluations,
    convergenceReason: result.convergenceReason,
    timeMs,
    vertices: sphere.vertices.length,
    hingeVertices: classification.hingeVertices.length,
    seamVertices: classification.seamVertices.length,
  };

  testLog(`  ✓ Developable: ${(developableRatio * 100).toFixed(2)}%`);
  testLog(`    Energy: ${result.finalCost.toExponential(3)}`);
  testLog(`    Iterations: ${result.iterations}`);
  testLog(`    Convergence: ${result.convergenceReason}`);
  testLog(`    Time: ${timeMs.toFixed(0)}ms`);

  return stats;
}

describe('Developable Sphere Optimization', () => {
  testFn('Parameter sweep for optimal settings', async () => {
    const varianceEnergy = new VarianceEnergy();
    const boundingBoxEnergy = new BoundingBoxEnergy();
    const angleDefectEnergy = new AngleDefectEnergy();

    const configs: OptimizationConfig[] = [
      // Test all three energies with subdivision 3 (642 vertices)
      { subdivisions: 3, maxIterations: 100, gradientTolerance: 1e-6, chunkSize: 5, energyFunction: varianceEnergy },
      { subdivisions: 3, maxIterations: 100, gradientTolerance: 1e-6, chunkSize: 5, energyFunction: boundingBoxEnergy },
      { subdivisions: 3, maxIterations: 100, gradientTolerance: 1e-6, chunkSize: 5, energyFunction: angleDefectEnergy },
    ];

    const results: OptimizationResult[] = [];

    testLog(`\n${'='.repeat(60)}`);
    testLog('PARAMETER SWEEP - DEVELOPABLE SPHERE');
    testLog(`Testing ${configs.length} configurations`);
    testLog('='.repeat(60));

    for (const config of configs) {
      try {
        const result = await runOptimization(config);
        results.push(result);
      } catch (error: any) {
        console.error(`  ✗ Failed: ${error.message}`);
      }
    }

    // Analysis
    testLog(`\n${'='.repeat(60)}`);
    testLog('RESULTS SUMMARY');
    testLog('='.repeat(60));

    const sorted = [...results].sort((a, b) => b.developableRatio - a.developableRatio);

    testLog('\nTop 5 configurations:');
    sorted.slice(0, 5).forEach((r, i) => {
      testLog(`\n${i + 1}. ${(r.developableRatio * 100).toFixed(2)}% developable`);
      testLog(`   Energy: ${r.config.energyFunction.name}, subdiv=${r.config.subdivisions}, maxIter=${r.config.maxIterations}, gradTol=${r.config.gradientTolerance}`);
      testLog(`   Final Energy: ${r.finalEnergy.toExponential(3)}, Time: ${r.timeMs.toFixed(0)}ms`);
      testLog(`   Convergence: ${r.convergenceReason}`);
    });

    testLog('\n\nBest per subdivision level:');
    for (const subdiv of [2, 3, 4]) {
      const best = sorted.find(r => r.config.subdivisions === subdiv);
      if (best) {
        testLog(`\nSubdivision ${subdiv} (${best.vertices} vertices):`);
        testLog(`  ${(best.developableRatio * 100).toFixed(2)}% developable`);
        testLog(`  maxIter=${best.config.maxIterations}, gradTol=${best.config.gradientTolerance}`);
        testLog(`  Time: ${best.timeMs.toFixed(0)}ms`);
      }
    }

    testLog('\n' + '='.repeat(60));
  }, { timeout: 600000 }); // 10 minute timeout
});
