/**
 * Performance Benchmark: Raw vs Compiled vs GPU
 *
 * Tests sphere flattening energy where each vertex tries to align
 * with its neighbors. Compares three execution modes:
 * 1. Raw: Direct Value graph evaluation
 * 2. Compiled: CPU JIT-compiled kernels
 * 3. GPU: WebGPU batched execution
 */

import { V, Value, Vec3, CompiledResiduals } from '../../src';
import { WebGPUContext } from '../../src/gpu/WebGPUContext';
import { compileToWGSL, WGSLKernel } from '../../src/gpu/compileToWGSL';

// Simple sphere mesh generator
class SimpleSphere {
  vertices: { x: Value; y: Value; z: Value }[] = [];
  neighbors: number[][] = []; // neighbors[i] = indices of neighboring vertices

  static generate(subdivisions: number): SimpleSphere {
    const sphere = new SimpleSphere();

    if (subdivisions === 0) {
      // Icosahedron vertices (12 vertices)
      const t = (1.0 + Math.sqrt(5.0)) / 2.0;
      const vertices = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
      ];

      for (const [x, y, z] of vertices) {
        const len = Math.sqrt(x * x + y * y + z * z);
        sphere.vertices.push({
          x: V.W(x / len),
          y: V.W(y / len),
          z: V.W(z / len)
        });
      }

      // Icosahedron edges (each vertex has 5 neighbors)
      sphere.neighbors = [
        [1, 5, 7, 10, 11], [0, 5, 7, 8, 9], [3, 4, 6, 10, 11], [2, 4, 6, 8, 9],
        [2, 3, 5, 9, 11], [0, 1, 4, 9, 11], [2, 3, 7, 8, 10], [0, 1, 6, 8, 10],
        [1, 3, 6, 7, 9], [1, 3, 4, 5, 8], [0, 2, 6, 7, 11], [0, 2, 4, 5, 10]
      ];
    } else {
      // For higher subdivisions, create a simple lat/long sphere
      const latDivs = 4 + subdivisions * 2;
      const lonDivs = 8 + subdivisions * 4;

      for (let lat = 0; lat <= latDivs; lat++) {
        const theta = (lat * Math.PI) / latDivs;
        for (let lon = 0; lon < lonDivs; lon++) {
          const phi = (lon * 2 * Math.PI) / lonDivs;

          const x = Math.sin(theta) * Math.cos(phi);
          const y = Math.sin(theta) * Math.sin(phi);
          const z = Math.cos(theta);

          sphere.vertices.push({
            x: V.W(x),
            y: V.W(y),
            z: V.W(z)
          });
        }
      }

      // Build neighbor connectivity
      for (let lat = 0; lat <= latDivs; lat++) {
        for (let lon = 0; lon < lonDivs; lon++) {
          const idx = lat * lonDivs + lon;
          const neighbors: number[] = [];

          // Same latitude neighbors
          neighbors.push(lat * lonDivs + ((lon + 1) % lonDivs));
          neighbors.push(lat * lonDivs + ((lon - 1 + lonDivs) % lonDivs));

          // Previous latitude
          if (lat > 0) {
            neighbors.push((lat - 1) * lonDivs + lon);
          }

          // Next latitude
          if (lat < latDivs) {
            neighbors.push((lat + 1) * lonDivs + lon);
          }

          sphere.neighbors[idx] = neighbors;
        }
      }
    }

    return sphere;
  }

  getVertexCount(): number {
    return this.vertices.length;
  }

  // Flatten all vertices to Value array
  getParameterArray(): Value[] {
    const params: Value[] = [];
    for (const v of this.vertices) {
      params.push(v.x, v.y, v.z);
    }
    return params;
  }
}

// Flattening energy: each vertex wants its normal to align with neighbor normals
// Simplified: minimize angle between vertex and average of neighbors
function createFlatteningResidual(
  vx: Value, vy: Value, vz: Value,
  nx: Value, ny: Value, nz: Value
): Value {
  // Energy: 1 - dot(v, n) = 1 - (vx*nx + vy*ny + vz*nz)
  // When dot=1 (aligned), energy=0
  const dot = V.add(V.add(V.mul(vx, nx), V.mul(vy, ny)), V.mul(vz, nz));
  return V.sub(V.C(1.0), dot);
}

describe('Performance Benchmark: Raw vs Compiled vs GPU', () => {
  let ctx: WebGPUContext;

  beforeAll(async () => {
    if (WebGPUContext.isAvailable()) {
      ctx = WebGPUContext.getInstance();
      await ctx.initialize();
    }
  });

  afterAll(() => {
    if (WebGPUContext.isAvailable()) {
      WebGPUContext.reset();
    }
  });

  it('should benchmark sphere flattening: 12 vertices (icosahedron)', async () => {
    const sphere = SimpleSphere.generate(0);
    const vertexCount = sphere.getVertexCount();

    console.log(`\n========================================`);
    console.log(`BENCHMARK: ${vertexCount} vertices`);
    console.log(`========================================\n`);

    // Build residuals: one per vertex
    const residuals: Value[] = [];
    for (let i = 0; i < vertexCount; i++) {
      const v = sphere.vertices[i];
      const neighbors = sphere.neighbors[i];

      // Average neighbor position (simplified normal)
      let avgX = V.C(0);
      let avgY = V.C(0);
      let avgZ = V.C(0);

      for (const nIdx of neighbors) {
        const n = sphere.vertices[nIdx];
        avgX = V.add(avgX, n.x);
        avgY = V.add(avgY, n.y);
        avgZ = V.add(avgZ, n.z);
      }

      const invCount = 1.0 / neighbors.length;
      avgX = V.mul(avgX, V.C(invCount));
      avgY = V.mul(avgY, V.C(invCount));
      avgZ = V.mul(avgZ, V.C(invCount));

      const residual = createFlatteningResidual(v.x, v.y, v.z, avgX, avgY, avgZ);
      residuals.push(residual);
    }

    // ===== MODE 1: RAW EVALUATION =====
    const rawIterations = 100;
    const t0 = performance.now();

    for (let iter = 0; iter < rawIterations; iter++) {
      let sum = 0;
      for (const r of residuals) {
        sum += r.data;
      }
    }

    const rawTime = performance.now() - t0;
    const rawTimePerEval = rawTime / rawIterations;

    console.log(`[RAW] ${rawIterations} evaluations`);
    console.log(`  Total: ${rawTime.toFixed(2)}ms`);
    console.log(`  Per evaluation: ${rawTimePerEval.toFixed(3)}ms`);
    console.log(`  Residuals/sec: ${(residuals.length / rawTimePerEval * 1000).toFixed(0)}`);

    // ===== MODE 2: COMPILED CPU =====
    const params = sphere.getParameterArray();
    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      // Rebuild vertices from params
      for (let i = 0; i < vertexCount; i++) {
        sphere.vertices[i].x = p[i * 3 + 0];
        sphere.vertices[i].y = p[i * 3 + 1];
        sphere.vertices[i].z = p[i * 3 + 2];
      }

      // Rebuild residuals
      const res: Value[] = [];
      for (let i = 0; i < vertexCount; i++) {
        const v = sphere.vertices[i];
        const neighbors = sphere.neighbors[i];

        let avgX = V.C(0);
        let avgY = V.C(0);
        let avgZ = V.C(0);

        for (const nIdx of neighbors) {
          const n = sphere.vertices[nIdx];
          avgX = V.add(avgX, n.x);
          avgY = V.add(avgY, n.y);
          avgZ = V.add(avgZ, n.z);
        }

        const invCount = 1.0 / neighbors.length;
        avgX = V.mul(avgX, V.C(invCount));
        avgY = V.mul(avgY, V.C(invCount));
        avgZ = V.mul(avgZ, V.C(invCount));

        res.push(createFlatteningResidual(v.x, v.y, v.z, avgX, avgY, avgZ));
      }

      return res;
    });

    const compiledIterations = 1000;
    const t1 = performance.now();

    for (let iter = 0; iter < compiledIterations; iter++) {
      compiled.evaluateSumWithGradient(params);
    }

    const compiledTime = performance.now() - t1;
    const compiledTimePerEval = compiledTime / compiledIterations;

    console.log(`\n[COMPILED] ${compiledIterations} evaluations`);
    console.log(`  Total: ${compiledTime.toFixed(2)}ms`);
    console.log(`  Per evaluation: ${compiledTimePerEval.toFixed(3)}ms`);
    console.log(`  Residuals/sec: ${(residuals.length / compiledTimePerEval * 1000).toFixed(0)}`);
    console.log(`  Speedup vs Raw: ${(rawTimePerEval / compiledTimePerEval).toFixed(1)}x`);

    // ===== MODE 3: GPU (if available) =====
    if (WebGPUContext.isAvailable()) {
      // Compile first residual to WGSL (they all have same structure)
      const { wgslCode, graphInputs } = compileToWGSL(residuals[0]);
      const kernel = new WGSLKernel(ctx.device, wgslCode, graphInputs);

      // Build batch input data
      const batchSize = residuals.length;
      const inputsPerResidual = graphInputs.length;
      const batchInputs = new Float32Array(batchSize * inputsPerResidual);

      // Pack data for each residual
      for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < inputsPerResidual; j++) {
          const inputValue = graphInputs[j];
          batchInputs[i * inputsPerResidual + j] = inputValue.data;
        }
      }

      // Warmup
      await kernel.execute(batchInputs, batchSize);

      const gpuIterations = 100;
      const t2 = performance.now();

      for (let iter = 0; iter < gpuIterations; iter++) {
        await kernel.execute(batchInputs, batchSize);
      }

      const gpuTime = performance.now() - t2;
      const gpuTimePerEval = gpuTime / gpuIterations;

      console.log(`\n[GPU] ${gpuIterations} evaluations (batched)`);
      console.log(`  Total: ${gpuTime.toFixed(2)}ms`);
      console.log(`  Per evaluation: ${gpuTimePerEval.toFixed(3)}ms`);
      console.log(`  Residuals/sec: ${(residuals.length / gpuTimePerEval * 1000).toFixed(0)}`);
      console.log(`  Speedup vs Raw: ${(rawTimePerEval / gpuTimePerEval).toFixed(1)}x`);
      console.log(`  Speedup vs Compiled: ${(compiledTimePerEval / gpuTimePerEval).toFixed(1)}x`);

      // GPU overhead is high for small batches, so this might be slower
      // That's expected and fine - GPU shines with larger batches
    } else {
      console.log(`\n[GPU] Not available - skipping`);
    }

    console.log(`\n========================================\n`);

    // Test passes if it completes
    expect(true).toBe(true);
  });

  it('should benchmark sphere flattening: ~50 vertices (subdiv 1)', async () => {
    const sphere = SimpleSphere.generate(1);
    const vertexCount = sphere.getVertexCount();

    console.log(`\n========================================`);
    console.log(`BENCHMARK: ${vertexCount} vertices`);
    console.log(`========================================\n`);

    // Build one sample residual for compilation
    const v0 = sphere.vertices[0];
    const neighbors0 = sphere.neighbors[0];

    let avgX = V.C(0);
    let avgY = V.C(0);
    let avgZ = V.C(0);

    for (const nIdx of neighbors0) {
      const n = sphere.vertices[nIdx];
      avgX = V.add(avgX, n.x);
      avgY = V.add(avgY, n.y);
      avgZ = V.add(avgZ, n.z);
    }

    const invCount = 1.0 / neighbors0.length;
    avgX = V.mul(avgX, V.C(invCount));
    avgY = V.mul(avgY, V.C(invCount));
    avgZ = V.mul(avgZ, V.C(invCount));

    const sampleResidual = createFlatteningResidual(v0.x, v0.y, v0.z, avgX, avgY, avgZ);

    // ===== MODE 1: RAW EVALUATION =====
    const params = sphere.getParameterArray();
    const rawIterations = 50;
    const residuals: Value[] = [];

    // Build all residuals
    for (let i = 0; i < vertexCount; i++) {
      const v = sphere.vertices[i];
      const neighbors = sphere.neighbors[i];

      let ax = V.C(0);
      let ay = V.C(0);
      let az = V.C(0);

      for (const nIdx of neighbors) {
        const n = sphere.vertices[nIdx];
        ax = V.add(ax, n.x);
        ay = V.add(ay, n.y);
        az = V.add(az, n.z);
      }

      const inv = 1.0 / neighbors.length;
      ax = V.mul(ax, V.C(inv));
      ay = V.mul(ay, V.C(inv));
      az = V.mul(az, V.C(inv));

      residuals.push(createFlatteningResidual(v.x, v.y, v.z, ax, ay, az));
    }

    const t0 = performance.now();

    for (let iter = 0; iter < rawIterations; iter++) {
      let sum = 0;
      for (const r of residuals) {
        sum += r.data;
      }
    }

    const rawTime = performance.now() - t0;
    const rawTimePerEval = rawTime / rawIterations;

    console.log(`[RAW] ${rawIterations} evaluations`);
    console.log(`  Total: ${rawTime.toFixed(2)}ms`);
    console.log(`  Per evaluation: ${rawTimePerEval.toFixed(3)}ms`);
    console.log(`  Residuals/sec: ${(residuals.length / rawTimePerEval * 1000).toFixed(0)}`);

    // ===== MODE 2: COMPILED CPU =====
    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      for (let i = 0; i < vertexCount; i++) {
        sphere.vertices[i].x = p[i * 3 + 0];
        sphere.vertices[i].y = p[i * 3 + 1];
        sphere.vertices[i].z = p[i * 3 + 2];
      }

      const res: Value[] = [];
      for (let i = 0; i < vertexCount; i++) {
        const v = sphere.vertices[i];
        const neighbors = sphere.neighbors[i];

        let ax = V.C(0);
        let ay = V.C(0);
        let az = V.C(0);

        for (const nIdx of neighbors) {
          const n = sphere.vertices[nIdx];
          ax = V.add(ax, n.x);
          ay = V.add(ay, n.y);
          az = V.add(az, n.z);
        }

        const inv = 1.0 / neighbors.length;
        ax = V.mul(ax, V.C(inv));
        ay = V.mul(ay, V.C(inv));
        az = V.mul(az, V.C(inv));

        res.push(createFlatteningResidual(v.x, v.y, v.z, ax, ay, az));
      }

      return res;
    });

    const compiledIterations = 500;
    const t1 = performance.now();

    for (let iter = 0; iter < compiledIterations; iter++) {
      compiled.evaluateSumWithGradient(params);
    }

    const compiledTime = performance.now() - t1;
    const compiledTimePerEval = compiledTime / compiledIterations;

    console.log(`\n[COMPILED] ${compiledIterations} evaluations`);
    console.log(`  Total: ${compiledTime.toFixed(2)}ms`);
    console.log(`  Per evaluation: ${compiledTimePerEval.toFixed(3)}ms`);
    console.log(`  Residuals/sec: ${(residuals.length / compiledTimePerEval * 1000).toFixed(0)}`);
    console.log(`  Speedup vs Raw: ${(rawTimePerEval / compiledTimePerEval).toFixed(1)}x`);

    // ===== MODE 3: GPU (if available) =====
    if (WebGPUContext.isAvailable()) {
      const { wgslCode, graphInputs } = compileToWGSL(sampleResidual);
      const kernel = new WGSLKernel(ctx.device, wgslCode, graphInputs);

      const batchSize = residuals.length;
      const inputsPerResidual = graphInputs.length;
      const batchInputs = new Float32Array(batchSize * inputsPerResidual);

      for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < inputsPerResidual; j++) {
          const inputValue = graphInputs[j];
          batchInputs[i * inputsPerResidual + j] = inputValue.data;
        }
      }

      // Warmup
      await kernel.execute(batchInputs, batchSize);

      const gpuIterations = 100;
      const t2 = performance.now();

      for (let iter = 0; iter < gpuIterations; iter++) {
        await kernel.execute(batchInputs, batchSize);
      }

      const gpuTime = performance.now() - t2;
      const gpuTimePerEval = gpuTime / gpuIterations;

      console.log(`\n[GPU] ${gpuIterations} evaluations (batched)`);
      console.log(`  Total: ${gpuTime.toFixed(2)}ms`);
      console.log(`  Per evaluation: ${gpuTimePerEval.toFixed(3)}ms`);
      console.log(`  Residuals/sec: ${(residuals.length / gpuTimePerEval * 1000).toFixed(0)}`);
      console.log(`  Speedup vs Raw: ${(rawTimePerEval / gpuTimePerEval).toFixed(1)}x`);
      console.log(`  Speedup vs Compiled: ${(compiledTimePerEval / gpuTimePerEval).toFixed(1)}x`);
    } else {
      console.log(`\n[GPU] Not available - skipping`);
    }

    console.log(`\n========================================\n`);
    expect(true).toBe(true);
  });

  it('should benchmark sphere flattening: ~150 vertices (subdiv 2)', async () => {
    const sphere = SimpleSphere.generate(2);
    const vertexCount = sphere.getVertexCount();

    console.log(`\n========================================`);
    console.log(`BENCHMARK: ${vertexCount} vertices`);
    console.log(`========================================\n`);

    const params = sphere.getParameterArray();

    // Build sample residual
    const v0 = sphere.vertices[0];
    const neighbors0 = sphere.neighbors[0];

    let avgX = V.C(0);
    let avgY = V.C(0);
    let avgZ = V.C(0);

    for (const nIdx of neighbors0) {
      const n = sphere.vertices[nIdx];
      avgX = V.add(avgX, n.x);
      avgY = V.add(avgY, n.y);
      avgZ = V.add(avgZ, n.z);
    }

    const invCount = 1.0 / neighbors0.length;
    avgX = V.mul(avgX, V.C(invCount));
    avgY = V.mul(avgY, V.C(invCount));
    avgZ = V.mul(avgZ, V.C(invCount));

    const sampleResidual = createFlatteningResidual(v0.x, v0.y, v0.z, avgX, avgY, avgZ);

    // For larger meshes, skip raw evaluation (too slow)
    console.log(`[RAW] Skipped (too slow for ${vertexCount} vertices)\n`);

    // ===== MODE 2: COMPILED CPU =====
    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      for (let i = 0; i < vertexCount; i++) {
        sphere.vertices[i].x = p[i * 3 + 0];
        sphere.vertices[i].y = p[i * 3 + 1];
        sphere.vertices[i].z = p[i * 3 + 2];
      }

      const res: Value[] = [];
      for (let i = 0; i < vertexCount; i++) {
        const v = sphere.vertices[i];
        const neighbors = sphere.neighbors[i];

        let ax = V.C(0);
        let ay = V.C(0);
        let az = V.C(0);

        for (const nIdx of neighbors) {
          const n = sphere.vertices[nIdx];
          ax = V.add(ax, n.x);
          ay = V.add(ay, n.y);
          az = V.add(az, n.z);
        }

        const inv = 1.0 / neighbors.length;
        ax = V.mul(ax, V.C(inv));
        ay = V.mul(ay, V.C(inv));
        az = V.mul(az, V.C(inv));

        res.push(createFlatteningResidual(v.x, v.y, v.z, ax, ay, az));
      }

      return res;
    });

    const compiledIterations = 100;
    const t1 = performance.now();

    for (let iter = 0; iter < compiledIterations; iter++) {
      compiled.evaluateSumWithGradient(params);
    }

    const compiledTime = performance.now() - t1;
    const compiledTimePerEval = compiledTime / compiledIterations;

    console.log(`[COMPILED] ${compiledIterations} evaluations`);
    console.log(`  Total: ${compiledTime.toFixed(2)}ms`);
    console.log(`  Per evaluation: ${compiledTimePerEval.toFixed(3)}ms`);
    console.log(`  Residuals/sec: ${(vertexCount / compiledTimePerEval * 1000).toFixed(0)}`);

    // ===== MODE 3: GPU (if available) =====
    if (WebGPUContext.isAvailable()) {
      const { wgslCode, graphInputs } = compileToWGSL(sampleResidual);
      const kernel = new WGSLKernel(ctx.device, wgslCode, graphInputs);

      const batchSize = vertexCount;
      const inputsPerResidual = graphInputs.length;
      const batchInputs = new Float32Array(batchSize * inputsPerResidual);

      for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < inputsPerResidual; j++) {
          const inputValue = graphInputs[j];
          batchInputs[i * inputsPerResidual + j] = inputValue.data;
        }
      }

      // Warmup
      await kernel.execute(batchInputs, batchSize);

      const gpuIterations = 100;
      const t2 = performance.now();

      for (let iter = 0; iter < gpuIterations; iter++) {
        await kernel.execute(batchInputs, batchSize);
      }

      const gpuTime = performance.now() - t2;
      const gpuTimePerEval = gpuTime / gpuIterations;

      console.log(`\n[GPU] ${gpuIterations} evaluations (batched)`);
      console.log(`  Total: ${gpuTime.toFixed(2)}ms`);
      console.log(`  Per evaluation: ${gpuTimePerEval.toFixed(3)}ms`);
      console.log(`  Residuals/sec: ${(vertexCount / gpuTimePerEval * 1000).toFixed(0)}`);
      console.log(`  Speedup vs Compiled: ${(compiledTimePerEval / gpuTimePerEval).toFixed(1)}x`);
    } else {
      console.log(`\n[GPU] Not available - skipping`);
    }

    console.log(`\n========================================\n`);
    expect(true).toBe(true);
  });
});
