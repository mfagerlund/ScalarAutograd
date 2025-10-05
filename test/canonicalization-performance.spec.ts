/**
 * Performance tests for graph canonicalization
 */
import { describe, it, expect } from 'vitest';
import { V, Value } from '../src';
import { testLog } from './testUtils';

describe('Canonicalization Performance', () => {
  it('should canonicalize large sum expression efficiently', () => {
    // Build a large sum: x0 + x1 + x2 + ... + x99
    const params: Value[] = [];
    for (let i = 0; i < 100; i++) {
      params.push(V.W(i));
    }

    let sum = params[0];
    for (let i = 1; i < params.length; i++) {
      sum = V.add(sum, params[i]);
    }

    const start = performance.now();
    const compiled = V.compileObjective(params, () => sum);
    const duration = performance.now() - start;

    testLog(`100-term sum: ${duration.toFixed(0)}ms, ${compiled.kernelCount} kernels`);

    // Should be fast - under 100ms
    expect(duration).toBeLessThan(100);
  });

  it('should canonicalize deep nested expression efficiently', () => {
    // Build nested structure: ((x0 + x1) * (x2 + x3)) + ((x4 + x5) * (x6 + x7)) + ...
    const params: Value[] = [];
    for (let i = 0; i < 64; i++) {
      params.push(V.W(i));
    }

    const terms: Value[] = [];
    for (let i = 0; i < 64; i += 4) {
      const sum1 = V.add(params[i], params[i + 1]);
      const sum2 = V.add(params[i + 2], params[i + 3]);
      const prod = V.mul(sum1, sum2);
      terms.push(prod);
    }

    let result = terms[0];
    for (let i = 1; i < terms.length; i++) {
      result = V.add(result, terms[i]);
    }

    const start = performance.now();
    const compiled = V.compileObjective(params, () => result);
    const duration = performance.now() - start;

    testLog(`Deep nested (64 params, 16 terms): ${duration.toFixed(0)}ms, ${compiled.kernelCount} kernels`);

    // Should be fast
    expect(duration).toBeLessThan(100);
  });

  it('should handle variance-like computation efficiently', () => {
    // Simulate variance calculation like in developable sphere
    // variance = sum((xi - mean)^2) / n
    const n = 30; // Typical vertex star size
    const params: Value[] = [];
    for (let i = 0; i < n; i++) {
      params.push(V.W(Math.random()));
    }

    // Mean = sum(xi) / n
    let sum = params[0];
    for (let i = 1; i < n; i++) {
      sum = V.add(sum, params[i]);
    }
    const mean = V.div(sum, n);

    // Variance = sum((xi - mean)^2) / n
    let variance = V.C(0);
    for (let i = 0; i < n; i++) {
      const diff = V.sub(params[i], mean);
      const sq = V.mul(diff, diff);
      variance = V.add(variance, sq);
    }
    variance = V.div(variance, n);

    const start = performance.now();
    const compiled = V.compileObjective(params, () => variance);
    const duration = performance.now() - start;

    testLog(`Variance (${n} params): ${duration.toFixed(0)}ms, ${compiled.kernelCount} kernels`);

    // Should be very fast - this is the core pattern
    expect(duration).toBeLessThan(50);
  });

  it('should handle multiple variance computations efficiently', () => {
    // Simulate multiple variance calculations like in mesh optimization
    const numRegions = 10;
    const regionSize = 20;

    const params: Value[] = [];
    for (let i = 0; i < numRegions * regionSize; i++) {
      params.push(V.W(Math.random()));
    }

    let totalEnergy = V.C(0);
    for (let r = 0; r < numRegions; r++) {
      const start = r * regionSize;
      const regionParams = params.slice(start, start + regionSize);

      // Mean
      let sum = regionParams[0];
      for (let i = 1; i < regionSize; i++) {
        sum = V.add(sum, regionParams[i]);
      }
      const mean = V.div(sum, regionSize);

      // Variance
      let variance = V.C(0);
      for (let i = 0; i < regionSize; i++) {
        const diff = V.sub(regionParams[i], mean);
        const sq = V.mul(diff, diff);
        variance = V.add(variance, sq);
      }
      variance = V.div(variance, regionSize);

      totalEnergy = V.add(totalEnergy, variance);
    }

    const start = performance.now();
    const compiled = V.compileObjective(params, () => totalEnergy);
    const duration = performance.now() - start;

    testLog(`Multiple variance (${numRegions} regions, ${regionSize} each): ${duration.toFixed(0)}ms, ${compiled.kernelCount} kernels`);

    // This is closer to real mesh optimization
    expect(duration).toBeLessThan(200);
  });

  it('should handle mesh-like energy with reasonable performance', () => {
    // Simulate actual mesh vertex energy computation
    const numVertices = 50;
    const avgNeighbors = 6;

    const params: Value[] = [];
    for (let i = 0; i < numVertices * 3; i++) {
      params.push(V.W(Math.random()));
    }

    let totalEnergy = V.C(0);

    for (let v = 0; v < numVertices; v++) {
      // Get vertex star (simplified - just use neighbors)
      const neighbors: Value[] = [];
      for (let n = 0; n < avgNeighbors; n++) {
        const idx = ((v * avgNeighbors + n) * 3) % params.length;
        neighbors.push(params[idx], params[idx + 1], params[idx + 2]);
      }

      // Compute mean normal (simplified)
      let sumX = V.C(0), sumY = V.C(0), sumZ = V.C(0);
      for (let i = 0; i < neighbors.length; i += 3) {
        sumX = V.add(sumX, neighbors[i]);
        sumY = V.add(sumY, neighbors[i + 1]);
        sumZ = V.add(sumZ, neighbors[i + 2]);
      }
      const meanX = V.div(sumX, avgNeighbors);
      const meanY = V.div(sumY, avgNeighbors);
      const meanZ = V.div(sumZ, avgNeighbors);

      // Variance
      let variance = V.C(0);
      for (let i = 0; i < neighbors.length; i += 3) {
        const dx = V.sub(neighbors[i], meanX);
        const dy = V.sub(neighbors[i + 1], meanY);
        const dz = V.sub(neighbors[i + 2], meanZ);
        const sq = V.add(V.add(V.mul(dx, dx), V.mul(dy, dy)), V.mul(dz, dz));
        variance = V.add(variance, sq);
      }
      variance = V.div(variance, avgNeighbors);

      totalEnergy = V.add(totalEnergy, variance);
    }

    const start = performance.now();
    const compiled = V.compileObjective(params, () => totalEnergy);
    const duration = performance.now() - start;

    testLog(`Mesh-like energy (${numVertices} vertices): ${duration.toFixed(0)}ms, ${compiled.kernelCount} kernels`);

    // This should be reasonable - the real mesh has 162-642 vertices
    expect(duration).toBeLessThan(500);
  });
});
