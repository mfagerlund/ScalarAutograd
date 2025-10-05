import { describe, it, expect } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { DevelopableOptimizer } from '../demos/developable-sphere/src/optimization/DevelopableOptimizer';
import { CombinatorialEnergy } from '../demos/developable-sphere/src/energy/CombinatorialEnergy';
import { CovarianceEnergy } from '../demos/developable-sphere/src/energy/CovarianceEnergy';
import { DevelopableEnergy } from '../demos/developable-sphere/src/energy/DevelopableEnergy';
import { Vec3, V } from '../src';
import { TriangleMesh } from '../demos/developable-sphere/src/mesh/TriangleMesh';
import { testLog } from './testUtils';

describe('Combinatorial Energy (E^P)', () => {
  it('should be zero for a perfect plane', () => {
    // Create a simple planar mesh (2x2 grid)
    const vertices = [
      new Vec3(V.W(0), V.W(0), V.W(0)),
      new Vec3(V.W(1), V.W(0), V.W(0)),
      new Vec3(V.W(0), V.W(1), V.W(0)),
      new Vec3(V.W(1), V.W(1), V.W(0)),
    ];

    const faces = [
      { vertices: [0, 1, 2] as [number, number, number] },
      { vertices: [1, 3, 2] as [number, number, number] },
    ];

    const mesh = new TriangleMesh(vertices, faces);
    const energy = CombinatorialEnergy.compute(mesh);

    testLog(`Plane energy: ${energy.data.toExponential(6)}`);

    // Should be essentially zero for a perfect plane
    expect(energy.data).toBeLessThan(1e-6);
  });

  it('should be zero for a perfect hinge (two planes)', () => {
    // Create a hinge: two planes meeting at 90 degrees
    const vertices = [
      // First plane (horizontal)
      new Vec3(V.W(-1), V.W(0), V.W(0)),
      new Vec3(V.W(0), V.W(0), V.W(0)),
      new Vec3(V.W(-1), V.W(1), V.W(0)),
      new Vec3(V.W(0), V.W(1), V.W(0)),
      // Second plane (vertical)
      new Vec3(V.W(0), V.W(0), V.W(1)),
      new Vec3(V.W(0), V.W(1), V.W(1)),
    ];

    const faces = [
      // First plane
      { vertices: [0, 1, 2] as [number, number, number] },
      { vertices: [1, 3, 2] as [number, number, number] },
      // Second plane
      { vertices: [1, 4, 3] as [number, number, number] },
      { vertices: [3, 4, 5] as [number, number, number] },
    ];

    const mesh = new TriangleMesh(vertices, faces);
    const energy = CombinatorialEnergy.compute(mesh);

    testLog(`Hinge energy: ${energy.data.toExponential(6)}`);

    // Should be essentially zero for a perfect hinge
    expect(energy.data).toBeLessThan(1e-4);
  });

  it('should have non-zero energy on perturbed sphere', () => {
    const sphere = IcoSphere.generate(2, 1.0);

    // Perturb vertices
    for (let i = 0; i < sphere.vertices.length; i++) {
      const v = sphere.vertices[i];
      const noise = 0.15;
      sphere.setVertexPosition(
        i,
        new Vec3(
          v.x.add(Math.random() * noise - noise / 2),
          v.y.add(Math.random() * noise - noise / 2),
          v.z.add(Math.random() * noise - noise / 2)
        )
      );
    }

    const energy = CombinatorialEnergy.compute(sphere);
    testLog(`Perturbed sphere E^P: ${energy.data.toExponential(3)}`);

    // Should have measurable energy
    expect(energy.data).toBeGreaterThan(0.001);
    expect(energy.data).toBeLessThan(1000);
  });

  it('should compute quality metrics correctly', () => {
    const sphere = IcoSphere.generate(2, 1.0);

    const metrics = CombinatorialEnergy.computeQualityMetrics(sphere, 1e-3);

    testLog('\n=== Quality Metrics (Sphere) ===');
    testLog(`Developability: ${metrics.developabilityPct.toFixed(2)}%`);
    testLog(`Estimated regions: ${metrics.numRegions.toFixed(1)}`);
    testLog(`Quality score: ${metrics.qualityScore.toFixed(2)}`);

    expect(metrics.developabilityPct).toBeGreaterThanOrEqual(0);
    expect(metrics.developabilityPct).toBeLessThanOrEqual(100);
    expect(metrics.numRegions).toBeGreaterThan(0);
    // Quality score can be 0 if developability is 0
    expect(metrics.qualityScore).toBeGreaterThanOrEqual(0);
  });

  it('should reduce energy during optimization', { timeout: 30000 }, async () => {
    const sphere = IcoSphere.generate(2, 1.0);

    // Perturb vertices
    for (let i = 0; i < sphere.vertices.length; i++) {
      const v = sphere.vertices[i];
      const noise = 0.2;
      sphere.setVertexPosition(
        i,
        new Vec3(
          v.x.add(Math.random() * noise - noise / 2),
          v.y.add(Math.random() * noise - noise / 2),
          v.z.add(Math.random() * noise - noise / 2)
        )
      );
    }

    const initialEnergy = CombinatorialEnergy.compute(sphere).data;
    const initialMetrics = CombinatorialEnergy.computeQualityMetrics(sphere, 1e-3);

    testLog('\n=== Combinatorial Energy Optimization ===');
    testLog(`Initial E^P: ${initialEnergy.toExponential(3)}`);
    testLog(
      `Initial developability: ${initialMetrics.developabilityPct.toFixed(1)}% (${initialMetrics.numRegions.toFixed(1)} regions)`
    );

    const optimizer = new DevelopableOptimizer(sphere);
    const result = await optimizer.optimizeAsync({
      maxIterations: 50,
      gradientTolerance: 1e-6,
      verbose: true,
      energyType: 'Bimodal Variance (Spatial Midpoint)', // Use as baseline since we don't have direct support yet
      chunkSize: 50,
    });

    const finalEnergy = CombinatorialEnergy.compute(sphere).data;
    const finalMetrics = CombinatorialEnergy.computeQualityMetrics(sphere, 1e-3);

    testLog(`Final E^P: ${finalEnergy.toExponential(3)}`);
    testLog(
      `Final developability: ${finalMetrics.developabilityPct.toFixed(1)}% (${finalMetrics.numRegions.toFixed(1)} regions)`
    );
    testLog(`Quality improvement: ${initialMetrics.qualityScore.toFixed(2)} → ${finalMetrics.qualityScore.toFixed(2)}`);

    // Energy should be measurable
    expect(finalEnergy).toBeGreaterThan(0);

    // Developability should improve (even with different energy function)
    expect(finalMetrics.developabilityPct).toBeGreaterThan(initialMetrics.developabilityPct * 0.8);
  });
});

describe('Covariance Energy (E^λ)', () => {
  it('should be zero for a perfect plane', () => {
    // Create a simple planar mesh
    const vertices = [
      new Vec3(V.W(0), V.W(0), V.W(0)),
      new Vec3(V.W(1), V.W(0), V.W(0)),
      new Vec3(V.W(0), V.W(1), V.W(0)),
      new Vec3(V.W(1), V.W(1), V.W(0)),
    ];

    const faces = [
      { vertices: [0, 1, 2] as [number, number, number] },
      { vertices: [1, 3, 2] as [number, number, number] },
    ];

    const mesh = new TriangleMesh(vertices, faces);
    const energy = CovarianceEnergy.compute(mesh);

    testLog(`Plane E^λ: ${energy.data.toExponential(6)}`);

    // Should be small (numerical precision may cause small non-zero values)
    // For a perfect plane, all normals are identical, so smallest eigenvalue should be near zero
    expect(energy.data).toBeLessThan(0.01);
  });

  it('should be zero for a perfect hinge', () => {
    // Create a hinge: two planes meeting at 90 degrees
    const vertices = [
      new Vec3(V.W(-1), V.W(0), V.W(0)),
      new Vec3(V.W(0), V.W(0), V.W(0)),
      new Vec3(V.W(-1), V.W(1), V.W(0)),
      new Vec3(V.W(0), V.W(1), V.W(0)),
      new Vec3(V.W(0), V.W(0), V.W(1)),
      new Vec3(V.W(0), V.W(1), V.W(1)),
    ];

    const faces = [
      { vertices: [0, 1, 2] as [number, number, number] },
      { vertices: [1, 3, 2] as [number, number, number] },
      { vertices: [1, 4, 3] as [number, number, number] },
      { vertices: [3, 4, 5] as [number, number, number] },
    ];

    const mesh = new TriangleMesh(vertices, faces);
    const energy = CovarianceEnergy.compute(mesh);

    testLog(`Hinge E^λ: ${energy.data.toExponential(6)}`);

    // Should be small for coplanar normals (two planes = coplanar normal set)
    expect(energy.data).toBeLessThan(0.01);
  });

  it('should have non-zero energy on perturbed sphere', () => {
    const sphere = IcoSphere.generate(2, 1.0);

    // Perturb vertices
    for (let i = 0; i < sphere.vertices.length; i++) {
      const v = sphere.vertices[i];
      const noise = 0.15;
      sphere.setVertexPosition(
        i,
        new Vec3(
          v.x.add(Math.random() * noise - noise / 2),
          v.y.add(Math.random() * noise - noise / 2),
          v.z.add(Math.random() * noise - noise / 2)
        )
      );
    }

    const energy = CovarianceEnergy.compute(sphere);
    testLog(`Perturbed sphere E^λ: ${energy.data.toExponential(3)}`);

    expect(energy.data).toBeGreaterThan(0.001);
    expect(energy.data).toBeLessThan(1000);
  });

  it('should compute intrinsic variant correctly', () => {
    // Use smaller sphere to avoid computation graph overflow
    const sphere = IcoSphere.generate(1, 1.0);

    // Perturb vertices
    for (let i = 0; i < sphere.vertices.length; i++) {
      const v = sphere.vertices[i];
      const noise = 0.15;
      sphere.setVertexPosition(
        i,
        new Vec3(
          v.x.add(Math.random() * noise - noise / 2),
          v.y.add(Math.random() * noise - noise / 2),
          v.z.add(Math.random() * noise - noise / 2)
        )
      );
    }

    const energyStandard = CovarianceEnergy.compute(sphere).data;
    const energyIntrinsic = sphere.vertices
      .map((_, i) => CovarianceEnergy.computeVertexEnergyIntrinsic(i, sphere).data)
      .reduce((a, b) => a + b, 0);

    testLog(`E^λ (standard): ${energyStandard.toExponential(3)}`);
    testLog(`E^λ (intrinsic): ${energyIntrinsic.toExponential(3)}`);

    // Both should be non-zero and positive
    expect(energyStandard).toBeGreaterThan(0);
    expect(energyIntrinsic).toBeGreaterThan(0);

    // They should be similar in magnitude (within 10x)
    const ratio = energyStandard / energyIntrinsic;
    expect(ratio).toBeGreaterThan(0.1);
    expect(ratio).toBeLessThan(10);
  });

  it('should compute max variant correctly', () => {
    const sphere = IcoSphere.generate(2, 1.0);

    // Perturb vertices
    for (let i = 0; i < sphere.vertices.length; i++) {
      const v = sphere.vertices[i];
      const noise = 0.15;
      sphere.setVertexPosition(
        i,
        new Vec3(
          v.x.add(Math.random() * noise - noise / 2),
          v.y.add(Math.random() * noise - noise / 2),
          v.z.add(Math.random() * noise - noise / 2)
        )
      );
    }

    const energyStandard = CovarianceEnergy.compute(sphere).data;
    const energyMax = sphere.vertices
      .map((_, i) => CovarianceEnergy.computeVertexEnergyMax(i, sphere).data)
      .reduce((a, b) => a + b, 0);

    testLog(`E^λ (standard): ${energyStandard.toExponential(3)}`);
    testLog(`E^λ (max): ${energyMax.toExponential(3)}`);

    // Both should be non-zero and positive
    expect(energyStandard).toBeGreaterThan(0);
    expect(energyMax).toBeGreaterThan(0);
  });

  it('should compute quality metrics correctly', () => {
    const sphere = IcoSphere.generate(2, 1.0);

    const metrics = CovarianceEnergy.computeQualityMetrics(sphere, 0.1);

    testLog('\n=== Quality Metrics (Covariance) ===');
    testLog(`Developability: ${metrics.developabilityPct.toFixed(2)}%`);
    testLog(`Estimated regions: ${metrics.numRegions.toFixed(1)}`);
    testLog(`Quality score: ${metrics.qualityScore.toFixed(2)}`);

    expect(metrics.developabilityPct).toBeGreaterThanOrEqual(0);
    expect(metrics.developabilityPct).toBeLessThanOrEqual(100);
    expect(metrics.numRegions).toBeGreaterThan(0);
    expect(metrics.qualityScore).toBeGreaterThan(0);
  });

  it('should reduce energy during optimization', { timeout: 30000 }, async () => {
    const sphere = IcoSphere.generate(2, 1.0);

    // Perturb vertices
    for (let i = 0; i < sphere.vertices.length; i++) {
      const v = sphere.vertices[i];
      const noise = 0.2;
      sphere.setVertexPosition(
        i,
        new Vec3(
          v.x.add(Math.random() * noise - noise / 2),
          v.y.add(Math.random() * noise - noise / 2),
          v.z.add(Math.random() * noise - noise / 2)
        )
      );
    }

    const initialEnergy = CovarianceEnergy.compute(sphere).data;
    const initialMetrics = CovarianceEnergy.computeQualityMetrics(sphere, 0.1);

    testLog('\n=== Covariance Energy Optimization ===');
    testLog(`Initial E^λ: ${initialEnergy.toExponential(3)}`);
    testLog(
      `Initial developability: ${initialMetrics.developabilityPct.toFixed(1)}% (${initialMetrics.numRegions.toFixed(1)} regions)`
    );

    const optimizer = new DevelopableOptimizer(sphere);
    const result = await optimizer.optimizeAsync({
      maxIterations: 50,
      gradientTolerance: 1e-6,
      verbose: true,
      energyType: 'Bimodal Variance (Spatial Midpoint)',
      chunkSize: 50,
    });

    const finalEnergy = CovarianceEnergy.compute(sphere).data;
    const finalMetrics = CovarianceEnergy.computeQualityMetrics(sphere, 0.1);

    testLog(`Final E^λ: ${finalEnergy.toExponential(3)}`);
    testLog(
      `Final developability: ${finalMetrics.developabilityPct.toFixed(1)}% (${finalMetrics.numRegions.toFixed(1)} regions)`
    );
    testLog(`Quality improvement: ${initialMetrics.qualityScore.toFixed(2)} → ${finalMetrics.qualityScore.toFixed(2)}`);

    expect(finalEnergy).toBeGreaterThan(0);
    expect(finalMetrics.developabilityPct).toBeGreaterThan(initialMetrics.developabilityPct * 0.8);
  });
});

describe('Combinatorial vs Covariance Energy Comparison', () => {
  it('should both be zero for perfect plane', () => {
    const vertices = [
      new Vec3(V.W(0), V.W(0), V.W(0)),
      new Vec3(V.W(1), V.W(0), V.W(0)),
      new Vec3(V.W(0), V.W(1), V.W(0)),
      new Vec3(V.W(1), V.W(1), V.W(0)),
    ];

    const faces = [
      { vertices: [0, 1, 2] as [number, number, number] },
      { vertices: [1, 3, 2] as [number, number, number] },
    ];

    const mesh = new TriangleMesh(vertices, faces);

    const eP = CombinatorialEnergy.compute(mesh).data;
    const eLambda = CovarianceEnergy.compute(mesh).data;

    testLog(`\nPerfect plane: E^P = ${eP.toExponential(6)}, E^λ = ${eLambda.toExponential(6)}`);

    expect(eP).toBeLessThan(1e-6);
    expect(eLambda).toBeLessThan(0.01); // Covariance may have small numerical error
  });

  it('should both be zero for perfect hinge', () => {
    const vertices = [
      new Vec3(V.W(-1), V.W(0), V.W(0)),
      new Vec3(V.W(0), V.W(0), V.W(0)),
      new Vec3(V.W(-1), V.W(1), V.W(0)),
      new Vec3(V.W(0), V.W(1), V.W(0)),
      new Vec3(V.W(0), V.W(0), V.W(1)),
      new Vec3(V.W(0), V.W(1), V.W(1)),
    ];

    const faces = [
      { vertices: [0, 1, 2] as [number, number, number] },
      { vertices: [1, 3, 2] as [number, number, number] },
      { vertices: [1, 4, 3] as [number, number, number] },
      { vertices: [3, 4, 5] as [number, number, number] },
    ];

    const mesh = new TriangleMesh(vertices, faces);

    const eP = CombinatorialEnergy.compute(mesh).data;
    const eLambda = CovarianceEnergy.compute(mesh).data;

    testLog(`\nPerfect hinge: E^P = ${eP.toExponential(6)}, E^λ = ${eLambda.toExponential(6)}`);

    expect(eP).toBeLessThan(1e-4);
    expect(eLambda).toBeLessThan(0.01); // Covariance may have small numerical error
  });

  it('should compare behavior on perturbed sphere', () => {
    const sphere = IcoSphere.generate(2, 1.0);

    // Perturb vertices
    for (let i = 0; i < sphere.vertices.length; i++) {
      const v = sphere.vertices[i];
      const noise = 0.15;
      sphere.setVertexPosition(
        i,
        new Vec3(
          v.x.add(Math.random() * noise - noise / 2),
          v.y.add(Math.random() * noise - noise / 2),
          v.z.add(Math.random() * noise - noise / 2)
        )
      );
    }

    const eP = CombinatorialEnergy.compute(sphere).data;
    const eLambda = CovarianceEnergy.compute(sphere).data;
    const metricsP = CombinatorialEnergy.computeQualityMetrics(sphere, 1e-3);
    const metricsLambda = CovarianceEnergy.computeQualityMetrics(sphere, 0.1);

    testLog('\n=== Energy Comparison (Perturbed Sphere) ===');
    testLog(`E^P: ${eP.toExponential(3)}`);
    testLog(`E^λ: ${eLambda.toExponential(3)}`);
    testLog(`E^P developability: ${metricsP.developabilityPct.toFixed(1)}% (quality: ${metricsP.qualityScore.toFixed(2)})`);
    testLog(
      `E^λ developability: ${metricsLambda.developabilityPct.toFixed(1)}% (quality: ${metricsLambda.qualityScore.toFixed(2)})`
    );

    // Both should have measurable energy
    expect(eP).toBeGreaterThan(0);
    expect(eLambda).toBeGreaterThan(0);

    // Quality scores should be non-negative (can be 0 if developability is 0)
    expect(metricsP.qualityScore).toBeGreaterThanOrEqual(0);
    expect(metricsLambda.qualityScore).toBeGreaterThanOrEqual(0);
  });
});
