import { describe, it, expect } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { DevelopableOptimizer } from '../demos/developable-sphere/src/optimization/DevelopableOptimizer';
import { EigenProxyEnergy } from '../demos/developable-sphere/src/energy/EigenProxyEnergy';
import { DevelopableEnergy } from '../demos/developable-sphere/src/energy/DevelopableEnergy';
import { Vec3 } from '../src/Vec3';
import { testLog } from './testUtils';

describe('Eigenvalue Proxy Energy', () => {
  it('should have non-zero energy on perturbed sphere', () => {
    const sphere = IcoSphere.generate(2, 1.0);

    // Perturb vertices to create area distribution
    for (let i = 0; i < sphere.vertices.length; i++) {
      const v = sphere.vertices[i];
      const noise = 0.15;
      sphere.setVertexPosition(i, new Vec3(
        v.x.add(Math.random() * noise - noise/2),
        v.y.add(Math.random() * noise - noise/2),
        v.z.add(Math.random() * noise - noise/2)
      ));
    }

    const energy = EigenProxyEnergy.compute(sphere);
    testLog(`Initial perturbed energy: ${energy.data.toExponential(3)}`);

    // Should have measurable energy (not zero, not too huge)
    expect(energy.data).toBeGreaterThan(0.001);
    expect(energy.data).toBeLessThan(100);
  });

  it('should run multiple iterations and reduce energy', { timeout: 15000 }, async () => {
    const sphere = IcoSphere.generate(2, 1.0);

    // Perturb vertices
    for (let i = 0; i < sphere.vertices.length; i++) {
      const v = sphere.vertices[i];
      const noise = 0.2;
      sphere.setVertexPosition(i, new Vec3(
        v.x.add(Math.random() * noise - noise/2),
        v.y.add(Math.random() * noise - noise/2),
        v.z.add(Math.random() * noise - noise/2)
      ));
    }

    const initialEnergy = EigenProxyEnergy.compute(sphere).data;
    const initialVariance = DevelopableEnergy.compute(sphere).data;

    testLog('\n=== Eigenvalue Proxy Energy Test ===');
    testLog(`Initial eigen-proxy energy: ${initialEnergy.toExponential(3)}`);
    testLog(`Initial variance: ${initialVariance.toExponential(3)}`);

    const optimizer = new DevelopableOptimizer(sphere);
    const result = await optimizer.optimizeAsync({
      maxIterations: 50,
      gradientTolerance: 1e-6,
      verbose: true,
      energyType: 'Eigenvalue Proxy (Trace - Frobenius)',
      chunkSize: 50,
    });

    const finalEnergy = EigenProxyEnergy.compute(sphere).data;
    const finalVariance = DevelopableEnergy.compute(sphere).data;

    testLog(`\nFinal eigen-proxy energy: ${finalEnergy.toExponential(3)}`);
    testLog(`Final variance: ${finalVariance.toExponential(3)}`);
    testLog(`Iterations: ${result.iterations}`);
    testLog(`Convergence: ${result.convergenceReason}`);

    // Should run at least a few iterations
    expect(result.iterations).toBeGreaterThan(3);

    // Should reduce energy significantly
    expect(finalEnergy).toBeLessThan(initialEnergy * 0.8);

    // Should improve developability (reduce variance)
    expect(finalVariance).toBeLessThan(initialVariance);
  });

  it('should compare eigenproxy with variance and bounding box', { timeout: 30000 }, async () => {
    const energyTypes = ['Bimodal Variance (Spatial Midpoint)', 'Bounding Box Spread', 'Eigenvalue Proxy (Trace - Frobenius)'];
    const results: Record<string, any> = {};

    for (const energyType of energyTypes) {
      // Create fresh perturbed sphere for each test
      const sphere = IcoSphere.generate(2, 1.0);

      for (let i = 0; i < sphere.vertices.length; i++) {
        const v = sphere.vertices[i];
        const noise = 0.15;
        sphere.setVertexPosition(i, new Vec3(
          v.x.add(Math.random() * noise - noise/2),
          v.y.add(Math.random() * noise - noise/2),
          v.z.add(Math.random() * noise - noise/2)
        ));
      }

      const initialVariance = DevelopableEnergy.compute(sphere).data;
      const initialDev = DevelopableEnergy.classifyVertices(sphere).hingeVertices.length / sphere.vertices.length;

      const optimizer = new DevelopableOptimizer(sphere);
      const result = await optimizer.optimizeAsync({
        maxIterations: 50,
        gradientTolerance: 1e-6,
        verbose: false,
        energyType,
        chunkSize: 50,
      });

      const finalVariance = DevelopableEnergy.compute(sphere).data;
      const finalDev = DevelopableEnergy.classifyVertices(sphere).hingeVertices.length / sphere.vertices.length;

      results[energyType] = {
        initialVariance,
        finalVariance,
        initialDev,
        finalDev,
        iterations: result.iterations,
        convergence: result.convergenceReason,
      };
    }

    testLog('\n=== Energy Function Comparison ===');
    for (const [type, res] of Object.entries(results)) {
      testLog(`\n${type}:`);
      testLog(`  Variance: ${res.initialVariance.toExponential(3)} → ${res.finalVariance.toExponential(3)}`);
      testLog(`  Developable: ${(res.initialDev * 100).toFixed(1)}% → ${(res.finalDev * 100).toFixed(1)}%`);
      testLog(`  Iterations: ${res.iterations}, ${res.convergence}`);
    }

    // All should improve developability and run multiple iterations
    for (const type of energyTypes) {
      expect(results[type].finalDev).toBeGreaterThan(results[type].initialDev);
      expect(results[type].iterations).toBeGreaterThan(3);
    }
  });
});
