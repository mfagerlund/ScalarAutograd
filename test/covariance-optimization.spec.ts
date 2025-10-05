import { describe, it, expect } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { DevelopableOptimizer } from '../demos/developable-sphere/src/optimization/DevelopableOptimizer';
import { CovarianceEnergy } from '../demos/developable-sphere/src/energy/CovarianceEnergy';
import { CurvatureClassifier } from '../demos/developable-sphere/src/energy/CurvatureClassifier';
import { Vec3 } from '../src/Vec3';
import { testLog } from './testUtils';

describe('Covariance Energy Optimization Test', () => {
  it('should produce good regions with improved covariance energy', { timeout: 180000 }, async () => {
    testLog('\n=== Covariance Energy Optimization Test ===');

    // Create sphere with perturbation (like a real optimization would start)
    const sphere = IcoSphere.generate(2, 1.0);

    // Add small perturbation to break symmetry
    for (let i = 0; i < sphere.vertices.length; i++) {
      const v = sphere.vertices[i];
      const noise = 0.1;
      sphere.setVertexPosition(
        i,
        new Vec3(
          v.x.add(Math.random() * noise - noise / 2),
          v.y.add(Math.random() * noise - noise / 2),
          v.z.add(Math.random() * noise - noise / 2)
        )
      );
    }

    testLog(`Mesh: ${sphere.vertices.length} vertices, ${sphere.faces.length} faces`);

    // Initial metrics
    const initialEnergy = CovarianceEnergy.compute(sphere).data;
    const initialClassification = CurvatureClassifier.classifyVertices(sphere);
    const initialDev = (initialClassification.hingeVertices.length / sphere.vertices.length) * 100;
    const initialRegions = CurvatureClassifier.countDevelopableRegions(sphere);

    testLog(`\nInitial State:`);
    testLog(`  Covariance Energy: ${initialEnergy.toExponential(3)}`);
    testLog(`  Developability: ${initialDev.toFixed(1)}%`);
    testLog(`  Regions: ${initialRegions}`);

    // Optimize (non-compiled for debugging)
    testLog(`\nOptimizing with Covariance Energy (non-compiled)...`);
    const optimizer = new DevelopableOptimizer(sphere);

    const result = await optimizer.optimizeAsync({
      maxIterations: 50,
      gradientTolerance: 1e-6,
      verbose: true,
      energyType: 'Covariance (Smallest Eigenvalue)',
      useCompiled: false,
      chunkSize: 50,
    });

    // Final metrics
    const finalMesh = result.history[result.history.length - 1];
    const finalEnergy = CovarianceEnergy.compute(finalMesh).data;
    const finalClassification = CurvatureClassifier.classifyVertices(finalMesh);
    const finalDev = (finalClassification.hingeVertices.length / finalMesh.vertices.length) * 100;
    const finalRegions = CurvatureClassifier.countDevelopableRegions(finalMesh);

    testLog(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    testLog(`RESULTS:`);
    testLog(`  Iterations: ${result.iterations}`);
    testLog(`  Convergence: ${result.convergenceReason}`);
    testLog(`\nEnergy:`);
    testLog(`  Before: ${initialEnergy.toExponential(3)}`);
    testLog(`  After:  ${finalEnergy.toExponential(3)}`);
    testLog(`  Reduction: ${((1 - finalEnergy / initialEnergy) * 100).toFixed(1)}%`);
    testLog(`\nDevelopability:`);
    testLog(`  Before: ${initialDev.toFixed(1)}%`);
    testLog(`  After:  ${finalDev.toFixed(1)}%`);
    testLog(`  Improvement: +${(finalDev - initialDev).toFixed(1)}%`);
    testLog(`\nRegions:`);
    testLog(`  Before: ${initialRegions}`);
    testLog(`  After:  ${finalRegions}`);
    testLog(`  Quality: ${finalDev.toFixed(1)}% / ${finalRegions} = ${(finalDev / finalRegions).toFixed(1)}% per region`);
    testLog(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`);

    // Expectations
    expect(finalEnergy).toBeLessThan(initialEnergy);
    expect(finalDev).toBeGreaterThan(initialDev);

    // Check for fragmentation - should have reasonable number of regions
    // For a sphere approximation with ~8-12 faces, we expect ~8-16 regions
    testLog(`Region count: ${finalRegions} (target: 8-20 for good quality)`);

    if (finalRegions > 40) {
      console.warn(`⚠ WARNING: High fragmentation - ${finalRegions} regions suggests single-triangle regions`);
    } else if (finalRegions < 20) {
      testLog(`✓ Good region count: ${finalRegions} regions`);
    }

    // The quality metric should be reasonable (>1% per region ideally)
    const qualityPerRegion = finalDev / finalRegions;
    testLog(`Quality per region: ${qualityPerRegion.toFixed(1)}%`);

    expect(qualityPerRegion).toBeGreaterThan(0.5); // At least some developability per region
  });
});
