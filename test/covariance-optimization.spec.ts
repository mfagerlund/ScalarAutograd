import { describe, it, expect } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { DevelopableOptimizer } from '../demos/developable-sphere/src/optimization/DevelopableOptimizer';
import { CovarianceEnergy } from '../demos/developable-sphere/src/energy/CovarianceEnergy';
import { CurvatureClassifier } from '../demos/developable-sphere/src/energy/CurvatureClassifier';
import { Vec3 } from '../src/Vec3';

describe('Covariance Energy Optimization Test', () => {
  it('should produce good regions with improved covariance energy', { timeout: 180000 }, async () => {
    console.log('\n=== Covariance Energy Optimization Test ===');

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

    console.log(`Mesh: ${sphere.vertices.length} vertices, ${sphere.faces.length} faces`);

    // Initial metrics
    const initialEnergy = CovarianceEnergy.compute(sphere).data;
    const initialClassification = CurvatureClassifier.classifyVertices(sphere);
    const initialDev = (initialClassification.hingeVertices.length / sphere.vertices.length) * 100;
    const initialRegions = CurvatureClassifier.countDevelopableRegions(sphere);

    console.log(`\nInitial State:`);
    console.log(`  Covariance Energy: ${initialEnergy.toExponential(3)}`);
    console.log(`  Developability: ${initialDev.toFixed(1)}%`);
    console.log(`  Regions: ${initialRegions}`);

    // Optimize (non-compiled for debugging)
    console.log(`\nOptimizing with Covariance Energy (non-compiled)...`);
    const optimizer = new DevelopableOptimizer(sphere);

    const result = await optimizer.optimizeAsync({
      maxIterations: 50,
      gradientTolerance: 1e-6,
      verbose: true,
      energyType: 'covariance',
      useCompiled: false,
      chunkSize: 50,
    });

    // Final metrics
    const finalMesh = result.history[result.history.length - 1];
    const finalEnergy = CovarianceEnergy.compute(finalMesh).data;
    const finalClassification = CurvatureClassifier.classifyVertices(finalMesh);
    const finalDev = (finalClassification.hingeVertices.length / finalMesh.vertices.length) * 100;
    const finalRegions = CurvatureClassifier.countDevelopableRegions(finalMesh);

    console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`RESULTS:`);
    console.log(`  Iterations: ${result.iterations}`);
    console.log(`  Convergence: ${result.convergenceReason}`);
    console.log(`\nEnergy:`);
    console.log(`  Before: ${initialEnergy.toExponential(3)}`);
    console.log(`  After:  ${finalEnergy.toExponential(3)}`);
    console.log(`  Reduction: ${((1 - finalEnergy / initialEnergy) * 100).toFixed(1)}%`);
    console.log(`\nDevelopability:`);
    console.log(`  Before: ${initialDev.toFixed(1)}%`);
    console.log(`  After:  ${finalDev.toFixed(1)}%`);
    console.log(`  Improvement: +${(finalDev - initialDev).toFixed(1)}%`);
    console.log(`\nRegions:`);
    console.log(`  Before: ${initialRegions}`);
    console.log(`  After:  ${finalRegions}`);
    console.log(`  Quality: ${finalDev.toFixed(1)}% / ${finalRegions} = ${(finalDev / finalRegions).toFixed(1)}% per region`);
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`);

    // Expectations
    expect(finalEnergy).toBeLessThan(initialEnergy);
    expect(finalDev).toBeGreaterThan(initialDev);

    // Check for fragmentation - should have reasonable number of regions
    // For a sphere approximation with ~8-12 faces, we expect ~8-16 regions
    console.log(`Region count: ${finalRegions} (target: 8-20 for good quality)`);

    if (finalRegions > 40) {
      console.warn(`⚠ WARNING: High fragmentation - ${finalRegions} regions suggests single-triangle regions`);
    } else if (finalRegions < 20) {
      console.log(`✓ Good region count: ${finalRegions} regions`);
    }

    // The quality metric should be reasonable (>1% per region ideally)
    const qualityPerRegion = finalDev / finalRegions;
    console.log(`Quality per region: ${qualityPerRegion.toFixed(1)}%`);

    expect(qualityPerRegion).toBeGreaterThan(0.5); // At least some developability per region
  });
});
