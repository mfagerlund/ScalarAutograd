import { describe, it, expect } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { SubdividedMesh } from '../demos/developable-sphere/src/mesh/SubdividedMesh';
import { DevelopableOptimizer } from '../demos/developable-sphere/src/optimization/DevelopableOptimizer';
import { CurvatureClassifier } from '../demos/developable-sphere/src/energy/CurvatureClassifier';
import { CovarianceEnergy } from '../demos/developable-sphere/src/energy/CovarianceEnergy';
import { testLog } from './testUtils';

describe('Multi-Resolution Optimization', () => {
  it.concurrent.skip('should reduce fragmentation compared to single-level optimization', { timeout: 180000 }, async () => {
    testLog('\n=== Multi-Resolution Optimization Test ===\n');

    // ========================================
    // Part 1: Single-level optimization (baseline - should fragment)
    // ========================================
    testLog('--- Baseline: Single-Level Optimization (Level 2) ---');
    const singleLevelMesh = IcoSphere.generate(2, 1.0);
    testLog(`Mesh: ${singleLevelMesh.vertices.length} vertices, ${singleLevelMesh.faces.length} faces`);

    const singleLevelOptimizer = new DevelopableOptimizer(singleLevelMesh);
    const singleLevelResult = await singleLevelOptimizer.optimizeAsync({
      maxIterations: 50,
      gradientTolerance: 1e-6,
      verbose: true,
      energyType: 'Combinatorial (E^P)',
      useCompiled: false,
      chunkSize: 50,
    });

    const singleLevelFinal = singleLevelResult.history[singleLevelResult.history.length - 1];
    const singleLevelClassification = CurvatureClassifier.classifyVertices(singleLevelFinal);
    const singleLevelDev = (singleLevelClassification.hingeVertices.length / singleLevelFinal.vertices.length) * 100;
    const singleLevelRegions = CurvatureClassifier.countDevelopableRegions(singleLevelFinal);

    testLog(`\nSingle-Level Results:`);
    testLog(`  Developability: ${singleLevelDev.toFixed(1)}%`);
    testLog(`  Regions: ${singleLevelRegions}`);
    testLog(`  Quality: ${(singleLevelDev / Math.max(1, singleLevelRegions)).toFixed(1)}% per region`);

    // ========================================
    // Part 2: Multi-resolution optimization (should avoid fragmentation)
    // ========================================
    testLog('\n--- Multi-Resolution: Level 0 → 1 → 2 ---');

    // Start with coarse mesh (level 0)
    const baseMesh = IcoSphere.generate(0, 1.0);
    const subdividedBase = SubdividedMesh.fromMesh(baseMesh);
    testLog(`Base mesh: ${baseMesh.vertices.length} vertices, ${baseMesh.faces.length} faces`);

    // Run multi-resolution optimization
    const multiResResult = await DevelopableOptimizer.optimizeMultiResolution(subdividedBase, {
      startLevel: 0,
      targetLevel: 2,
      iterationsPerLevel: 50,
      gradientTolerance: 1e-6,
      verbose: true,
      coarseEnergyType: 'Combinatorial (E^P)',
      fineEnergyType: 'Covariance (Smallest Eigenvalue)',
      useCompiled: false,
    });

    const multiResFinal = multiResResult.history[multiResResult.history.length - 1];
    const multiResClassification = CurvatureClassifier.classifyVertices(multiResFinal);
    const multiResDev = (multiResClassification.hingeVertices.length / multiResFinal.vertices.length) * 100;
    const multiResRegions = CurvatureClassifier.countDevelopableRegions(multiResFinal);

    testLog(`\nMulti-Resolution Results:`);
    testLog(`  Developability: ${multiResDev.toFixed(1)}%`);
    testLog(`  Regions: ${multiResRegions}`);
    testLog(`  Quality: ${(multiResDev / Math.max(1, multiResRegions)).toFixed(1)}% per region`);

    // ========================================
    // Comparison
    // ========================================
    testLog(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    testLog(`COMPARISON:`);
    testLog(`\nSingle-Level (Baseline):`);
    testLog(`  Developability: ${singleLevelDev.toFixed(1)}%`);
    testLog(`  Regions: ${singleLevelRegions}`);
    testLog(`  Quality/region: ${(singleLevelDev / Math.max(1, singleLevelRegions)).toFixed(1)}%`);
    testLog(`\nMulti-Resolution (Paper Method):`);
    testLog(`  Developability: ${multiResDev.toFixed(1)}%`);
    testLog(`  Regions: ${multiResRegions}`);
    testLog(`  Quality/region: ${(multiResDev / Math.max(1, multiResRegions)).toFixed(1)}%`);
    testLog(`\nImprovement:`);
    testLog(`  Regions: ${singleLevelRegions} → ${multiResRegions} (${((1 - multiResRegions / singleLevelRegions) * 100).toFixed(1)}% reduction)`);
    testLog(`  Developability: ${singleLevelDev.toFixed(1)}% → ${multiResDev.toFixed(1)}% (${(multiResDev - singleLevelDev > 0 ? '+' : '')}${(multiResDev - singleLevelDev).toFixed(1)}%)`);
    testLog(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`);

    // ========================================
    // Expectations
    // ========================================

    // Multi-resolution should produce significantly fewer regions
    testLog(`Expected: Multi-resolution should produce < 20 regions (ideally 8-12)`);
    testLog(`Actual: ${multiResRegions} regions`);

    if (multiResRegions < singleLevelRegions) {
      testLog(`✓ Multi-resolution reduced fragmentation: ${singleLevelRegions} → ${multiResRegions}`);
    } else {
      testLog(`⚠ Multi-resolution did not reduce fragmentation`);
    }

    // Target region count for a sphere: 8-16 regions is good
    if (multiResRegions <= 20) {
      testLog(`✓ Good region count: ${multiResRegions} regions`);
    } else if (multiResRegions > 40) {
      testLog(`⚠ WARNING: Still high fragmentation (${multiResRegions} regions)`);
    }

    // Quality per region should be reasonable
    const multiResQuality = multiResDev / Math.max(1, multiResRegions);
    testLog(`Quality per region: ${multiResQuality.toFixed(1)}%`);
    expect(multiResQuality).toBeGreaterThan(1.0); // At least 1% per region

    // Multi-resolution should have fewer regions than single-level
    expect(multiResRegions).toBeLessThan(singleLevelRegions);
  });

  it.concurrent.skip('should work with subdivision from level 1 to 2', { timeout: 120000 }, async () => {
    testLog('\n=== Multi-Resolution: Level 1 → 2 ===\n');

    // Start with level 1 (42 vertices)
    const baseMesh = IcoSphere.generate(1, 1.0);
    const subdividedBase = SubdividedMesh.fromMesh(baseMesh);
    testLog(`Base mesh (level 1): ${baseMesh.vertices.length} vertices`);

    const result = await DevelopableOptimizer.optimizeMultiResolution(subdividedBase, {
      startLevel: 1,
      targetLevel: 2,
      iterationsPerLevel: 40,
      verbose: true,
      coarseEnergyType: 'Combinatorial (E^P)',
      fineEnergyType: 'Covariance (Smallest Eigenvalue)',
      useCompiled: false,
    });

    const finalMesh = result.history[result.history.length - 1];
    const classification = CurvatureClassifier.classifyVertices(finalMesh);
    const dev = (classification.hingeVertices.length / finalMesh.vertices.length) * 100;
    const regions = CurvatureClassifier.countDevelopableRegions(finalMesh);

    testLog(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    testLog(`RESULTS:`);
    testLog(`  Developability: ${dev.toFixed(1)}%`);
    testLog(`  Regions: ${regions}`);
    testLog(`  Quality: ${(dev / Math.max(1, regions)).toFixed(1)}% per region`);
    testLog(`  Optimization levels: ${result.subdivisionLevels.join(' → ')}`);
    testLog(`  Energies: ${result.energiesPerLevel.map(e => e.toExponential(2)).join(' → ')}`);
    testLog(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`);

    expect(result.success).toBe(true);
    expect(regions).toBeLessThan(30); // Should avoid excessive fragmentation
    expect(dev).toBeGreaterThan(10); // Should achieve some developability
  });
});
