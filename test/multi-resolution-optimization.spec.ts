import { describe, it, expect } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { SubdividedMesh } from '../demos/developable-sphere/src/mesh/SubdividedMesh';
import { DevelopableOptimizer } from '../demos/developable-sphere/src/optimization/DevelopableOptimizer';
import { CurvatureClassifier } from '../demos/developable-sphere/src/energy/CurvatureClassifier';
import { CovarianceEnergy } from '../demos/developable-sphere/src/energy/CovarianceEnergy';

describe('Multi-Resolution Optimization', () => {
  it('should reduce fragmentation compared to single-level optimization', { timeout: 180000 }, async () => {
    console.log('\n=== Multi-Resolution Optimization Test ===\n');

    // ========================================
    // Part 1: Single-level optimization (baseline - should fragment)
    // ========================================
    console.log('--- Baseline: Single-Level Optimization (Level 2) ---');
    const singleLevelMesh = IcoSphere.generate(2, 1.0);
    console.log(`Mesh: ${singleLevelMesh.vertices.length} vertices, ${singleLevelMesh.faces.length} faces`);

    const singleLevelOptimizer = new DevelopableOptimizer(singleLevelMesh);
    const singleLevelResult = await singleLevelOptimizer.optimizeAsync({
      maxIterations: 50,
      gradientTolerance: 1e-6,
      verbose: true,
      energyType: 'combinatorial',
      useCompiled: false,
      chunkSize: 50,
    });

    const singleLevelFinal = singleLevelResult.history[singleLevelResult.history.length - 1];
    const singleLevelClassification = CurvatureClassifier.classifyVertices(singleLevelFinal);
    const singleLevelDev = (singleLevelClassification.hingeVertices.length / singleLevelFinal.vertices.length) * 100;
    const singleLevelRegions = CurvatureClassifier.countDevelopableRegions(singleLevelFinal);

    console.log(`\nSingle-Level Results:`);
    console.log(`  Developability: ${singleLevelDev.toFixed(1)}%`);
    console.log(`  Regions: ${singleLevelRegions}`);
    console.log(`  Quality: ${(singleLevelDev / Math.max(1, singleLevelRegions)).toFixed(1)}% per region`);

    // ========================================
    // Part 2: Multi-resolution optimization (should avoid fragmentation)
    // ========================================
    console.log('\n--- Multi-Resolution: Level 0 → 1 → 2 ---');

    // Start with coarse mesh (level 0)
    const baseMesh = IcoSphere.generate(0, 1.0);
    const subdividedBase = SubdividedMesh.fromMesh(baseMesh);
    console.log(`Base mesh: ${baseMesh.vertices.length} vertices, ${baseMesh.faces.length} faces`);

    // Run multi-resolution optimization
    const multiResResult = await DevelopableOptimizer.optimizeMultiResolution(subdividedBase, {
      startLevel: 0,
      targetLevel: 2,
      iterationsPerLevel: 50,
      gradientTolerance: 1e-6,
      verbose: true,
      coarseEnergyType: 'combinatorial',
      fineEnergyType: 'covariance',
      useCompiled: false,
    });

    const multiResFinal = multiResResult.history[multiResResult.history.length - 1];
    const multiResClassification = CurvatureClassifier.classifyVertices(multiResFinal);
    const multiResDev = (multiResClassification.hingeVertices.length / multiResFinal.vertices.length) * 100;
    const multiResRegions = CurvatureClassifier.countDevelopableRegions(multiResFinal);

    console.log(`\nMulti-Resolution Results:`);
    console.log(`  Developability: ${multiResDev.toFixed(1)}%`);
    console.log(`  Regions: ${multiResRegions}`);
    console.log(`  Quality: ${(multiResDev / Math.max(1, multiResRegions)).toFixed(1)}% per region`);

    // ========================================
    // Comparison
    // ========================================
    console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`COMPARISON:`);
    console.log(`\nSingle-Level (Baseline):`);
    console.log(`  Developability: ${singleLevelDev.toFixed(1)}%`);
    console.log(`  Regions: ${singleLevelRegions}`);
    console.log(`  Quality/region: ${(singleLevelDev / Math.max(1, singleLevelRegions)).toFixed(1)}%`);
    console.log(`\nMulti-Resolution (Paper Method):`);
    console.log(`  Developability: ${multiResDev.toFixed(1)}%`);
    console.log(`  Regions: ${multiResRegions}`);
    console.log(`  Quality/region: ${(multiResDev / Math.max(1, multiResRegions)).toFixed(1)}%`);
    console.log(`\nImprovement:`);
    console.log(`  Regions: ${singleLevelRegions} → ${multiResRegions} (${((1 - multiResRegions / singleLevelRegions) * 100).toFixed(1)}% reduction)`);
    console.log(`  Developability: ${singleLevelDev.toFixed(1)}% → ${multiResDev.toFixed(1)}% (${(multiResDev - singleLevelDev > 0 ? '+' : '')}${(multiResDev - singleLevelDev).toFixed(1)}%)`);
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`);

    // ========================================
    // Expectations
    // ========================================

    // Multi-resolution should produce significantly fewer regions
    console.log(`Expected: Multi-resolution should produce < 20 regions (ideally 8-12)`);
    console.log(`Actual: ${multiResRegions} regions`);

    if (multiResRegions < singleLevelRegions) {
      console.log(`✓ Multi-resolution reduced fragmentation: ${singleLevelRegions} → ${multiResRegions}`);
    } else {
      console.warn(`⚠ Multi-resolution did not reduce fragmentation`);
    }

    // Target region count for a sphere: 8-16 regions is good
    if (multiResRegions <= 20) {
      console.log(`✓ Good region count: ${multiResRegions} regions`);
    } else if (multiResRegions > 40) {
      console.warn(`⚠ WARNING: Still high fragmentation (${multiResRegions} regions)`);
    }

    // Quality per region should be reasonable
    const multiResQuality = multiResDev / Math.max(1, multiResRegions);
    console.log(`Quality per region: ${multiResQuality.toFixed(1)}%`);
    expect(multiResQuality).toBeGreaterThan(1.0); // At least 1% per region

    // Multi-resolution should have fewer regions than single-level
    expect(multiResRegions).toBeLessThan(singleLevelRegions);
  });

  it('should work with subdivision from level 1 to 2', { timeout: 120000 }, async () => {
    console.log('\n=== Multi-Resolution: Level 1 → 2 ===\n');

    // Start with level 1 (42 vertices)
    const baseMesh = IcoSphere.generate(1, 1.0);
    const subdividedBase = SubdividedMesh.fromMesh(baseMesh);
    console.log(`Base mesh (level 1): ${baseMesh.vertices.length} vertices`);

    const result = await DevelopableOptimizer.optimizeMultiResolution(subdividedBase, {
      startLevel: 1,
      targetLevel: 2,
      iterationsPerLevel: 40,
      verbose: true,
      coarseEnergyType: 'combinatorial',
      fineEnergyType: 'covariance',
      useCompiled: false,
    });

    const finalMesh = result.history[result.history.length - 1];
    const classification = CurvatureClassifier.classifyVertices(finalMesh);
    const dev = (classification.hingeVertices.length / finalMesh.vertices.length) * 100;
    const regions = CurvatureClassifier.countDevelopableRegions(finalMesh);

    console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`RESULTS:`);
    console.log(`  Developability: ${dev.toFixed(1)}%`);
    console.log(`  Regions: ${regions}`);
    console.log(`  Quality: ${(dev / Math.max(1, regions)).toFixed(1)}% per region`);
    console.log(`  Optimization levels: ${result.subdivisionLevels.join(' → ')}`);
    console.log(`  Energies: ${result.energiesPerLevel.map(e => e.toExponential(2)).join(' → ')}`);
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`);

    expect(result.success).toBe(true);
    expect(regions).toBeLessThan(30); // Should avoid excessive fragmentation
    expect(dev).toBeGreaterThan(10); // Should achieve some developability
  });
});
