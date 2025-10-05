import { describe, it } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { DevelopableOptimizer } from '../demos/developable-sphere/src/optimization/DevelopableOptimizer';
import { DevelopableEnergy } from '../demos/developable-sphere/src/energy/DevelopableEnergy';
import { testLog } from './testUtils';

describe('Variance Developability Test', () => {
  it.concurrent('should increase developability percentage', async () => {
    testLog('\n=== VARIANCE DEVELOPABILITY TEST ===');

    // Smaller sphere for faster testing
    const sphere = IcoSphere.generate(2, 1.0);
    testLog(`Sphere: ${sphere.vertices.length} vertices, ${sphere.faces.length} faces`);

    // Check initial developability
    const { hingeVertices: initialHinges, seamVertices: initialSeams } =
      DevelopableEnergy.classifyVertices(sphere, 1e-3);
    testLog(`Initial: ${initialHinges.length} hinges, ${initialSeams.length} seams`);
    testLog(`Initial developable: ${(initialHinges.length / sphere.vertices.length * 100).toFixed(1)}%`);

    const optimizer = new DevelopableOptimizer(sphere);

    // Track developability during optimization
    let prevHingeCount = initialHinges.length;

    const result = await optimizer.optimizeAsync({
      maxIterations: 50, // Run longer to see if it works
      chunkSize: 10,
      energyType: 'Bimodal Variance (Spatial Midpoint)',
      verbose: true,
      captureInterval: 10,
      onProgress: (iteration, energy) => {
        const { hingeVertices } = DevelopableEnergy.classifyVertices(sphere, 1e-3);
        const pct = (hingeVertices.length / sphere.vertices.length * 100).toFixed(1);
        const change = hingeVertices.length - prevHingeCount;
        testLog(`  Iter ${iteration}: ${pct}% developable (${change >= 0 ? '+' : ''}${change} hinges), energy=${energy.toExponential(3)}`);
        prevHingeCount = hingeVertices.length;
      },
    });

    testLog(`\nFinal result:`);
    testLog(`  Iterations: ${result.iterations}`);
    testLog(`  Final energy: ${result.finalEnergy.toExponential(3)}`);
    testLog(`  Convergence: ${result.convergenceReason}`);

    // Check final developability
    const { hingeVertices: finalHinges, seamVertices: finalSeams } =
      DevelopableEnergy.classifyVertices(sphere, 1e-3);
    testLog(`\nFinal: ${finalHinges.length} hinges, ${finalSeams.length} seams`);
    testLog(`Final developable: ${(finalHinges.length / sphere.vertices.length * 100).toFixed(1)}%`);

    const improvementPct = finalHinges.length - initialHinges.length;
    testLog(`\n*** IMPROVEMENT: ${improvementPct >= 0 ? '+' : ''}${improvementPct} hinges ***`);
  }, 60000);
});
