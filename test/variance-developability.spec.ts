import { describe, it } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { DevelopableOptimizer } from '../demos/developable-sphere/src/optimization/DevelopableOptimizer';
import { DevelopableEnergy } from '../demos/developable-sphere/src/energy/DevelopableEnergy';

describe('Variance Developability Test', () => {
  it('should increase developability percentage', async () => {
    console.log('\n=== VARIANCE DEVELOPABILITY TEST ===');

    // Smaller sphere for faster testing
    const sphere = IcoSphere.generate(2, 1.0);
    console.log(`Sphere: ${sphere.vertices.length} vertices, ${sphere.faces.length} faces`);

    // Check initial developability
    const { hingeVertices: initialHinges, seamVertices: initialSeams } =
      DevelopableEnergy.classifyVertices(sphere, 1e-3);
    console.log(`Initial: ${initialHinges.length} hinges, ${initialSeams.length} seams`);
    console.log(`Initial developable: ${(initialHinges.length / sphere.vertices.length * 100).toFixed(1)}%`);

    const optimizer = new DevelopableOptimizer(sphere);

    // Track developability during optimization
    let prevHingeCount = initialHinges.length;

    const result = await optimizer.optimizeAsync({
      maxIterations: 50, // Run longer to see if it works
      chunkSize: 10,
      energyType: 'variance',
      verbose: true,
      captureInterval: 10,
      onProgress: (iteration, energy) => {
        const { hingeVertices } = DevelopableEnergy.classifyVertices(sphere, 1e-3);
        const pct = (hingeVertices.length / sphere.vertices.length * 100).toFixed(1);
        const change = hingeVertices.length - prevHingeCount;
        console.log(`  Iter ${iteration}: ${pct}% developable (${change >= 0 ? '+' : ''}${change} hinges), energy=${energy.toExponential(3)}`);
        prevHingeCount = hingeVertices.length;
      },
    });

    console.log(`\nFinal result:`);
    console.log(`  Iterations: ${result.iterations}`);
    console.log(`  Final energy: ${result.finalEnergy.toExponential(3)}`);
    console.log(`  Convergence: ${result.convergenceReason}`);

    // Check final developability
    const { hingeVertices: finalHinges, seamVertices: finalSeams } =
      DevelopableEnergy.classifyVertices(sphere, 1e-3);
    console.log(`\nFinal: ${finalHinges.length} hinges, ${finalSeams.length} seams`);
    console.log(`Final developable: ${(finalHinges.length / sphere.vertices.length * 100).toFixed(1)}%`);

    const improvementPct = finalHinges.length - initialHinges.length;
    console.log(`\n*** IMPROVEMENT: ${improvementPct >= 0 ? '+' : ''}${improvementPct} hinges ***`);
  }, 60000);
});
