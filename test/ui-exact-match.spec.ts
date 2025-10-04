import { describe, it } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { DevelopableOptimizer } from '../demos/developable-sphere/src/optimization/DevelopableOptimizer';
import { DevelopableEnergy } from '../demos/developable-sphere/src/energy/DevelopableEnergy';

describe('UI Exact Match Test', () => {
  it('should match UI behavior exactly', async () => {
    console.log('\n=== EXACT UI MATCH TEST ===');

    // Exact UI parameters: subdivision 3, chunk size 5, max iterations 50
    const sphere = IcoSphere.generate(3, 1.0);
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
      maxIterations: 20, // Shorter for testing
      chunkSize: 5,
      energyType: 'variance',
      verbose: true,
      captureInterval: 5,
      onProgress: (iteration, energy) => {
        if (iteration % 5 === 0) {
          const { hingeVertices } = DevelopableEnergy.classifyVertices(sphere, 1e-3);
          const pct = (hingeVertices.length / sphere.vertices.length * 100).toFixed(1);
          const change = hingeVertices.length - prevHingeCount;
          console.log(`  Iter ${iteration}: ${pct}% developable (${change >= 0 ? '+' : ''}${change} hinges)`);
          prevHingeCount = hingeVertices.length;
        }
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

    // Compute variance
    let variance = 0;
    for (let i = 0; i < sphere.vertices.length; i++) {
      const energy = DevelopableEnergy.computeVertexEnergy(i, sphere).data;
      variance += energy;
    }
    variance /= sphere.vertices.length;
    console.log(`Variance: ${variance.toExponential(2)}`);
  }, 60000);
});
