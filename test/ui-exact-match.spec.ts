import { describe, it } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { DevelopableOptimizer } from '../demos/developable-sphere/src/optimization/DevelopableOptimizer';
import { DevelopableEnergy } from '../demos/developable-sphere/src/energy/DevelopableEnergy';
import { testLog } from './testUtils';

describe('UI Exact Match Test', () => {
  it.concurrent.skip('should match UI behavior exactly', async () => {
    testLog('\n=== EXACT UI MATCH TEST ===');

    // Exact UI parameters: subdivision 3, chunk size 5, max iterations 50
    const sphere = IcoSphere.generate(3, 1.0);
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
      maxIterations: 20, // Shorter for testing
      chunkSize: 5,
      energyType: 'Bimodal Variance (Spatial Midpoint)',
      verbose: true,
      captureInterval: 5,
      onProgress: (iteration, energy) => {
        if (iteration % 5 === 0) {
          const { hingeVertices } = DevelopableEnergy.classifyVertices(sphere, 1e-3);
          const pct = (hingeVertices.length / sphere.vertices.length * 100).toFixed(1);
          const change = hingeVertices.length - prevHingeCount;
          testLog(`  Iter ${iteration}: ${pct}% developable (${change >= 0 ? '+' : ''}${change} hinges)`);
          prevHingeCount = hingeVertices.length;
        }
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

    // Compute variance
    let variance = 0;
    for (let i = 0; i < sphere.vertices.length; i++) {
      const energy = DevelopableEnergy.computeVertexEnergy(i, sphere).data;
      variance += energy;
    }
    variance /= sphere.vertices.length;
    testLog(`Variance: ${variance.toExponential(2)}`);
  }, 60000);
});
