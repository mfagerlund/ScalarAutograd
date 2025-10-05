import { describe, it } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { DevelopableOptimizer } from '../demos/developable-sphere/src/optimization/DevelopableOptimizer';
import { testLog } from './testUtils';

describe('UI Variance Energy Test', () => {
  it('should work in async mode like the UI', async () => {
    testLog('\n=== Testing Variance Energy (UI Mode) ===');

    const sphere = IcoSphere.generate(2, 1.0);
    const optimizer = new DevelopableOptimizer(sphere);

    const result = await optimizer.optimizeAsync({
      maxIterations: 10,
      chunkSize: 5,
      energyType: 'Bimodal Variance (Spatial Midpoint)',
      verbose: true,
      captureInterval: 5,
    });

    testLog(`\nFinal result:`);
    testLog(`  Iterations: ${result.iterations}`);
    testLog(`  Final energy: ${result.finalEnergy.toExponential(3)}`);
    testLog(`  Convergence: ${result.convergenceReason}`);
  }, 30000);
});
