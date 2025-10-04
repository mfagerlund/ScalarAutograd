import { describe, it } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { DevelopableOptimizer } from '../demos/developable-sphere/src/optimization/DevelopableOptimizer';

describe('UI Variance Energy Test', () => {
  it('should work in async mode like the UI', async () => {
    console.log('\n=== Testing Variance Energy (UI Mode) ===');

    const sphere = IcoSphere.generate(2, 1.0);
    const optimizer = new DevelopableOptimizer(sphere);

    const result = await optimizer.optimizeAsync({
      maxIterations: 10,
      chunkSize: 5,
      energyType: 'variance',
      verbose: true,
      captureInterval: 5,
    });

    console.log(`\nFinal result:`);
    console.log(`  Iterations: ${result.iterations}`);
    console.log(`  Final energy: ${result.finalEnergy.toExponential(3)}`);
    console.log(`  Convergence: ${result.convergenceReason}`);
  }, 30000);
});
