import { describe, it } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { DevelopableEnergy } from '../demos/developable-sphere/src/energy/DevelopableEnergy';
import { testLog } from './testUtils';

describe('Variance Simple Test', () => {
  it('should not be 100% developable on a sphere', () => {
    const sphere = IcoSphere.generate(2, 1.0);

    testLog(`\nSphere: ${sphere.vertices.length} vertices`);

    const { hingeVertices, seamVertices } =
      DevelopableEnergy.classifyVertices(sphere, 1e-3);

    testLog(`Hinges: ${hingeVertices.length}`);
    testLog(`Seams: ${seamVertices.length}`);
    testLog(`Developable: ${(hingeVertices.length / sphere.vertices.length * 100).toFixed(1)}%`);

    // Check a few vertex energies
    for (let i = 0; i < Math.min(5, sphere.vertices.length); i++) {
      const energy = DevelopableEnergy.computeVertexEnergy(i, sphere);
      testLog(`Vertex ${i} energy: ${energy.data.toExponential(3)}`);
    }
  });
});
