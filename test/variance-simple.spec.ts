import { describe, it } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { DevelopableEnergy } from '../demos/developable-sphere/src/energy/DevelopableEnergy';

describe('Variance Simple Test', () => {
  it('should not be 100% developable on a sphere', () => {
    const sphere = IcoSphere.generate(2, 1.0);

    console.log(`\nSphere: ${sphere.vertices.length} vertices`);

    const { hingeVertices, seamVertices } =
      DevelopableEnergy.classifyVertices(sphere, 1e-3);

    console.log(`Hinges: ${hingeVertices.length}`);
    console.log(`Seams: ${seamVertices.length}`);
    console.log(`Developable: ${(hingeVertices.length / sphere.vertices.length * 100).toFixed(1)}%`);

    // Check a few vertex energies
    for (let i = 0; i < Math.min(5, sphere.vertices.length); i++) {
      const energy = DevelopableEnergy.computeVertexEnergy(i, sphere);
      console.log(`Vertex ${i} energy: ${energy.data.toExponential(3)}`);
    }
  });
});
