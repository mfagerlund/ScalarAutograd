import { describe, it, expect } from 'vitest';
import { TriangleMesh } from '../src/mesh/TriangleMesh';
import { IcoSphere } from '../src/mesh/IcoSphere';
import { DevelopableOptimizer } from '../src/optimization/DevelopableOptimizer';
import { DevelopableEnergy } from '../src/energy/DevelopableEnergy';
import { ContiguousBimodalEnergy } from '../src/energy/ContiguousBimodalEnergy';

describe('Bimodal vs Contiguous Energy Comparison', () => {
  it('should compare quasi-random bimodal vs contiguous bimodal', { timeout: 30000 }, () => {
    // Generate icosphere (same as demo)
    const subdiv = 3;
    const baseMesh = IcoSphere.generate(subdiv, 1.0);

    // Test bimodal (quasi-random split) - clone the mesh
    const mesh1 = new TriangleMesh(
      baseMesh.vertices.map(v => v.clone()),
      baseMesh.faces.map(f => ({ vertices: [...f.vertices] }))
    );
    const optimizer1 = new DevelopableOptimizer(mesh1);

    const initialEnergy1 = DevelopableEnergy.compute(mesh1).data;
    const initialDev1 = DevelopableEnergy.classifyVertices(mesh1);
    const initialDevPct1 = (initialDev1.hingeVertices.length / mesh1.vertices.length) * 100;

    console.log('\n=== BIMODAL (quasi-random split) ===');
    console.log(`Initial: ${initialDevPct1.toFixed(1)}% developable, energy=${initialEnergy1.toExponential(3)}`);

    const result1 = optimizer1.optimize({
      maxIterations: 50,
      gradientTolerance: 1e-5,
      verbose: false,
      energyType: 'bimodal',
    });

    const finalDev1 = DevelopableEnergy.classifyVertices(mesh1);
    const finalDevPct1 = (finalDev1.hingeVertices.length / mesh1.vertices.length) * 100;

    console.log(`Final: ${finalDevPct1.toFixed(1)}% developable, energy=${result1.finalEnergy.toExponential(3)}`);
    console.log(`Iterations: ${result1.iterations}, reason: ${result1.convergenceReason}`);

    // Test contiguous (spatial split) - clone the mesh
    const mesh2 = new TriangleMesh(
      baseMesh.vertices.map(v => v.clone()),
      baseMesh.faces.map(f => ({ vertices: [...f.vertices] }))
    );
    const optimizer2 = new DevelopableOptimizer(mesh2);

    const initialEnergy2 = ContiguousBimodalEnergy.compute(mesh2).data;
    const initialDev2 = ContiguousBimodalEnergy.classifyVertices(mesh2);
    const initialDevPct2 = (initialDev2.hingeVertices.length / mesh2.vertices.length) * 100;

    console.log('\n=== CONTIGUOUS (spatial split) ===');
    console.log(`Initial: ${initialDevPct2.toFixed(1)}% developable, energy=${initialEnergy2.toExponential(3)}`);

    const result2 = optimizer2.optimize({
      maxIterations: 50,
      gradientTolerance: 1e-5,
      verbose: false,
      energyType: 'contiguous',
    });

    const finalDev2 = ContiguousBimodalEnergy.classifyVertices(mesh2);
    const finalDevPct2 = (finalDev2.hingeVertices.length / mesh2.vertices.length) * 100;

    console.log(`Final: ${finalDevPct2.toFixed(1)}% developable, energy=${result2.finalEnergy.toExponential(3)}`);
    console.log(`Iterations: ${result2.iterations}, reason: ${result2.convergenceReason}`);

    // Compare
    console.log('\n=== COMPARISON ===');
    console.log(`Bimodal improvement: ${(finalDevPct1 - initialDevPct1).toFixed(1)}%`);
    console.log(`Contiguous improvement: ${(finalDevPct2 - initialDevPct2).toFixed(1)}%`);
    console.log(`Winner: ${finalDevPct2 > finalDevPct1 ? 'CONTIGUOUS' : 'BIMODAL'} (+${Math.abs(finalDevPct2 - finalDevPct1).toFixed(1)}%)`);

    // Both should improve developability
    expect(finalDevPct1).toBeGreaterThan(initialDevPct1);
    expect(finalDevPct2).toBeGreaterThan(initialDevPct2);
  });
});
