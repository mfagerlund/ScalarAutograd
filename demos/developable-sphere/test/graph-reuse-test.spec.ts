/**
 * Test whether DifferentiablePlaneAlignment builds the same graph each time.
 */

import { V } from 'scalar-autograd';
import { Vec3 } from 'scalar-autograd';
import { IcoSphere } from '../src/mesh/IcoSphere';
import { DifferentiablePlaneAlignment } from '../src/energy/DifferentiablePlaneAlignment';

describe('Graph Reuse Test - DifferentiablePlaneAlignment', () => {
  it('should build the same graph when called twice with same mesh state', () => {
    const mesh = IcoSphere.generate(0, 1.0); // 12 vertices

    console.log(`\n=== GRAPH REUSE TEST ===`);
    console.log(`Testing if DifferentiablePlaneAlignment builds the same graph twice\n`);

    // Create params
    const params: any[] = [];
    for (const v of mesh.vertices) {
      params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
    }

    const paramsToMesh = (p: any[]) => {
      for (let i = 0; i < mesh.vertices.length; i++) {
        const x = p[3 * i];
        const y = p[3 * i + 1];
        const z = p[3 * i + 2];
        mesh.setVertexPosition(i, new Vec3(x, y, z));
      }
    };

    // Set mesh to initial state
    paramsToMesh(params);

    // Build graph FIRST time
    console.log('Building graph #1...');
    const residuals1 = DifferentiablePlaneAlignment.computeResiduals(mesh);
    console.log(`Graph #1: ${residuals1.length} residuals`);
    console.log(`  First residual value: ${residuals1[0].data.toExponential(10)}`);
    console.log(`  First residual object ID: ${(residuals1[0] as any)._id || 'none'}`);

    // Check if mesh vertices are still the same Value objects
    console.log(`\nMesh vertex 0 after graph #1:`);
    console.log(`  x: ${mesh.vertices[0].x.data} (same object as param? ${mesh.vertices[0].x === params[0]})`);

    // Reset mesh to SAME state
    paramsToMesh(params);

    // Build graph SECOND time
    console.log('\nBuilding graph #2...');
    const residuals2 = DifferentiablePlaneAlignment.computeResiduals(mesh);
    console.log(`Graph #2: ${residuals2.length} residuals`);
    console.log(`  First residual value: ${residuals2[0].data.toExponential(10)}`);
    console.log(`  First residual object ID: ${(residuals2[0] as any)._id || 'none'}`);

    // Check if they're the same objects or different objects
    console.log(`\nComparison:`);
    console.log(`  Same object? ${residuals1[0] === residuals2[0]}`);
    console.log(`  Same value? ${Math.abs(residuals1[0].data - residuals2[0].data) < 1e-15}`);

    // The key question: do the graphs have the same structure?
    console.log(`\nGraph structure:`);
    console.log(`  Graph #1 first residual has ${(residuals1[0] as any).prev?.length || 0} children`);
    console.log(`  Graph #2 first residual has ${(residuals2[0] as any).prev?.length || 0} children`);

    // Now test with gradients
    console.log(`\n=== GRADIENT TEST ===`);

    // Reset and compute gradients from graph #1
    paramsToMesh(params);
    params.forEach((p: any) => p.grad = 0);
    const energy1 = V.sum(residuals1);
    energy1.backward();
    const grads1 = params.map((p: any) => p.grad);

    // Reset and compute gradients from graph #2
    paramsToMesh(params);
    params.forEach((p: any) => p.grad = 0);
    const energy2 = V.sum(residuals2);
    energy2.backward();
    const grads2 = params.map((p: any) => p.grad);

    console.log(`\nGradient comparison (first 5):`);
    for (let i = 0; i < Math.min(5, params.length); i++) {
      const diff = Math.abs(grads1[i] - grads2[i]);
      console.log(`  p[${i}]: ${grads1[i].toExponential(10)} vs ${grads2[i].toExponential(10)} (diff: ${diff.toExponential(6)})`);
    }

    const maxDiff = Math.max(...params.map((_: any, i: number) => Math.abs(grads1[i] - grads2[i])));
    console.log(`\nMax gradient difference: ${maxDiff.toExponential(6)}`);

    if (maxDiff < 1e-15) {
      console.log('✓ Both graphs produce identical gradients\n');
    } else {
      console.log('❌ PROBLEM: Graphs produce different gradients!\n');
      console.log('This means the graph structure is changing between calls.\n');
    }
  });
});
