/**
 * Debug the compiled kernel backward pass by comparing it step-by-step with graph backward.
 */

import { V, Value, Vec3, CompiledResiduals } from 'scalar-autograd';
import { IcoSphere } from '../src/mesh/IcoSphere';
import { DifferentiablePlaneAlignment } from '../src/energy/DifferentiablePlaneAlignment';

describe('Debug Kernel Backward', () => {
  it.concurrent('should trace through backward pass step by step', () => {
    const mesh = IcoSphere.generate(0, 1.0); // 12 vertices

    console.log(`\n=== STEP-BY-STEP BACKWARD DEBUG ===\n`);

    // Create params
    const params: Value[] = [];
    for (const v of mesh.vertices) {
      params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
    }

    const paramsToMesh = (p: Value[]) => {
      for (let i = 0; i < mesh.vertices.length; i++) {
        mesh.vertices[i].x = p[3 * i];
        mesh.vertices[i].y = p[3 * i + 1];
        mesh.vertices[i].z = p[3 * i + 2];
      }
    };

    // Compile
    console.log('Compiling...');
    let compilationGraph: Value[] | null = null;
    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      paramsToMesh(p);
      const residuals = DifferentiablePlaneAlignment.computeResiduals(mesh);
      compilationGraph = residuals;
      return residuals;
    });
    console.log(`Compiled ${compiled.kernelCount} kernels\n`);

    // Use FIRST residual only for detailed debugging
    const residual = compilationGraph![0];
    console.log(`Debugging first residual (vertex 0 energy)`);
    console.log(`Residual value: ${residual.data.toExponential(15)}\n`);

    // Collect all nodes in topological order
    function collectTopoOrder(root: Value): Value[] {
      const visited = new Set<Value>();
      const topo: Value[] = [];

      function visit(node: Value) {
        if (visited.has(node)) return;
        visited.add(node);
        const prev = (node as any).prev as Value[];
        for (const child of prev) {
          visit(child);
        }
        topo.push(node);
      }

      visit(root);
      return topo;
    }

    const topoOrder = collectTopoOrder(residual);
    console.log(`Graph has ${topoOrder.length} nodes (including ${topoOrder.filter(n => (n as any).prev.length === 0).length} leaves)\n`);

    // Print forward pass values for first few nodes
    console.log('=== FORWARD PASS (first 10 nodes) ===');
    for (let i = 0; i < Math.min(10, topoOrder.length); i++) {
      const node = topoOrder[i];
      const prev = (node as any).prev as Value[];
      const isLeaf = prev.length === 0;
      const isParam = params.includes(node);
      const op = (node as any).op || 'leaf';
      console.log(`[${i}] ${op.padEnd(10)} value=${node.data.toExponential(6)} ${isLeaf ? '(leaf' + (isParam ? ', param)' : ', const)') : '(' + prev.length + ' inputs)'}`);
    }

    // Do GRAPH backward pass
    console.log(`\n=== GRAPH BACKWARD PASS ===`);
    params.forEach(p => p.grad = 0);
    residual.backward();
    const graphGrads = params.map(p => p.grad);
    console.log(`Graph backward complete`);
    console.log(`First 5 param gradients: [${graphGrads.slice(0, 5).map(g => g.toExponential(6)).join(', ')}]\n`);

    // Do COMPILED backward pass
    console.log(`=== COMPILED BACKWARD PASS ===`);
    const firstDesc = (compiled as any).functionDescriptors[0];
    const kernelPool = (compiled as any).kernelPool;
    const kernelDesc = kernelPool.kernels.get(firstDesc.kernelHash);

    console.log(`Kernel has ${firstDesc.inputIndices.length} inputs`);
    console.log(`InputIndices (first 10): [${firstDesc.inputIndices.slice(0, 10).join(', ')}]`);
    console.log(`GradientIndices (first 10): [${firstDesc.gradientIndices.slice(0, 10).join(', ')}]`);

    // Get registry and prepare allValues
    const registry = (compiled as any).registry;
    const allValues = registry.getDataArray();

    // Collect leaves from graph for verification
    function collectLeaves(root: Value): Value[] {
      const visited = new Set<Value>();
      const leaves: Value[] = [];
      function visit(node: Value) {
        if (visited.has(node)) return;
        visited.add(node);
        const prev = (node as any).prev as Value[];
        if (prev.length === 0 && !leaves.includes(node)) {
          leaves.push(node);
        } else {
          for (const child of prev) visit(child);
        }
      }
      visit(root);
      return leaves;
    }

    const leaves = collectLeaves(residual);
    console.log(`\n=== KERNEL INPUT VERIFICATION (BEFORE update) ===`);
    console.log(`Graph has ${leaves.length} leaves`);
    console.log(`First 10 leaf values in graph vs registry:`);
    for (let i = 0; i < Math.min(10, leaves.length); i++) {
      const leafRegId = registry.getId(leaves[i]);
      console.log(`  leaf[${i}]: graph.data=${leaves[i].data.toExponential(6)}, registry[${leafRegId}]=${allValues[leafRegId].toExponential(6)}, match=${leaves[i].data === allValues[leafRegId]}`);
    }

    // Update registry with current param values
    params.forEach(p => {
      const id = registry.getId(p);
      allValues[id] = p.data;
    });

    console.log(`\n=== AFTER updating params in registry ===`);
    console.log(`First 10 leaf values after param update:`);
    for (let i = 0; i < Math.min(10, leaves.length); i++) {
      const leafRegId = registry.getId(leaves[i]);
      console.log(`  leaf[${i}]: graph.data=${leaves[i].data.toExponential(6)}, registry[${leafRegId}]=${allValues[leafRegId].toExponential(6)}, match=${leaves[i].data === allValues[leafRegId]}`);
    }
    console.log();

    const kernelGradient = new Array(params.length).fill(0);
    const kernelValue = kernelDesc.kernel(allValues, firstDesc.inputIndices, firstDesc.gradientIndices, kernelGradient);

    console.log(`Kernel value: ${kernelValue.toExponential(15)}`);
    console.log(`First 5 kernel gradients: [${kernelGradient.slice(0, 5).map(g => g.toExponential(6)).join(', ')}]\n`);

    // Compare
    console.log(`=== COMPARISON ===`);
    console.log(`Values match? ${Math.abs(residual.data - kernelValue) < 1e-14}`);
    console.log(`Value diff: ${Math.abs(residual.data - kernelValue).toExponential(6)}\n`);

    console.log(`Gradient comparison (first 10 params):`);
    console.log(`Idx | Graph Grad       | Kernel Grad      | Difference       | Match?`);
    console.log(`----|------------------|------------------|------------------|-------`);
    for (let i = 0; i < Math.min(10, params.length); i++) {
      const diff = Math.abs(graphGrads[i] - kernelGradient[i]);
      const match = diff < 1e-12;
      console.log(`${i.toString().padStart(3)} | ${graphGrads[i].toExponential(10)} | ${kernelGradient[i].toExponential(10)} | ${diff.toExponential(6)} | ${match ? '✓' : '✗'}`);
    }

    const maxDiff = Math.max(...params.map((_, i) => Math.abs(graphGrads[i] - kernelGradient[i])));
    console.log(`\nMax gradient diff: ${maxDiff.toExponential(6)}`);

    // Save the kernel code to file for inspection
    console.log(`\n=== KERNEL CODE ===`);
    const kernelCode = kernelDesc.kernel.toString();
    const lines = kernelCode.split('\n');
    console.log(`Kernel has ${lines.length} lines`);

    const fs = require('fs');
    fs.writeFileSync('C:/Dev/ScalarAutograd/demos/developable-sphere/test/kernel-output.js', kernelCode);
    console.log(`Saved to kernel-output.js`);

    // Print backward section (look for grad_ assignments)
    console.log(`\n=== BACKWARD SECTION (gradient accumulation) ===`);
    const backwardStart = lines.findIndex(l => l.includes('grad_'));
    if (backwardStart >= 0) {
      console.log(`Backward starts at line ${backwardStart}`);
      console.log(lines.slice(backwardStart, Math.min(backwardStart + 50, lines.length)).join('\n'));
    }
  });
});
