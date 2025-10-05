/**
 * Test that uses the SAME computation graph for both compiled and uncompiled execution.
 * This eliminates any differences due to graph construction.
 */

import { V } from "../src/V";
import { Value } from "../src/Value";
import { ValueRegistry } from "../src/ValueRegistry";
import { compileIndirectKernel } from "../src/compileIndirectKernel";
import { testLog } from "./testUtils";

describe('Same Graph Test', () => {
  it('should produce identical gradients when using the same graph', () => {
    testLog('\n=== SAME GRAPH TEST ===\n');

    // Build a simple computation graph ONCE
    const params = [V.W(2.0, 'a'), V.W(3.0, 'b'), V.W(4.0, 'c')];
    params.forEach((p, i) => p.paramName = `p${i}`);

    // Expression: (a + b) * c
    const sum = V.add(params[0], params[1]);
    const result = V.mul(sum, params[2]);

    testLog('Built graph: (a + b) * c');
    testLog(`Initial values: a=${params[0].data}, b=${params[1].data}, c=${params[2].data}`);
    testLog(`Result value: ${result.data}\n`);

    // Method 1: Use graph backward (normal)
    testLog('Method 1: Graph backward...');
    params.forEach(p => p.grad = 0);
    result.backward();
    const graphGrads = params.map(p => p.grad);
    testLog(`Graph gradients: [${graphGrads.map(g => g.toExponential(10)).join(', ')}]`);

    // Method 2: Compile the SAME graph and use kernel
    testLog('\nMethod 2: Compile kernel from same graph...');
    const registry = new ValueRegistry();
    params.forEach(p => registry.register(p));

    const kernel = compileIndirectKernel(result, params, registry);
    testLog('Kernel function:');
    testLog(kernel.toString());

    // Build input/gradient indices
    const indices: number[] = [];
    const gradientIndices: number[] = [];

    // Collect all leaf nodes in the graph
    const visited = new Set<Value>();
    const leaves: Value[] = [];
    function collectLeaves(node: Value) {
      if (visited.has(node)) return;
      visited.add(node);
      const prev = (node as any).prev as Value[];
      if (prev.length === 0) {
        if (!leaves.includes(node)) {
          leaves.push(node);
        }
      } else {
        for (const child of prev) {
          collectLeaves(child);
        }
      }
    }
    collectLeaves(result);

    // Map leaves to indices
    for (const leaf of leaves) {
      const regId = registry.getId(leaf);
      indices.push(regId);

      const paramIdx = params.indexOf(leaf);
      gradientIndices.push(paramIdx >= 0 ? paramIdx : -1);
    }

    testLog(`\nIndices: [${indices.join(', ')}]`);
    testLog(`GradientIndices: [${gradientIndices.join(', ')}]`);

    // Execute kernel
    const allValues = registry.getDataArray();
    const gradient = new Array(params.length).fill(0);
    const kernelValue = kernel(allValues, indices, gradientIndices, gradient);

    testLog(`\nKernel value: ${kernelValue}`);
    testLog(`Kernel gradients: [${gradient.map(g => g.toExponential(10)).join(', ')}]`);

    // Compare
    testLog('\n=== COMPARISON ===');
    testLog('Param | Graph Gradient   | Kernel Gradient  | Difference');
    testLog('------|------------------|------------------|------------------');
    for (let i = 0; i < params.length; i++) {
      const diff = Math.abs(gradient[i] - graphGrads[i]);
      testLog(`  ${i}   | ${graphGrads[i].toExponential(10)} | ${gradient[i].toExponential(10)} | ${diff.toExponential(6)}`);
      expect(gradient[i]).toBeCloseTo(graphGrads[i], 15);
    }

    testLog('\n✓ Same graph produces identical gradients!\n');
  });

  it('complex expression with same graph', () => {
    testLog('\n=== COMPLEX SAME GRAPH TEST ===\n');

    const params = [
      V.W(1.0, 'x'), V.W(2.0, 'y'), V.W(3.0, 'z')
    ];
    params.forEach((p, i) => p.paramName = `p${i}`);

    // sqrt(x^2 + y^2 + z^2)
    const x2 = V.square(params[0]);
    const y2 = V.square(params[1]);
    const z2 = V.square(params[2]);
    const sum = V.add(V.add(x2, y2), z2);
    const result = V.sqrt(sum);

    testLog('Built graph: sqrt(x^2 + y^2 + z^2)');
    testLog(`Initial values: x=${params[0].data}, y=${params[1].data}, z=${params[2].data}`);
    testLog(`Result value: ${result.data}\n`);

    // Graph backward
    params.forEach(p => p.grad = 0);
    result.backward();
    const graphGrads = params.map(p => p.grad);

    // Compile kernel
    const registry = new ValueRegistry();
    params.forEach(p => registry.register(p));
    const kernel = compileIndirectKernel(result, params, registry);

    // Collect leaves
    const visited = new Set<Value>();
    const leaves: Value[] = [];
    function collectLeaves(node: Value) {
      if (visited.has(node)) return;
      visited.add(node);
      const prev = (node as any).prev as Value[];
      if (prev.length === 0 && !leaves.includes(node)) {
        leaves.push(node);
      } else {
        for (const child of prev) collectLeaves(child);
      }
    }
    collectLeaves(result);

    const indices = leaves.map(leaf => registry.getId(leaf));
    const gradientIndices = leaves.map(leaf => {
      const idx = params.indexOf(leaf);
      return idx >= 0 ? idx : -1;
    });

    // Execute kernel
    const allValues = registry.getDataArray();
    const gradient = new Array(params.length).fill(0);
    kernel(allValues, indices, gradientIndices, gradient);

    testLog('Graph gradients:  [' + graphGrads.map(g => g.toExponential(10)).join(', ') + ']');
    testLog('Kernel gradients: [' + gradient.map(g => g.toExponential(10)).join(', ') + ']');

    for (let i = 0; i < params.length; i++) {
      expect(gradient[i]).toBeCloseTo(graphGrads[i], 15);
    }

    testLog('\n✓ Complex graph produces identical gradients!\n');
  });
});
