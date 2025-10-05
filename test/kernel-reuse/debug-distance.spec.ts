/**
 * Debug distance constraint kernel compilation
 */

import { V } from "../../src/V";
import { ValueRegistry } from "../../src/ValueRegistry";
import { compileIndirectKernel, extractInputIndices } from "../../src/compileIndirectKernel";
import { Value } from "../../src/Value";
import { testLog } from "../testUtils";

describe('Debug Distance Constraint', () => {
  it('should debug distance constraint kernel', () => {
    const x1 = V.W(0.0, 'x1');
    const y1 = V.W(0.0, 'y1');
    const x2 = V.W(3.0, 'x2');
    const y2 = V.W(4.0, 'y2');
    [x1, y1, x2, y2].forEach(v => v.paramName = v.label);

    const dx = V.sub(x2, x1);
    const dy = V.sub(y2, y1);
    const distSq = V.add(V.mul(dx, dx), V.mul(dy, dy));
    const dist = V.sqrt(distSq);
    const residual = V.sub(dist, V.C(5.0));

    const params = [x1, y1, x2, y2];

    // Setup registry and compile kernel
    const registry = new ValueRegistry();
    params.forEach(p => registry.register(p));

    const kernel = compileIndirectKernel(residual, params, registry);
    const indices = extractInputIndices(residual, registry);

    testLog('\n=== Compiled Kernel ===');
    testLog(kernel.toString());

    testLog('\n=== Registry ===');
    testLog('Size:', registry.size);
    testLog('Data array:', registry.getDataArray());

    testLog('\n=== Indices ===');
    testLog('Input indices:', indices);

    testLog('\n=== Graph Info ===');
    testLog('Parameters:', params.map(p => `${p.label}=${p.data} (id=${p._registryId})`));
    testLog('Residual value:', residual.data);

    // Build gradient indices mapping
    const paramIds = params.map(p => registry.getId(p));
    const gradientIndices = indices.map(idx => {
      const paramIndex = paramIds.indexOf(idx);
      return paramIndex >= 0 ? paramIndex : -1;
    });

    // Run kernel
    const allValues = registry.getDataArray();
    const jacobianRow = new Array(params.length).fill(0);
    const kernelValue = kernel(allValues, indices, gradientIndices, jacobianRow);

    testLog('\n=== Kernel Results ===');
    testLog('Value:', kernelValue);
    testLog('Jacobian:', jacobianRow);

    // Run backward
    Value.zeroGradTree(residual);
    params.forEach(p => p.grad = 0);
    residual.backward();

    const graphGradients = params.map(p => p.grad);

    testLog('\n=== Graph Results ===');
    testLog('Value:', residual.data);
    testLog('Gradients:', graphGradients);

    testLog('\n=== Comparison ===');
    for (let i = 0; i < params.length; i++) {
      testLog(`${params[i].label}: kernel=${jacobianRow[i]}, graph=${graphGradients[i]}, diff=${jacobianRow[i] - graphGradients[i]}`);
    }
  });
});
