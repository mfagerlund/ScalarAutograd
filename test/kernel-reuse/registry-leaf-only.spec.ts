/**
 * Test that registry only contains leaf nodes, not intermediates
 */

import { V } from "../../src/V";
import { ValueRegistry } from "../../src/ValueRegistry";
import { compileIndirectKernel } from "../../src/compileIndirectKernel";

describe('Registry Leaf Nodes Only', () => {
  it('should only register leaf nodes, not intermediates', () => {
    const x = V.W(1, 'x');
    const y = V.W(2, 'y');
    x.paramName = 'x';
    y.paramName = 'y';

    // Build graph with intermediates
    const a = V.sub(x, y);      // intermediate
    const b = V.mul(a, a);      // intermediate
    const residual = V.add(b, V.C(5)); // intermediate + constant

    const registry = new ValueRegistry();
    const params = [x, y];

    compileIndirectKernel(residual, params, registry);

    // Registry should only have: x, y, constant(5)
    expect(registry.size).toBe(3);

    // Verify the values
    const data = registry.getDataArray();
    expect(data).toContain(1); // x
    expect(data).toContain(2); // y
    expect(data).toContain(5); // constant
  });

  it('should handle multiple graphs sharing leaf nodes', () => {
    const x = V.W(1, 'x');
    const y = V.W(2, 'y');
    x.paramName = 'x';
    y.paramName = 'y';

    const registry = new ValueRegistry();
    const params = [x, y];

    // Graph 1: (x + y) * 2
    const r1 = V.mul(V.add(x, y), V.C(2));
    compileIndirectKernel(r1, params, registry);

    const sizeAfterGraph1 = registry.size;

    // Graph 2: (x - y) * 3  (shares x, y)
    const r2 = V.mul(V.sub(x, y), V.C(3));
    compileIndirectKernel(r2, params, registry);

    // Should have: x, y, constant(2), constant(3) = 4 values
    // x and y are deduped (same paramName)
    expect(registry.size).toBe(4);
    expect(registry.size).toBe(sizeAfterGraph1 + 1); // Only added constant(3)
  });

  it('should dedupe constants across graphs', () => {
    const x = V.W(1, 'x');
    x.paramName = 'x';

    const registry = new ValueRegistry();
    const params = [x];

    // Graph 1: x + 5
    const r1 = V.add(x, V.C(5));
    compileIndirectKernel(r1, params, registry);

    // Graph 2: x * 5  (same constant)
    const r2 = V.mul(x, V.C(5));
    compileIndirectKernel(r2, params, registry);

    // Should have: x, constant(5) = 2 values (constant deduped)
    expect(registry.size).toBe(2);
  });

  it('should NOT dedupe intermediates even if identical', () => {
    const x = V.W(1, 'x');
    const y = V.W(2, 'y');
    x.paramName = 'x';
    y.paramName = 'y';

    const registry = new ValueRegistry();
    const params = [x, y];

    // Both graphs compute x+y as intermediate
    const a1 = V.add(x, y);
    const r1 = V.mul(a1, V.C(2));

    const a2 = V.add(x, y);  // Same computation as a1
    const r2 = V.mul(a2, V.C(3));

    compileIndirectKernel(r1, params, registry);
    compileIndirectKernel(r2, params, registry);

    // Intermediates (a1, a2) not in registry
    // Registry: x, y, constant(2), constant(3) = 4
    expect(registry.size).toBe(4);
  });

  it('should handle complex nested graph', () => {
    const w1 = V.W(1.5, 'w1');
    const w2 = V.W(2.5, 'w2');
    const w3 = V.W(0.5, 'w3');
    [w1, w2, w3].forEach(v => v.paramName = v.label);

    const registry = new ValueRegistry();
    const params = [w1, w2, w3];

    // Complex: ((w1 + w2) * w3) - (w1 * w2)
    // Intermediates: sum, prod1, prod2, result
    const sum = V.add(w1, w2);
    const prod1 = V.mul(sum, w3);
    const prod2 = V.mul(w1, w2);
    const residual = V.sub(prod1, prod2);

    compileIndirectKernel(residual, params, registry);

    // Registry should only have: w1, w2, w3 (no intermediates)
    expect(registry.size).toBe(3);

    const data = registry.getDataArray();
    expect(data).toContain(1.5);
    expect(data).toContain(2.5);
    expect(data).toContain(0.5);
  });
});
