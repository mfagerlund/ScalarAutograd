/**
 * Test ValueRegistry deduplication behavior
 */

import { V } from "../../src/V";
import { ValueRegistry } from "../../src/ValueRegistry";

describe('ValueRegistry', () => {
  it('should assign unique IDs to values', () => {
    const reg = new ValueRegistry();
    const v1 = V.W(1.0);
    const v2 = V.W(2.0);

    const id1 = reg.register(v1);
    const id2 = reg.register(v2);

    expect(id1).toBe(0);
    expect(id2).toBe(1);
    expect(v1._registryId).toBe(0);
    expect(v2._registryId).toBe(1);
    expect(reg.size).toBe(2);
  });

  it('should dedupe constants by value only', () => {
    const reg = new ValueRegistry();
    const c1 = V.C(5.0, 'width');
    const c2 = V.C(5.0, 'height');
    const c3 = V.C(3.0, 'depth');

    const id1 = reg.register(c1);
    const id2 = reg.register(c2);
    const id3 = reg.register(c3);

    expect(id1).toBe(id2); // Same value, different labels → same ID
    expect(id1).not.toBe(id3); // Different value → different ID
    expect(reg.size).toBe(2); // Only 2 unique constants
  });

  it('should NOT dedupe weights even with same value', () => {
    const reg = new ValueRegistry();
    const w1 = V.W(1.0);
    const w2 = V.W(1.0);

    const id1 = reg.register(w1);
    const id2 = reg.register(w2);

    expect(id1).not.toBe(id2); // Always unique
    expect(reg.size).toBe(2);
  });

  it('should dedupe variables by paramName', () => {
    const reg = new ValueRegistry();
    const x1 = V.W(1.0);
    const x2 = V.W(2.0);
    x1.paramName = 'x';
    x2.paramName = 'x';

    const id1 = reg.register(x1);
    const id2 = reg.register(x2);

    expect(id1).toBe(id2); // Same paramName → same ID
    expect(reg.size).toBe(1); // Deduped
  });

  it('should NOT dedupe variables with different paramNames', () => {
    const reg = new ValueRegistry();
    const x = V.W(1.0);
    const y = V.W(1.0);
    x.paramName = 'x';
    y.paramName = 'y';

    const id1 = reg.register(x);
    const id2 = reg.register(y);

    expect(id1).not.toBe(id2);
    expect(reg.size).toBe(2);
  });

  it('should NOT dedupe variables without paramName', () => {
    const reg = new ValueRegistry();
    const w1 = V.W(1.0);
    const w2 = V.W(1.0);
    // No paramName set

    const id1 = reg.register(w1);
    const id2 = reg.register(w2);

    expect(id1).not.toBe(id2);
    expect(reg.size).toBe(2);
  });

  it('should NOT dedupe computed values', () => {
    const reg = new ValueRegistry();
    const a = V.C(2);
    const b = V.C(3);

    const sum1 = V.add(a, b);
    const sum2 = V.add(a, b);

    reg.register(a);
    reg.register(b);
    const id1 = reg.register(sum1);
    const id2 = reg.register(sum2);

    expect(id1).not.toBe(id2); // Different Value objects
  });

  it('should handle registering same value multiple times', () => {
    const reg = new ValueRegistry();
    const v = V.W(1.0);

    const id1 = reg.register(v);
    const id2 = reg.register(v);
    const id3 = reg.register(v);

    expect(id1).toBe(id2);
    expect(id2).toBe(id3);
    expect(reg.size).toBe(1); // Only registered once
  });

  it('should provide data array', () => {
    const reg = new ValueRegistry();
    const a = V.W(1.0);
    const b = V.W(2.0);
    const c = V.C(3.0);

    reg.register(a);
    reg.register(b);
    reg.register(c);

    const data = reg.getDataArray();
    expect(data).toEqual([1.0, 2.0, 3.0]);
  });

  it('should update data array', () => {
    const reg = new ValueRegistry();
    const a = V.W(1.0);
    const b = V.W(2.0);

    reg.register(a);
    reg.register(b);

    a.data = 10.0;
    b.data = 20.0;

    const data = [0, 0];
    reg.updateDataArray(data);

    expect(data).toEqual([10.0, 20.0]);
  });

  it('should throw when getting ID of unregistered value', () => {
    const reg = new ValueRegistry();
    const v = V.W(1.0);

    expect(() => reg.getId(v)).toThrow('Value not registered');
  });

  it('should handle mixed constants and variables', () => {
    const reg = new ValueRegistry();

    const c1 = V.C(5.0);
    const c2 = V.C(5.0);
    const x = V.W(5.0);
    x.paramName = 'x';
    const y = V.W(5.0);
    y.paramName = 'y';
    const w = V.W(5.0); // No paramName

    reg.register(c1);
    reg.register(c2);
    reg.register(x);
    reg.register(y);
    reg.register(w);

    expect(reg.getId(c1)).toBe(reg.getId(c2)); // Constants deduped
    expect(reg.getId(x)).not.toBe(reg.getId(y)); // Different paramNames
    expect(reg.getId(x)).not.toBe(reg.getId(w)); // x has paramName, w doesn't
    expect(reg.getId(y)).not.toBe(reg.getId(w)); // y has paramName, w doesn't

    expect(reg.size).toBe(4); // c1=c2, x, y, w
  });
});
