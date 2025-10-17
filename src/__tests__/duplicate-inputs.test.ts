import { describe, it, expect } from 'vitest';
import { Value } from '../Value';

describe('Duplicate Input Gradient Accumulation', () => {
  it('should correctly compute gradients for different inputs (x * y)', () => {
    const x = new Value(2, 'x', true);
    const y = new Value(3, 'y', true);
    const result = x.mul(y);

    result.backward();

    expect(result.data).toBe(6);
    expect(x.grad).toBe(3);
    expect(y.grad).toBe(2);
  });

  it('should correctly compute gradients when same input is used twice (t * t)', () => {
    const t = new Value(2, 't', true);
    const result = t.mul(t);

    result.backward();

    expect(result.data).toBe(4);
    expect(t.grad).toBe(4); // Should be 2*t = 4, not half!

    const expected = 4.0;
    const ratio = t.grad / expected;
    expect(Math.abs(ratio - 1.0)).toBeLessThan(0.01);
  });

  it('should correctly compute gradients for log(1 - t^2) - SAC entropy pattern', () => {
    const t = new Value(0.5, 't', true);
    const tSquared = t.mul(t);
    const oneMinusTSquared = new Value(1, '1').sub(tSquared);
    const result = oneMinusTSquared.log();

    result.backward();

    // Analytical derivative of log(1-t^2) is -2t/(1-t^2)
    const expected = -2 * 0.5 / (1 - 0.25);
    expect(Math.abs(t.grad - expected)).toBeLessThan(0.0001);
  });
});
