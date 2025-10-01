import { Value } from "../Value";

describe('Memory management', () => {
  it('handles large computation graphs', () => {
    let x = new Value(1, 'x', true);
    for (let i = 0; i < 100; i++) {
      x = x.add(1).mul(1.01);
    }
    expect(() => x.backward()).not.toThrow();
  });

  it('zeroGradAll handles multiple disconnected graphs', () => {
    const x1 = new Value(1, 'x1', true);
    const y1 = x1.mul(2);
    const x2 = new Value(2, 'x2', true);
    const y2 = x2.mul(3);
    
    y1.backward();
    y2.backward();
    
    Value.zeroGradAll([y1, y2]);
    expect(x1.grad).toBe(0);
    expect(x2.grad).toBe(0);
  });
});
