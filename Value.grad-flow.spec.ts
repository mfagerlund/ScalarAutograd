import { Value } from "./Value";

describe('Gradient flow control', () => {
  it('stops gradient at non-requiresGrad nodes', () => {
    const x = new Value(2, 'x', true);
    const y = new Value(3, 'y', false);
    const z = new Value(4, 'z', true);
    const out = x.mul(y).add(z);
    out.backward();
    expect(x.grad).toBe(3);
    expect(y.grad).toBe(0);
    expect(z.grad).toBe(1);
  });

  it('handles detached computation graphs', () => {
    const x = new Value(2, 'x', true);
    const y = x.mul(3);
    const z = new Value(y.data, 'z', true); // detached
    const out = z.mul(4);
    out.backward();
    expect(z.grad).toBe(4);
    expect(x.grad).toBe(0); // no gradient flows to x
  });
});
