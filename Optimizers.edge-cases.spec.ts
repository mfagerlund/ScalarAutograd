import { Value } from "./Value";
import { SGD, Adam } from "./Optimizers";

describe('Optimizer edge cases', () => {
  it('handles empty parameter list', () => {
    const opt = new SGD([], { learningRate: 0.1 });
    expect(() => opt.step()).not.toThrow();
  });

  it('filters out non-trainable parameters', () => {
    const x = new Value(1, 'x', true);
    const y = new Value(2, 'y', false);
    const opt = new SGD([x, y], { learningRate: 0.1 });
    x.grad = 1;
    y.grad = 1;
    opt.step();
    expect(x.data).toBe(0.9);
    expect(y.data).toBe(2); // unchanged
  });

  it('Adam handles zero gradients correctly', () => {
    const x = new Value(1, 'x', true);
    const opt = new Adam([x], { learningRate: 0.1 });
    x.grad = 0;
    for (let i = 0; i < 10; i++) {
      opt.step();
    }
    expect(x.data).toBe(1); // unchanged
  });
});
