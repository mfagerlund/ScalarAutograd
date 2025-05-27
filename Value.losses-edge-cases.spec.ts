import { Value } from "./Value";
import { mse, mae, binaryCrossEntropy, categoricalCrossEntropy } from "./Losses";

describe('Loss function edge cases', () => {
  it('handles empty arrays', () => {
    expect(mse([], []).data).toBe(0);
    expect(mae([], []).data).toBe(0);
    expect(binaryCrossEntropy([], []).data).toBe(0);
    expect(categoricalCrossEntropy([], []).data).toBe(0);
  });

  it('throws on mismatched lengths', () => {
    const a = [new Value(1)];
    const b = [new Value(1), new Value(2)];
    expect(() => mse(a, b)).toThrow();
  });

  it('handles extreme values in binary cross entropy', () => {
    const out = new Value(0.999999, 'out', true);
    const target = new Value(1);
    const loss = binaryCrossEntropy([out], [target]);
    expect(loss.data).toBeGreaterThan(0);
    expect(loss.data).toBeLessThan(0.1);
  });

  it('throws on invalid class indices in categorical cross entropy', () => {
    const outputs = [new Value(1), new Value(2)];
    expect(() => categoricalCrossEntropy(outputs, [2])).toThrow();
    expect(() => categoricalCrossEntropy(outputs, [-1])).toThrow();
    expect(() => categoricalCrossEntropy(outputs, [1.5])).toThrow();
  });
});
