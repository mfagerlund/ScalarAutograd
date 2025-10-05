import { Value } from "../src/Value";
import { Losses } from "../src/Losses";

describe("Losses", () => {
  it("mse computes value and gradients correctly", () => {
    const x = new Value(2, "x", true);
    const y = new Value(3, "y", true);
    const tx = new Value(5, "tx");
    const ty = new Value(1, "ty");
    const loss = Losses.mse([x, y], [tx, ty]); // (1/2)*((2-5)^2 + (3-1)^2) = (1/2)*(9+4) = 6.5
    expect(loss.data).toBeCloseTo(6.5);
    loss.backward();
    expect(x.grad).toBeCloseTo(-3);
    expect(y.grad).toBeCloseTo(2);
  });

  it("mae computes value and gradients correctly", () => {
    const x = new Value(2, "x", true);
    const y = new Value(-3, "y", true);
    const tx = new Value(5, "tx");
    const ty = new Value(2, "ty");
    const loss = Losses.mae([x, y], [tx, ty]); // (1/2)*(abs(2-5)+abs(-3-2)) = (1/2)*(3+5)=4
    expect(loss.data).toBeCloseTo(4);
    loss.backward();
    expect(x.grad).toBeCloseTo(-0.5);
    expect(y.grad).toBeCloseTo(-0.5);
  });

  it("binaryCrossEntropy computes value and gradients correctly for easy case", () => {
    const out = new Value(0.9, "out", true);
    const target = new Value(1, "target");
    const loss = Losses.binaryCrossEntropy([out], [target]);
    expect(loss.data).toBeCloseTo(-Math.log(0.9));
    loss.backward();
    expect(out.grad).toBeCloseTo(-1/0.9, 4);
  });

  it("categoricalCrossEntropy computes value and gradients (softmax+NLL)", () => {
    // logits: [2, 1, 0], true = 0
    const a = new Value(2, "a", true);
    const b = new Value(1, "b", true);
    const c = new Value(0, "c", true);
    const targets = [0];
    const loss = Losses.categoricalCrossEntropy([a, b, c], targets);
    const softmax = [
      Math.exp(2)/(Math.exp(2)+Math.exp(1)+Math.exp(0)),
      Math.exp(1)/(Math.exp(2)+Math.exp(1)+Math.exp(0)),
      Math.exp(0)/(Math.exp(2)+Math.exp(1)+Math.exp(0))
    ];
    expect(loss.data).toBeCloseTo(-Math.log(softmax[0]), 4);
    loss.backward();
    expect(a.grad).toBeCloseTo(softmax[0] - 1, 4);
    expect(b.grad).toBeCloseTo(softmax[1], 4);
    expect(c.grad).toBeCloseTo(softmax[2], 4);
  });

  it('handles empty arrays', () => {
    expect(Losses.mse([], []).data).toBe(0);
    expect(Losses.mae([], []).data).toBe(0);
    expect(Losses.binaryCrossEntropy([], []).data).toBe(0);
    expect(Losses.categoricalCrossEntropy([], []).data).toBe(0);
  });

  it('throws on mismatched lengths', () => {
    const a = [new Value(1)];
    const b = [new Value(1), new Value(2)];
    expect(() => Losses.mse(a, b)).toThrow();
  });

  it('handles extreme values in binary cross entropy', () => {
    const out = new Value(0.999999, 'out', true);
    const target = new Value(1);
    const loss = Losses.binaryCrossEntropy([out], [target]);
    expect(loss.data).toBeGreaterThan(0);
    expect(loss.data).toBeLessThan(0.1);
  });

  it('throws on invalid class indices in categorical cross entropy', () => {
    const outputs = [new Value(1), new Value(2)];
    expect(() => Losses.categoricalCrossEntropy(outputs, [2])).toThrow();
    expect(() => Losses.categoricalCrossEntropy(outputs, [-1])).toThrow();
    expect(() => Losses.categoricalCrossEntropy(outputs, [1.5])).toThrow();
  });
});
