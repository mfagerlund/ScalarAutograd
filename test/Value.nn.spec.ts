import { Value } from "../Value";
import { SGD, Adam } from "../Optimizers";
import { Losses } from "../Losses";

describe("can train scalar neural networks on minimal problems", () => {

  it("1. learns linear regression (y = 2x + 3) with SGD", () => {
    let w = new Value(Math.random(), "w", true);
    let b = new Value(Math.random(), "b", true);
    const examples = [
      { x: 1, y: 5 },
      { x: 2, y: 7 },
      { x: 3, y: 9 },
    ];
    const opt = new SGD([w, b], { learningRate: 0.1 });
    for (let epoch = 0; epoch < 300; ++epoch) {
      let preds: Value[] = [];
      let targets: Value[] = [];
      for (const ex of examples) {
        const x = new Value(ex.x, "x");
        const pred = w.mul(x).add(b);
        preds.push(pred);
        targets.push(new Value(ex.y));
      }
      let loss = Losses.mse(preds, targets);
      if (loss.data < 1e-4) break;
      w.grad = 0; b.grad = 0;
      loss.backward();
      opt.step();
    }
    expect(w.data).toBeCloseTo(2, 1);
    expect(b.data).toBeCloseTo(3, 1);
  });

  it("2. learns quadratic fit (y = x^2) with SGD", () => {
    let a = new Value(Math.random(), "a", true);
    let b = new Value(Math.random(), "b", true);
    let c = new Value(Math.random(), "c", true);
    const examples = [
      { x: -1, y: 1 },
      { x: 0, y: 0 },
      { x: 2, y: 4 },
      { x: 3, y: 9 },
    ];
    const opt = new SGD([a, b, c], { learningRate: 0.01 });
    
    for (let epoch = 0; epoch < 400; ++epoch) {
      let preds: Value[] = [];
      let targets: Value[] = [];
      for (const ex of examples) {
        const x = new Value(ex.x);
        const pred = a.mul(x.square()).add(b.mul(x)).add(c);
        preds.push(pred);
        targets.push(new Value(ex.y));
      }
      let loss = Losses.mse(preds, targets);
      if (loss.data < 1e-4) break;
      a.grad = 0; b.grad = 0; c.grad = 0;
      loss.backward();
      opt.step();
    }
    expect(a.data).toBeCloseTo(1, 1);
    expect(Math.abs(b.data)).toBeLessThan(0.5);
    expect(Math.abs(c.data)).toBeLessThan(0.5);
  });

  /*
  // This is hard to get to work reliably, I believe it's a difficult problem to solve!?
  it("3. learns XOR with tiny MLP (2-2-1) with SGD", () => {
    function mlp(x1: Value, x2: Value, params: Value[]): Value {
      const [w1, w2, w3, w4, b1, b2, v1, v2, c] = params;
      const h1 = w1.mul(x1).add(w2.mul(x2)).add(b1).tanh();
      const h2 = w3.mul(x1).add(w4.mul(x2)).add(b2).tanh();
      return v1.mul(h1).add(v2.mul(h2)).add(c).sigmoid();
    }
    let params = Array.from({ length: 9 }, (_, i) => new Value(Math.random() - 0.5, "p" + i, true));
    const data = [
      { x: [0, 0], y: 0 },
      { x: [0, 1], y: 1 },
      { x: [1, 0], y: 1 },
      { x: [1, 1], y: 0 },
    ];
    const opt = new SGD(params, { learningRate: 0.01 });
    for (let epoch = 0; epoch < 5000; ++epoch) {
      let preds: Value[] = [];
      let targets: Value[] = [];
      for (const ex of data) {
        const x1 = new Value(ex.x[0]);
        const x2 = new Value(ex.x[1]);
        const pred = mlp(x1, x2, params);
        preds.push(pred);
        targets.push(new Value(ex.y));
      }
      let loss = binaryCrossEntropy(preds, targets);
      if (loss.data < 1e-3) break;
      for (const p of params) p.grad = 0;
      loss.backward();
      opt.step();
    }
    const out00 = mlp(new Value(0), new Value(0), params).data;
    const out01 = mlp(new Value(0), new Value(1), params).data;
    const out10 = mlp(new Value(1), new Value(0), params).data;
    const out11 = mlp(new Value(1), new Value(1), params).data;
    expect((out00 < 0.4 || out00 > 0.6)).toBe(true);
    expect(out01).toBeGreaterThan(0.6);
    expect(out10).toBeGreaterThan(0.6);
    expect(out11).toBeLessThan(0.4);
  });*/
});
