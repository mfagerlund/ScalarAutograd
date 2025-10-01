// Optimizers.spec.ts

import { Value } from "../Value";
import { SGD, Adam, AdamW } from "../Optimizers";

function createLoss(x: Value): Value {
  // Loss = (x - 5)^2
  const target = new Value(5);
  const diff = x.sub(target);
  return diff.mul(diff);
}

describe("Optimizers", () => {
  it("SGD minimizes simple quadratic loss", () => {
    const x = new Value(0, "x", true);
    const opt = new SGD([x], { learningRate: 0.1 });

    let lossVal;
    for (let i = 0; i < 100; i++) {
      const loss = createLoss(x);
      lossVal = loss.data;
      if (lossVal < 1e-6) break;
      opt.zeroGrad();
      loss.backward();
      opt.step();
    }

    // AdamW does true weight decay, so the final x is slightly under target
    expect(x.data).toBeLessThan(5.0);
    expect(x.data).toBeGreaterThan(4.8);
  });

  it("Adam minimizes simple quadratic loss", () => {
    const x = new Value(0, "x", true);
    const opt = new Adam([x], { learningRate: 0.1 });

    let lossVal;
    for (let i = 0; i < 100; i++) {
      const loss = createLoss(x);
      lossVal = loss.data;
      if (lossVal < 1e-6) break;
      opt.zeroGrad();
      loss.backward();
      opt.step();
    }

    expect(x.data).toBeCloseTo(5.0, 1);
  });

  it("AdamW minimizes simple quadratic loss with weight decay", () => {
    const x = new Value(0, "x", true);
    const opt = new AdamW([x], { learningRate: 0.1, beta1: 0.9, beta2: 0.999, weightDecay: 0.005 });

    for (let i = 0; i < 100; i++) {
      const loss = createLoss(x);
      opt.zeroGrad();
      loss.backward();
      opt.step();
    }

    expect(x.data).toBeCloseTo(5.0, 1);
  });
});
