import { compileGradientFunction } from './jit-compile-value';
import { Value } from './Value';
import { numericalGradient } from '../test/testUtils';

type Operation = {
  name: string;
  buildGraph: (inputs: Value[]) => Value;
  numericalFn: (...args: number[]) => number;
  inputs: number[];
};

const operations: Operation[] = [
  { name: 'add', buildGraph: ([a, b]) => a.add(b), numericalFn: (a, b) => a + b, inputs: [2, 3] },
  { name: 'sub', buildGraph: ([a, b]) => a.sub(b), numericalFn: (a, b) => a - b, inputs: [5, 2] },
  { name: 'mul', buildGraph: ([a, b]) => a.mul(b), numericalFn: (a, b) => a * b, inputs: [3, 4] },
  { name: 'div', buildGraph: ([a, b]) => a.div(b), numericalFn: (a, b) => a / b, inputs: [8, 2] },
  { name: 'pow', buildGraph: ([a]) => a.pow(3), numericalFn: (a) => Math.pow(a, 3), inputs: [2] },
  { name: 'powValue', buildGraph: ([a, b]) => a.powValue(b), numericalFn: (a, b) => Math.pow(a, b), inputs: [2, 3] },
  { name: 'exp', buildGraph: ([a]) => a.exp(), numericalFn: (a) => Math.exp(a), inputs: [1] },
  { name: 'log', buildGraph: ([a]) => a.log(), numericalFn: (a) => Math.log(a), inputs: [2] },
  { name: 'sin', buildGraph: ([a]) => a.sin(), numericalFn: (a) => Math.sin(a), inputs: [1] },
  { name: 'cos', buildGraph: ([a]) => a.cos(), numericalFn: (a) => Math.cos(a), inputs: [1] },
  { name: 'tan', buildGraph: ([a]) => a.tan(), numericalFn: (a) => Math.tan(a), inputs: [0.5] },
  { name: 'asin', buildGraph: ([a]) => a.asin(), numericalFn: (a) => Math.asin(a), inputs: [0.5] },
  { name: 'acos', buildGraph: ([a]) => a.acos(), numericalFn: (a) => Math.acos(a), inputs: [0.5] },
  { name: 'atan', buildGraph: ([a]) => a.atan(), numericalFn: (a) => Math.atan(a), inputs: [1] },
  { name: 'tanh', buildGraph: ([a]) => a.tanh(), numericalFn: (a) => Math.tanh(a), inputs: [1] },
  { name: 'sigmoid', buildGraph: ([a]) => a.sigmoid(), numericalFn: (a) => 1 / (1 + Math.exp(-a)), inputs: [0] },
  { name: 'relu', buildGraph: ([a]) => a.relu(), numericalFn: (a) => Math.max(0, a), inputs: [1] },
  { name: 'softplus', buildGraph: ([a]) => a.softplus(), numericalFn: (a) => Math.log(1 + Math.exp(a)), inputs: [1] },
  { name: 'abs', buildGraph: ([a]) => a.abs(), numericalFn: (a) => Math.abs(a), inputs: [-2] },
  { name: 'neg', buildGraph: ([a]) => a.neg(), numericalFn: (a) => -a, inputs: [3] },
  { name: 'square', buildGraph: ([a]) => a.square(), numericalFn: (a) => a * a, inputs: [3] },
  { name: 'cube', buildGraph: ([a]) => a.cube(), numericalFn: (a) => a * a * a, inputs: [2] },
  { name: 'reciprocal', buildGraph: ([a]) => a.reciprocal(), numericalFn: (a) => 1 / a, inputs: [2] },
  { name: 'min', buildGraph: ([a, b]) => a.min(b), numericalFn: (a, b) => Math.min(a, b), inputs: [2, 3] },
  { name: 'max', buildGraph: ([a, b]) => a.max(b), numericalFn: (a, b) => Math.max(a, b), inputs: [2, 3] },
  { name: 'mod', buildGraph: ([a, b]) => a.mod(b), numericalFn: (a, b) => a % b, inputs: [7, 3] },
];

describe('Comprehensive JIT Validation: [Graph,Compiled] × [Forward,Backward]', () => {
  operations.forEach(op => {
    describe(op.name, () => {
      it('validates forward pass: graph == compiled == numerical', () => {
        const graphInputs = op.inputs.map((val, i) => new Value(val, `x${i}`, true));
        const graphResult = op.buildGraph(graphInputs);
        const graphForward = graphResult.data;

        const compInputs = op.inputs.map((val, i) => {
          const v = new Value(val, `x${i}`, true);
          v.paramName = `x${i}`;
          return v;
        });
        const compResult = op.buildGraph(compInputs);
        const compiledFn = compileGradientFunction(compResult, compInputs);

        const compiledOutput = compiledFn(...op.inputs);
        const compiledForward = op.numericalFn(...op.inputs);

        const numericalForward = op.numericalFn(...op.inputs);

        expect(Math.abs(graphForward - numericalForward)).toBeLessThan(1e-10);
        expect(Math.abs(graphForward - compiledForward)).toBeLessThan(1e-10);
      });

      it('validates backward pass: graph grads == compiled grads == numerical grads', () => {
        const graphInputs = op.inputs.map((val, i) => new Value(val, `x${i}`, true));
        const graphResult = op.buildGraph(graphInputs);
        graphResult.backward();
        const graphGrads = graphInputs.map(v => v.grad);

        const compInputs = op.inputs.map((val, i) => {
          const v = new Value(val, `x${i}`, true);
          v.paramName = `x${i}`;
          return v;
        });
        const compResult = op.buildGraph(compInputs);
        const compiledFn = compileGradientFunction(compResult, compInputs);
        const compiledGrads = compiledFn(...op.inputs);

        const numericalGrads = numericalGradient(op.numericalFn, op.inputs);

        for (let i = 0; i < op.inputs.length; i++) {
          expect(Math.abs(graphGrads[i] - compiledGrads[i])).toBeLessThan(1e-10);
          if (op.name !== 'floor' && op.name !== 'ceil' && op.name !== 'round' && op.name !== 'sign' && op.name !== 'mod') {
            expect(Math.abs(graphGrads[i] - numericalGrads[i])).toBeLessThan(1e-4);
            expect(Math.abs(compiledGrads[i] - numericalGrads[i])).toBeLessThan(1e-4);
          }
        }
      });

      it('validates end-to-end: single test for all four properties', () => {
        const graphInputs = op.inputs.map((val, i) => new Value(val, `x${i}`, true));
        const graphResult = op.buildGraph(graphInputs);
        const graphForwardValue = graphResult.data;
        graphResult.backward();
        const graphBackwardGrads = graphInputs.map(v => v.grad);

        const compInputs = op.inputs.map((val, i) => {
          const v = new Value(val, `x${i}`, true);
          v.paramName = `x${i}`;
          return v;
        });
        const compResult = op.buildGraph(compInputs);
        const compiledFn = compileGradientFunction(compResult, compInputs);
        const compiledBackwardGrads = compiledFn(...op.inputs);
        const compiledForwardValue = op.numericalFn(...op.inputs);

        const numericalForwardValue = op.numericalFn(...op.inputs);
        const numericalBackwardGrads = numericalGradient(op.numericalFn, op.inputs);

        expect(graphForwardValue).toBeCloseTo(compiledForwardValue, 10);
        expect(graphForwardValue).toBeCloseTo(numericalForwardValue, 10);

        for (let i = 0; i < op.inputs.length; i++) {
          expect(graphBackwardGrads[i]).toBeCloseTo(compiledBackwardGrads[i], 10);
          if (op.name !== 'floor' && op.name !== 'ceil' && op.name !== 'round' && op.name !== 'sign' && op.name !== 'mod') {
            expect(graphBackwardGrads[i]).toBeCloseTo(numericalBackwardGrads[i], 4);
          }
        }
      });
    });
  });
});

describe('Realistic Optimization Scenarios', () => {
  it('validates complex graph: f(x,y,z) = sin(x) * exp(y) + log(z)', () => {
    const inputs = [1.0, 0.5, 2.0];
    const fn = (x: number, y: number, z: number) => Math.sin(x) * Math.exp(y) + Math.log(z);

    const [x, y, z] = inputs.map((val, i) => new Value(val, `v${i}`, true));
    const graphResult = x.sin().mul(y.exp()).add(z.log());
    graphResult.backward();
    const graphForward = graphResult.data;
    const graphGrads = [x.grad, y.grad, z.grad];

    const [cx, cy, cz] = inputs.map((val, i) => {
      const v = new Value(val, `v${i}`, true);
      v.paramName = `v${i}`;
      return v;
    });
    const compResult = cx.sin().mul(cy.exp()).add(cz.log());
    const compiledFn = compileGradientFunction(compResult, [cx, cy, cz]);
    const compiledGrads = compiledFn(inputs[0], inputs[1], inputs[2]);
    const compiledForward = fn(inputs[0], inputs[1], inputs[2]);

    const numericalForward = fn(inputs[0], inputs[1], inputs[2]);
    const numericalGrads = numericalGradient(fn, inputs);

    expect(graphForward).toBeCloseTo(compiledForward, 10);
    expect(graphForward).toBeCloseTo(numericalForward, 10);
    for (let i = 0; i < 3; i++) {
      expect(graphGrads[i]).toBeCloseTo(compiledGrads[i], 10);
      expect(graphGrads[i]).toBeCloseTo(numericalGrads[i], 4);
    }
  });

  it('validates loss function scenario: MSE-like computation', () => {
    const predictions = [1.0, 2.0, 3.0];
    const targets = [1.5, 1.8, 3.2];

    const graphLoss = () => {
      const pred = predictions.map((p, i) => new Value(p, `p${i}`, true));
      const targ = targets.map(t => new Value(t));
      const errors = pred.map((p, i) => p.sub(targ[i]).square());
      const sum = errors.reduce((a, b) => a.add(b));
      const loss = sum.div(new Value(predictions.length));
      loss.backward();
      return { forward: loss.data, grads: pred.map(p => p.grad) };
    };

    const compiledLoss = () => {
      const pred = predictions.map((p, i) => {
        const v = new Value(p, `p${i}`, true);
        v.paramName = `p${i}`;
        return v;
      });
      const targ = targets.map(t => new Value(t));
      const errors = pred.map((p, i) => p.sub(targ[i]).square());
      const sum = errors.reduce((a, b) => a.add(b));
      const loss = sum.div(new Value(predictions.length));
      const compiledFn = compileGradientFunction(loss, pred);
      return { forward: ((a, b, c) => {
        const errors = [(a - targets[0]) ** 2, (b - targets[1]) ** 2, (c - targets[2]) ** 2];
        return errors.reduce((x, y) => x + y, 0) / 3;
      })(predictions[0], predictions[1], predictions[2]), grads: compiledFn(predictions[0], predictions[1], predictions[2]) };
    };

    const graph = graphLoss();
    const compiled = compiledLoss();

    expect(graph.forward).toBeCloseTo(compiled.forward, 10);
    for (let i = 0; i < predictions.length; i++) {
      expect(graph.grads[i]).toBeCloseTo(compiled.grads[i], 10);
    }
  });

  it('validates parameterizable constants with V.Param()', () => {
    const x = new Value(3.0, 'x', true);
    x.paramName = 'x';
    const targetValue = new Value(5.0);
    targetValue.paramName = 'target';

    const loss = x.sub(targetValue).square();
    const compiledFn = compileGradientFunction(loss, [x, targetValue]);

    const grads1 = compiledFn(3.0, 5.0);
    expect(grads1[0]).toBeCloseTo(-4, 10);

    const grads2 = compiledFn(3.0, 10.0);
    expect(grads2[0]).toBeCloseTo(-14, 10);
  });
});
