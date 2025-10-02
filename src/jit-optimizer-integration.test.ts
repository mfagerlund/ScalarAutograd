import { compileGradientFunction, applyCompiledGradients } from './jit-compile-value';
import { Value } from './Value';
import { SGD } from './Optimizers';

describe('JIT + Optimizer Integration', () => {
  it('validates SGD optimization works identically with graph vs compiled', () => {
    const graphParams = [new Value(0.5, 'x', true), new Value(-0.3, 'y', true)];
    const compParams = [new Value(0.5, 'x', true), new Value(-0.3, 'y', true)];
    compParams.forEach((p, i) => p.paramName = `param${i}`);

    const target = [1.0, 2.0];
    const lr = 0.01;
    const iterations = 50;

    const graphOptimizer = new SGD(graphParams, { learningRate: lr });
    const compOptimizer = new SGD(compParams, { learningRate: lr });

    const graphLoss = () => {
      const errors = graphParams.map((p, i) => p.sub(new Value(target[i])).square());
      return errors.reduce((a, b) => a.add(b));
    };

    const compLossGraph = compParams.map((p, i) => p.sub(new Value(target[i])).square())
      .reduce((a, b) => a.add(b));
    const compiledGradFn = compileGradientFunction(compLossGraph, compParams);

    const graphHistory: number[] = [];
    const compHistory: number[] = [];

    for (let i = 0; i < iterations; i++) {
      graphOptimizer.zeroGrad();
      const gLoss = graphLoss();
      gLoss.backward();
      graphHistory.push(gLoss.data);
      graphOptimizer.step();

      compOptimizer.zeroGrad();
      applyCompiledGradients(compiledGradFn, compParams);
      const cLoss = compParams.map((p, i) => (p.data - target[i]) ** 2).reduce((a, b) => a + b);
      compHistory.push(cLoss);
      compOptimizer.step();
    }

    for (let i = 0; i < iterations; i++) {
      expect(Math.abs(graphHistory[i] - compHistory[i])).toBeLessThan(1e-10);
    }

    for (let i = 0; i < graphParams.length; i++) {
      expect(Math.abs(graphParams[i].data - compParams[i].data)).toBeLessThan(1e-10);
    }
  });

  it('validates gradient descent converges to same solution', () => {
    const graphParams = [new Value(10.0, 'a', true), new Value(-5.0, 'b', true)];
    const compParams = [new Value(10.0, 'a', true), new Value(-5.0, 'b', true)];
    compParams.forEach((p, i) => p.paramName = String.fromCharCode(97 + i));

    const targetA = 3.0;
    const targetB = -1.0;

    const graphOptimizer = new SGD(graphParams, { learningRate: 0.01 });
    const compOptimizer = new SGD(compParams, { learningRate: 0.01 });

    const compLossGraph = compParams[0].sub(new Value(targetA)).square()
      .add(compParams[1].sub(new Value(targetB)).square());
    const compiledGradFn = compileGradientFunction(compLossGraph, compParams);

    for (let i = 0; i < 500; i++) {
      graphOptimizer.zeroGrad();
      const gLoss = graphParams[0].sub(new Value(targetA)).square()
        .add(graphParams[1].sub(new Value(targetB)).square());
      gLoss.backward();
      graphOptimizer.step();

      compOptimizer.zeroGrad();
      applyCompiledGradients(compiledGradFn, compParams);
      compOptimizer.step();
    }

    expect(graphParams[0].data).toBeCloseTo(targetA, 3);
    expect(graphParams[1].data).toBeCloseTo(targetB, 3);
    expect(compParams[0].data).toBeCloseTo(targetA, 3);
    expect(compParams[1].data).toBeCloseTo(targetB, 3);

    expect(Math.abs(graphParams[0].data - compParams[0].data)).toBeLessThan(1e-10);
    expect(Math.abs(graphParams[1].data - compParams[1].data)).toBeLessThan(1e-10);
  });
});
