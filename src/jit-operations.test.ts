import { compileGradientFunction } from './jit-compile-value';
import { Value } from './Value';
import { numericalGradient } from '../test/testUtils';

type UnaryOp = {
  name: string;
  valueFn: (v: Value) => Value;
  numericalFn: (x: number) => number;
  testInputs: number[];
  skip?: boolean;
  skipNumericalGrad?: boolean;
};

type BinaryOp = {
  name: string;
  valueFn: (a: Value, b: Value) => Value;
  numericalFn: (a: number, b: number) => number;
  testInputs: [number, number][];
  skip?: boolean;
  skipNumericalGrad?: boolean;
};

function testUnaryOperation(op: UnaryOp) {
  if (op.skip) {
    it.skip(`${op.name} (not implemented)`, () => {});
    return;
  }

  describe(op.name, () => {
    for (const input of op.testInputs) {
      it(`validates for input ${input}`, () => {
        const v = new Value(input, 'x', true);
        const result = op.valueFn(v);
        result.backward();
        const traditionalGrad = v.grad;

        const cv = new Value(input, 'x', true);
        cv.paramName = 'x';
        const cResult = op.valueFn(cv);
        const compiledFn = compileGradientFunction(cResult, [cv]);
        const [compiledGrad] = compiledFn(input);

        if (!op.skipNumericalGrad) {
          const numericalGrad = numericalGradient(op.numericalFn, [input])[0];
          expect(Math.abs(traditionalGrad - numericalGrad)).toBeLessThan(1e-4);
          expect(Math.abs(compiledGrad - numericalGrad)).toBeLessThan(1e-4);
        }

        expect(Math.abs(compiledGrad - traditionalGrad)).toBeLessThan(1e-10);
      });
    }
  });
}

function testBinaryOperation(op: BinaryOp) {
  if (op.skip) {
    it.skip(`${op.name} (not implemented)`, () => {});
    return;
  }

  describe(op.name, () => {
    for (const [inputA, inputB] of op.testInputs) {
      it(`validates for inputs ${inputA}, ${inputB}`, () => {
        const a = new Value(inputA, 'a', true);
        const b = new Value(inputB, 'b', true);
        const result = op.valueFn(a, b);
        result.backward();
        const traditionalGrads = [a.grad, b.grad];

        const ca = new Value(inputA, 'a', true);
        ca.paramName = 'a';
        const cb = new Value(inputB, 'b', true);
        cb.paramName = 'b';
        const cResult = op.valueFn(ca, cb);
        const compiledFn = compileGradientFunction(cResult, [ca, cb]);
        const compiledGrads = compiledFn(inputA, inputB);

        const numericalGrads = numericalGradient(
          (a, b) => op.numericalFn(a, b),
          [inputA, inputB]
        );

        for (let i = 0; i < 2; i++) {
          if (!op.skipNumericalGrad) {
            expect(Math.abs(traditionalGrads[i] - numericalGrads[i])).toBeLessThan(1e-4);
            expect(Math.abs(compiledGrads[i] - numericalGrads[i])).toBeLessThan(1e-4);
          }
          expect(Math.abs(compiledGrads[i] - traditionalGrads[i])).toBeLessThan(1e-10);
        }
      });
    }
  });
}

describe('JIT Compiled Operations - Tier 1 (Core ML)', () => {
  const tier1Unary: UnaryOp[] = [
    {
      name: 'exp',
      valueFn: (v) => v.exp(),
      numericalFn: (x) => Math.exp(x),
      testInputs: [0, 1, -1, 2, -2],
    },
    {
      name: 'log',
      valueFn: (v) => v.log(),
      numericalFn: (x) => Math.log(x),
      testInputs: [0.1, 1, 2, 10],
    },
    {
      name: 'tanh',
      valueFn: (v) => v.tanh(),
      numericalFn: (x) => Math.tanh(x),
      testInputs: [-2, -1, 0, 1, 2],
    },
    {
      name: 'sigmoid',
      valueFn: (v) => v.sigmoid(),
      numericalFn: (x) => 1 / (1 + Math.exp(-x)),
      testInputs: [-2, -1, 0, 1, 2],
    },
    {
      name: 'relu',
      valueFn: (v) => v.relu(),
      numericalFn: (x) => Math.max(0, x),
      testInputs: [-2, -0.5, 0.5, 2],
    },
  ];

  const tier1Binary: BinaryOp[] = [
    {
      name: 'pow (static exponent)',
      valueFn: (a, b) => a.pow(b.data),
      numericalFn: (a, b) => Math.pow(a, b),
      testInputs: [[2, 3], [3, 2], [0.5, 2], [10, 0.5]],
      skipNumericalGrad: true,
    },
  ];

  tier1Unary.forEach(testUnaryOperation);
  tier1Binary.forEach(testBinaryOperation);
});

describe('JIT Compiled Operations - Tier 2 (Common Math)', () => {
  const tier2Unary: UnaryOp[] = [
    {
      name: 'sin',
      valueFn: (v) => v.sin(),
      numericalFn: (x) => Math.sin(x),
      testInputs: [-Math.PI, -Math.PI/2, 0, Math.PI/2, Math.PI],
    },
    {
      name: 'cos',
      valueFn: (v) => v.cos(),
      numericalFn: (x) => Math.cos(x),
      testInputs: [-Math.PI, -Math.PI/2, 0, Math.PI/2, Math.PI],
    },
    {
      name: 'neg',
      valueFn: (v) => v.neg(),
      numericalFn: (x) => -x,
      testInputs: [-2, -1, 0, 1, 2],
    },
    {
      name: 'abs',
      valueFn: (v) => v.abs(),
      numericalFn: (x) => Math.abs(x),
      testInputs: [-2, -0.5, 0.5, 2],
    },
    {
      name: 'square',
      valueFn: (v) => v.square(),
      numericalFn: (x) => x * x,
      testInputs: [-2, -1, 0, 1, 2],
    },
  ];

  tier2Unary.forEach(testUnaryOperation);
});

describe('JIT Compiled Operations - Tier 3 (Extended)', () => {
  const tier3Unary: UnaryOp[] = [
    {
      name: 'tan',
      valueFn: (v) => v.tan(),
      numericalFn: (x) => Math.tan(x),
      testInputs: [-Math.PI/4, 0, Math.PI/4],
    },
    {
      name: 'asin',
      valueFn: (v) => v.asin(),
      numericalFn: (x) => Math.asin(x),
      testInputs: [-0.5, 0, 0.5],
    },
    {
      name: 'acos',
      valueFn: (v) => v.acos(),
      numericalFn: (x) => Math.acos(x),
      testInputs: [-0.5, 0, 0.5],
    },
    {
      name: 'atan',
      valueFn: (v) => v.atan(),
      numericalFn: (x) => Math.atan(x),
      testInputs: [-1, 0, 1],
    },
    {
      name: 'softplus',
      valueFn: (v) => v.softplus(),
      numericalFn: (x) => Math.log(1 + Math.exp(x)),
      testInputs: [-2, -1, 0, 1, 2],
    },
    {
      name: 'floor',
      valueFn: (v) => v.floor(),
      numericalFn: (x) => Math.floor(x),
      testInputs: [-2.5, -1.5, 0.5, 1.5, 2.5],
      skipNumericalGrad: true,
    },
    {
      name: 'ceil',
      valueFn: (v) => v.ceil(),
      numericalFn: (x) => Math.ceil(x),
      testInputs: [-2.5, -1.5, 0.5, 1.5, 2.5],
      skipNumericalGrad: true,
    },
    {
      name: 'round',
      valueFn: (v) => v.round(),
      numericalFn: (x) => Math.round(x),
      testInputs: [-2.5, -1.5, 0.5, 1.5, 2.5],
      skipNumericalGrad: true,
    },
    {
      name: 'cube',
      valueFn: (v) => v.cube(),
      numericalFn: (x) => x * x * x,
      testInputs: [-2, -1, 0, 1, 2],
    },
    {
      name: 'reciprocal',
      valueFn: (v) => v.reciprocal(),
      numericalFn: (x) => 1 / x,
      testInputs: [-2, -1, 0.5, 1, 2],
    },
    {
      name: 'sign',
      valueFn: (v) => v.sign(),
      numericalFn: (x) => Math.sign(x),
      testInputs: [-2, -0.5, 0, 0.5, 2],
      skipNumericalGrad: true,
    },
  ];

  const tier3Binary: BinaryOp[] = [
    {
      name: 'powValue (dynamic exponent)',
      valueFn: (a, b) => a.powValue(b),
      numericalFn: (a, b) => Math.pow(a, b),
      testInputs: [[2, 3], [3, 2], [0.5, 2]],
    },
    {
      name: 'mod',
      valueFn: (a, b) => a.mod(b),
      numericalFn: (a, b) => a % b,
      testInputs: [[5, 3], [7, 4], [10, 3]],
      skipNumericalGrad: true, // Modulo has discontinuous gradients
    },
    {
      name: 'min',
      valueFn: (a, b) => a.min(b),
      numericalFn: (a, b) => Math.min(a, b),
      testInputs: [[1, 2], [2, 1]], // Avoids a==b where gradient is undefined
    },
    {
      name: 'max',
      valueFn: (a, b) => a.max(b),
      numericalFn: (a, b) => Math.max(a, b),
      testInputs: [[1, 2], [2, 1]], // Avoids a==b where gradient is undefined
    },
  ];

  tier3Unary.forEach(testUnaryOperation);
  tier3Binary.forEach(testBinaryOperation);
});
