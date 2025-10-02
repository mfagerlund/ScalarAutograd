import { compileGradientFunction } from './jit-compile-value';
import { Value } from './Value';
import { numericalGradient } from '../test/testUtils';

describe('JIT Compilation Performance', () => {
  it('finds break-even point for simple expression', () => {
    const iterations = [1, 10, 100, 1000, 10000];

    console.log('\nPerformance comparison: (a * b) + (c / d)');
    console.log('Iterations | Traditional | Compiled | Speedup');
    console.log('-----------|-------------|----------|--------');

    for (const iters of iterations) {
      const traditionalTime = performance.now();
      for (let i = 0; i < iters; i++) {
        const a = new Value(2.0);
        const b = new Value(3.0);
        const c = new Value(6.0);
        const d = new Value(2.0);
        const result = a.mul(b).add(c.div(d));
        result.backward();
      }
      const traditionalDuration = performance.now() - traditionalTime;

      const compiledTime = performance.now();
      const a = new Value(2.0, 'a', true);
      a.paramName = 'a';
      const b = new Value(3.0, 'b', true);
      b.paramName = 'b';
      const c = new Value(6.0, 'c', true);
      c.paramName = 'c';
      const d = new Value(2.0, 'd', true);
      d.paramName = 'd';
      const result = a.mul(b).add(c.div(d));

      const compiledFn = compileGradientFunction(result, [a, b, c, d]);

      for (let i = 0; i < iters; i++) {
        compiledFn(2.0, 3.0, 6.0, 2.0);
      }
      const compiledDuration = performance.now() - compiledTime;

      const speedup = (traditionalDuration / compiledDuration).toFixed(2);
      console.log(
        `${iters.toString().padStart(10)} | ` +
        `${traditionalDuration.toFixed(2).padStart(11)}ms | ` +
        `${compiledDuration.toFixed(2).padStart(8)}ms | ` +
        `${speedup}x`
      );
    }
  });

  it('tests larger expression trees', () => {
    const sizes = [5, 10, 20, 50];
    const iterations = 1000;

    console.log(`\nPerformance for different expression sizes (${iterations} iterations each)`);
    console.log('Variables | Traditional | Compiled | Speedup');
    console.log('----------|-------------|----------|--------');

    for (const size of sizes) {
      const traditionalTime = performance.now();
      for (let i = 0; i < iterations; i++) {
        const vars = Array.from({ length: size }, (_, i) => new Value(i + 1));
        let result = vars[0];
        for (let j = 1; j < size; j++) {
          if (j % 4 === 0) result = result.add(vars[j]);
          else if (j % 4 === 1) result = result.sub(vars[j]);
          else if (j % 4 === 2) result = result.mul(vars[j]);
          else result = result.div(vars[j]);
        }
        result.backward();
      }
      const traditionalDuration = performance.now() - traditionalTime;

      const compiledTime = performance.now();
      const vars = Array.from({ length: size }, (_, i) => {
        const v = new Value(i + 1, `x${i}`, true);
        v.paramName = `x${i}`;
        return v;
      });
      let result = vars[0];
      for (let j = 1; j < size; j++) {
        if (j % 4 === 0) result = result.add(vars[j]);
        else if (j % 4 === 1) result = result.sub(vars[j]);
        else if (j % 4 === 2) result = result.mul(vars[j]);
        else result = result.div(vars[j]);
      }

      const compiledFn = compileGradientFunction(result, vars);

      const inputValues = Array.from({ length: size }, (_, i) => i + 1);
      for (let i = 0; i < iterations; i++) {
        compiledFn(...inputValues);
      }
      const compiledDuration = performance.now() - compiledTime;

      const speedup = (traditionalDuration / compiledDuration).toFixed(2);
      console.log(
        `${size.toString().padStart(9)} | ` +
        `${traditionalDuration.toFixed(2).padStart(11)}ms | ` +
        `${compiledDuration.toFixed(2).padStart(8)}ms | ` +
        `${speedup}x`
      );
    }
  });

  it('validates correctness of compiled gradients', () => {
    const a = new Value(2.0, 'a', true);
    const b = new Value(3.0, 'b', true);
    const c = new Value(6.0, 'c', true);
    const result = a.mul(b).add(c);

    result.backward();
    const traditionalGrads = [a.grad, b.grad, c.grad];

    const a2 = new Value(2.0, 'a', true);
    a2.paramName = 'a';
    const b2 = new Value(3.0, 'b', true);
    b2.paramName = 'b';
    const c2 = new Value(6.0, 'c', true);
    c2.paramName = 'c';
    const result2 = a2.mul(b2).add(c2);

    const compiledFn = compileGradientFunction(result2, [a2, b2, c2]);
    const compiledGrads = compiledFn(2.0, 3.0, 6.0);

    expect(compiledGrads).toEqual(traditionalGrads);
  });

  it('validates nested multiplication: (a * b) * (c * d)', () => {
    const a = new Value(2.0, 'a', true);
    const b = new Value(3.0, 'b', true);
    const c = new Value(4.0, 'c', true);
    const d = new Value(5.0, 'd', true);
    const result = a.mul(b).mul(c.mul(d));

    result.backward();
    const traditionalGrads = [a.grad, b.grad, c.grad, d.grad];

    const a2 = new Value(2.0, 'a', true);
    a2.paramName = 'a';
    const b2 = new Value(3.0, 'b', true);
    b2.paramName = 'b';
    const c2 = new Value(4.0, 'c', true);
    c2.paramName = 'c';
    const d2 = new Value(5.0, 'd', true);
    d2.paramName = 'd';
    const result2 = a2.mul(b2).mul(c2.mul(d2));

    const compiledFn = compileGradientFunction(result2, [a2, b2, c2, d2]);
    const compiledGrads = compiledFn(2.0, 3.0, 4.0, 5.0);

    expect(compiledGrads).toEqual(traditionalGrads);
  });

  it('validates nested division: (a / b) / (c / d)', () => {
    const a = new Value(8.0, 'a', true);
    const b = new Value(2.0, 'b', true);
    const c = new Value(6.0, 'c', true);
    const d = new Value(3.0, 'd', true);
    const result = a.div(b).div(c.div(d));

    result.backward();
    const traditionalGrads = [a.grad, b.grad, c.grad, d.grad];

    const a2 = new Value(8.0, 'a', true);
    a2.paramName = 'a';
    const b2 = new Value(2.0, 'b', true);
    b2.paramName = 'b';
    const c2 = new Value(6.0, 'c', true);
    c2.paramName = 'c';
    const d2 = new Value(3.0, 'd', true);
    d2.paramName = 'd';
    const result2 = a2.div(b2).div(c2.div(d2));

    const compiledFn = compileGradientFunction(result2, [a2, b2, c2, d2]);
    const compiledGrads = compiledFn(8.0, 2.0, 6.0, 3.0);

    expect(compiledGrads).toEqual(traditionalGrads);
  });

  it('validates mixed operations: (a + b) * (c - d) / (e + f)', () => {
    const a = new Value(1.0, 'a', true);
    const b = new Value(2.0, 'b', true);
    const c = new Value(10.0, 'c', true);
    const d = new Value(4.0, 'd', true);
    const e = new Value(1.0, 'e', true);
    const f = new Value(2.0, 'f', true);
    const result = a.add(b).mul(c.sub(d)).div(e.add(f));

    result.backward();
    const traditionalGrads = [a.grad, b.grad, c.grad, d.grad, e.grad, f.grad];

    const a2 = new Value(1.0, 'a', true);
    a2.paramName = 'a';
    const b2 = new Value(2.0, 'b', true);
    b2.paramName = 'b';
    const c2 = new Value(10.0, 'c', true);
    c2.paramName = 'c';
    const d2 = new Value(4.0, 'd', true);
    d2.paramName = 'd';
    const e2 = new Value(1.0, 'e', true);
    e2.paramName = 'e';
    const f2 = new Value(2.0, 'f', true);
    f2.paramName = 'f';
    const result2 = a2.add(b2).mul(c2.sub(d2)).div(e2.add(f2));

    const compiledFn = compileGradientFunction(result2, [a2, b2, c2, d2, e2, f2]);
    const compiledGrads = compiledFn(1.0, 2.0, 10.0, 4.0, 1.0, 2.0);

    expect(compiledGrads).toEqual(traditionalGrads);
  });
});

describe('Numerical Gradient Validation', () => {
  it('validates gradients against numerical approximation: (a * b) + c', () => {
    const inputs = [2.0, 3.0, 6.0];
    const [aVal, bVal, cVal] = inputs;

    const fn = (a: number, b: number, c: number) => a * b + c;
    const numericalGrads = numericalGradient(fn, inputs);

    const a = new Value(aVal, 'a', true);
    const b = new Value(bVal, 'b', true);
    const c = new Value(cVal, 'c', true);
    const result = a.mul(b).add(c);

    result.backward();
    const analyticalGrads = [a.grad, b.grad, c.grad];

    const a2 = new Value(aVal, 'a', true);
    a2.paramName = 'a';
    const b2 = new Value(bVal, 'b', true);
    b2.paramName = 'b';
    const c2 = new Value(cVal, 'c', true);
    c2.paramName = 'c';
    const result2 = a2.mul(b2).add(c2);
    const compiledFn = compileGradientFunction(result2, [a2, b2, c2]);
    const compiledGrads = compiledFn(...inputs);

    for (let i = 0; i < inputs.length; i++) {
      expect(Math.abs(analyticalGrads[i] - numericalGrads[i])).toBeLessThan(1e-4);
      expect(Math.abs(compiledGrads[i] - numericalGrads[i])).toBeLessThan(1e-4);
    }
  });

  it('validates gradients for nested multiplication: (a * b) * (c * d)', () => {
    const inputs = [2.0, 3.0, 4.0, 5.0];
    const [aVal, bVal, cVal, dVal] = inputs;

    const fn = (a: number, b: number, c: number, d: number) => (a * b) * (c * d);
    const numericalGrads = numericalGradient(fn, inputs);

    const a = new Value(aVal, 'a', true);
    const b = new Value(bVal, 'b', true);
    const c = new Value(cVal, 'c', true);
    const d = new Value(dVal, 'd', true);
    const result = a.mul(b).mul(c.mul(d));

    result.backward();
    const analyticalGrads = [a.grad, b.grad, c.grad, d.grad];

    const a2 = new Value(aVal, 'a', true);
    a2.paramName = 'a';
    const b2 = new Value(bVal, 'b', true);
    b2.paramName = 'b';
    const c2 = new Value(cVal, 'c', true);
    c2.paramName = 'c';
    const d2 = new Value(dVal, 'd', true);
    d2.paramName = 'd';
    const result2 = a2.mul(b2).mul(c2.mul(d2));
    const compiledFn = compileGradientFunction(result2, [a2, b2, c2, d2]);
    const compiledGrads = compiledFn(...inputs);

    for (let i = 0; i < inputs.length; i++) {
      expect(Math.abs(analyticalGrads[i] - numericalGrads[i])).toBeLessThan(1e-4);
      expect(Math.abs(compiledGrads[i] - numericalGrads[i])).toBeLessThan(1e-4);
    }
  });

  it('validates gradients for complex expression: (a + b) * (c - d) / (e + f)', () => {
    const inputs = [1.0, 2.0, 10.0, 4.0, 1.0, 2.0];
    const [aVal, bVal, cVal, dVal, eVal, fVal] = inputs;

    const fn = (a: number, b: number, c: number, d: number, e: number, f: number) =>
      ((a + b) * (c - d)) / (e + f);
    const numericalGrads = numericalGradient(fn, inputs);

    const a = new Value(aVal, 'a', true);
    const b = new Value(bVal, 'b', true);
    const c = new Value(cVal, 'c', true);
    const d = new Value(dVal, 'd', true);
    const e = new Value(eVal, 'e', true);
    const f = new Value(fVal, 'f', true);
    const result = a.add(b).mul(c.sub(d)).div(e.add(f));

    result.backward();
    const analyticalGrads = [a.grad, b.grad, c.grad, d.grad, e.grad, f.grad];

    const a2 = new Value(aVal, 'a', true);
    a2.paramName = 'a';
    const b2 = new Value(bVal, 'b', true);
    b2.paramName = 'b';
    const c2 = new Value(cVal, 'c', true);
    c2.paramName = 'c';
    const d2 = new Value(dVal, 'd', true);
    d2.paramName = 'd';
    const e2 = new Value(eVal, 'e', true);
    e2.paramName = 'e';
    const f2 = new Value(fVal, 'f', true);
    f2.paramName = 'f';
    const result2 = a2.add(b2).mul(c2.sub(d2)).div(e2.add(f2));
    const compiledFn = compileGradientFunction(result2, [a2, b2, c2, d2, e2, f2]);
    const compiledGrads = compiledFn(...inputs);

    for (let i = 0; i < inputs.length; i++) {
      expect(Math.abs(analyticalGrads[i] - numericalGrads[i])).toBeLessThan(1e-4);
      expect(Math.abs(compiledGrads[i] - numericalGrads[i])).toBeLessThan(1e-4);
    }
  });
});
