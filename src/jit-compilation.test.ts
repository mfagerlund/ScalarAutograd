import { CompilableValue, compileGradientFunction } from './jit-compile';

describe('JIT Compilation Performance', () => {
  it('finds break-even point for simple expression', () => {
    const iterations = [1, 10, 100, 1000, 10000];

    console.log('\nPerformance comparison: (a * b) + (c / d)');
    console.log('Iterations | Traditional | Compiled | Speedup');
    console.log('-----------|-------------|----------|--------');

    for (const iters of iterations) {
      const traditionalTime = performance.now();
      for (let i = 0; i < iters; i++) {
        const a = new CompilableValue(2.0);
        const b = new CompilableValue(3.0);
        const c = new CompilableValue(6.0);
        const d = new CompilableValue(2.0);
        const result = a.mul(b).add(c.div(d));
        result.backward();
      }
      const traditionalDuration = performance.now() - traditionalTime;

      const compiledTime = performance.now();
      const a = new CompilableValue(2.0, 'a');
      const b = new CompilableValue(3.0, 'b');
      const c = new CompilableValue(6.0, 'c');
      const d = new CompilableValue(2.0, 'd');
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
        const vars = Array.from({ length: size }, (_, i) => new CompilableValue(i + 1));
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
      const vars = Array.from({ length: size }, (_, i) =>
        new CompilableValue(i + 1, `x${i}`)
      );
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
    const a = new CompilableValue(2.0, 'a');
    const b = new CompilableValue(3.0, 'b');
    const c = new CompilableValue(6.0, 'c');
    const result = a.mul(b).add(c);

    result.backward();
    const traditionalGrads = [a.grad, b.grad, c.grad];

    const a2 = new CompilableValue(2.0, 'a');
    const b2 = new CompilableValue(3.0, 'b');
    const c2 = new CompilableValue(6.0, 'c');
    const result2 = a2.mul(b2).add(c2);

    const compiledFn = compileGradientFunction(result2, [a2, b2, c2]);
    const compiledGrads = compiledFn(2.0, 3.0, 6.0);

    expect(compiledGrads).toEqual(traditionalGrads);
  });
});
