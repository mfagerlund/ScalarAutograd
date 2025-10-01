import { CompilableValue, compileGradientFunction } from './jit-compile';

describe('JIT Compilation Optimization Strategies', () => {
  it('measures pure compilation time vs execution time', () => {
    const sizes = [10, 20, 50, 100];

    console.log('\nBreakdown: Compilation vs Execution Time');
    console.log('Variables | Compile Time | Exec Time (100 iters) | Ratio');
    console.log('----------|--------------|----------------------|-------');

    for (const size of sizes) {
      const vars = Array.from({ length: size }, (_, i) => new CompilableValue(i + 1, `x${i}`));
      let result = vars[0];
      for (let j = 1; j < size; j++) {
        if (j % 4 === 0) result = result.add(vars[j]);
        else if (j % 4 === 1) result = result.sub(vars[j]);
        else if (j % 4 === 2) result = result.mul(vars[j]);
        else result = result.div(vars[j]);
      }

      const compileStart = performance.now();
      const compiledFn = compileGradientFunction(result, vars);
      const compileTime = performance.now() - compileStart;

      const inputValues = Array.from({ length: size }, (_, i) => i + 1);

      const execStart = performance.now();
      for (let i = 0; i < 100; i++) {
        compiledFn(...inputValues);
      }
      const execTime = performance.now() - execStart;

      const ratio = (compileTime / execTime).toFixed(2);

      console.log(
        `${size.toString().padStart(9)} | ` +
        `${compileTime.toFixed(2).padStart(12)}ms | ` +
        `${execTime.toFixed(2).padStart(20)}ms | ` +
        `${ratio}x`
      );
    }
  });

  it('tests if reusing graph structure helps', () => {
    const size = 50;
    const runs = 10;

    console.log(`\n Recompiling vs Reusing (${size} variables, ${runs} runs)`);

    const recompileStart = performance.now();
    for (let run = 0; run < runs; run++) {
      const vars = Array.from({ length: size }, (_, i) => new CompilableValue(i + run + 1, `x${i}`));
      let result = vars[0];
      for (let j = 1; j < size; j++) {
        if (j % 4 === 0) result = result.add(vars[j]);
        else if (j % 4 === 1) result = result.sub(vars[j]);
        else if (j % 4 === 2) result = result.mul(vars[j]);
        else result = result.div(vars[j]);
      }
      const compiledFn = compileGradientFunction(result, vars);
      const inputValues = Array.from({ length: size }, (_, i) => i + 1);
      compiledFn(...inputValues);
    }
    const recompileTime = performance.now() - recompileStart;

    const vars = Array.from({ length: size }, (_, i) => new CompilableValue(i + 1, `x${i}`));
    let result = vars[0];
    for (let j = 1; j < size; j++) {
      if (j % 4 === 0) result = result.add(vars[j]);
      else if (j % 4 === 1) result = result.sub(vars[j]);
      else if (j % 4 === 2) result = result.mul(vars[j]);
      else result = result.div(vars[j]);
    }
    const compiledFn = compileGradientFunction(result, vars);

    const reuseStart = performance.now();
    for (let run = 0; run < runs; run++) {
      const inputValues = Array.from({ length: size }, (_, i) => i + run + 1);
      compiledFn(...inputValues);
    }
    const reuseTime = performance.now() - reuseStart;

    console.log(`Recompile every time: ${recompileTime.toFixed(2)}ms`);
    console.log(`Compile once, reuse:  ${reuseTime.toFixed(2)}ms`);
    console.log(`Speedup: ${(recompileTime / reuseTime).toFixed(1)}x`);
  });

  it('shows theoretical chunked compilation (simulate with smaller graphs)', () => {
    const totalSize = 100;
    const chunkSizes = [10, 20, 50, 100];

    console.log('\nSimulated Chunked Compilation');
    console.log('Chunk Size | Total Compile Time | Speedup');
    console.log('-----------|-------------------|--------');

    const results: Array<{size: number, time: number}> = [];

    for (const chunkSize of chunkSizes) {
      const numChunks = Math.ceil(totalSize / chunkSize);

      let totalTime = 0;
      for (let chunk = 0; chunk < numChunks; chunk++) {
        const actualSize = Math.min(chunkSize, totalSize - chunk * chunkSize);
        const vars = Array.from({ length: actualSize }, (_, i) => new CompilableValue(i + 1, `x${i}`));
        let result = vars[0];
        for (let j = 1; j < actualSize; j++) {
          if (j % 4 === 0) result = result.add(vars[j]);
          else if (j % 4 === 1) result = result.sub(vars[j]);
          else if (j % 4 === 2) result = result.mul(vars[j]);
          else result = result.div(vars[j]);
        }

        const start = performance.now();
        compileGradientFunction(result, vars);
        totalTime += performance.now() - start;
      }

      results.push({ size: chunkSize, time: totalTime });
    }

    const baseline = results[results.length - 1].time;
    for (const { size, time } of results) {
      const speedup = (baseline / time).toFixed(2);
      console.log(
        `${size.toString().padStart(10)} | ` +
        `${time.toFixed(2).padStart(17)}ms | ` +
        `${speedup}x`
      );
    }

    console.log('\n💡 Insight: Smaller chunks compile faster individually,');
    console.log('   but you need more of them. Overhead may dominate.');
  });
});
