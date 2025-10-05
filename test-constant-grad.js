const { V } = require('./dist/V');
const { CompiledFunctions } = require('./dist/CompiledFunctions');

// Simple graph: x is param, 2.0 is constant
const x = V.W(3);
const c = V.C(2.0);  // Constant
const result = V.mul(x, c);

const compiled = CompiledFunctions.compile([x], (params) => {
  const r = V.mul(params[0], V.C(2.0));
  return [r];
});

console.log("Compiled kernel code:");
console.log(compiled.kernelPool.kernels.values().next().value.kernel.toString());
