/**
 * Conditional console.log that only outputs when VERBOSE=true environment variable is set.
 * This keeps test output clean by default while allowing detailed logging when needed.
 *
 * Usage:
 *   testLog('This will only show if VERBOSE=true');
 *
 * Run with verbose output:
 *   VERBOSE=true npm test
 *   or on Windows: set VERBOSE=true && npm test
 *   or use: npm run test:verbose
 */
export function testLog(...args: any[]): void {
  if (process.env.VERBOSE === 'true') {
    console.log(...args);
  }
}

export function numericalGradient(
  fn: (...args: number[]) => number,
  inputs: number[],
  epsilon: number = 1e-5
): number[] {
  const grads: number[] = [];
  for (let i = 0; i < inputs.length; i++) {
    const inputsPlus = [...inputs];
    const inputsMinus = [...inputs];
    inputsPlus[i] += epsilon;
    inputsMinus[i] -= epsilon;
    const grad = (fn(...inputsPlus) - fn(...inputsMinus)) / (2 * epsilon);
    grads.push(grad);
  }
  return grads;
}
