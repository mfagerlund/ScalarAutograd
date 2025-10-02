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
