import { choleskySolve, computeJtJ, computeJtr, qrSolve } from "./LinearSolver";
import { Value } from "./Value";
import { CompiledFunctions } from "./CompiledFunctions";

/**
 * Configuration options for nonlinear least squares solver.
 * @public
 */
export interface NonlinearLeastSquaresOptions {
  maxIterations?: number;
  costTolerance?: number;
  paramTolerance?: number;
  gradientTolerance?: number;
  lineSearchSteps?: number;
  initialDamping?: number;
  adaptiveDamping?: boolean;
  dampingIncreaseFactor?: number;
  dampingDecreaseFactor?: number;
  maxInnerIterations?: number;
  useQR?: boolean;
  trustRegionRadius?: number;
  verbose?: boolean;
}

/**
 * Result object returned by nonlinear least squares solver.
 * @public
 */
export interface NonlinearLeastSquaresResult {
  success: boolean;
  iterations: number;
  finalCost: number;
  convergenceReason: string;
  computationTime: number;
}

function computeResidualsAndJacobian(
  params: Value[],
  residualFn: (params: Value[]) => Value[]
): { residuals: number[]; J: number[][]; cost: number } {
  const residualValues = residualFn(params);
  const residuals: number[] = [];
  const J: number[][] = [];
  let cost = 0;

  for (const r of residualValues) {
    cost += r.data * r.data;

    // This was not here before
    Value.zeroGradTree(r);
    params.forEach(p => p.grad = 0);

    r.backward();

    const jacobianRow = params.map((p) => p.grad);

    residuals.push(r.data);
    J.push(jacobianRow);
  }

  return { residuals, J, cost };
}


function solveNormalEquations(J: number[][], r: number[], lambda: number = 0, useQR: boolean = false): number[] {
  if (useQR) {
    const m = J.length;
    const n = J[0].length;

    if (lambda > 0) {
      const augmentedJ = [...J];
      for (let i = 0; i < n; i++) {
        const row = Array(n).fill(0);
        row[i] = Math.sqrt(lambda);
        augmentedJ.push(row);
      }
      const augmentedR = [...r, ...Array(n).fill(0)];
      const negAugmentedR = augmentedR.map(x => -x);
      return qrSolve(augmentedJ, negAugmentedR);
    } else {
      const negR = r.map(x => -x);
      return qrSolve(J, negR);
    }
  }

  const JtJ = computeJtJ(J);
  const Jtr = computeJtr(J, r);
  const negJtr = Jtr.map((x) => -x);

  if (lambda > 0) {
    for (let i = 0; i < JtJ.length; i++) {
      JtJ[i][i] += lambda;
    }
  }

  try {
    return choleskySolve(JtJ, negJtr);
  } catch (e) {
    if (lambda === 0) {
      const fallbackLambda = 1e-6;
      for (let i = 0; i < JtJ.length; i++) {
        JtJ[i][i] += fallbackLambda;
      }
      return choleskySolve(JtJ, negJtr);
    }
    throw e;
  }
}

function lineSearch(
  params: Value[],
  delta: number[],
  residualFn: ((params: Value[]) => Value[]) | CompiledFunctions,
  currentCost: number,
  maxSteps: number = 10
): number {
  const originalData = params.map((p) => p.data);
  let alpha = 1.0;

  for (let i = 0; i < maxSteps; i++) {
    params.forEach((p, idx) => {
      p.data = originalData[idx] + alpha * delta[idx];
    });

    const newCost = residualFn instanceof CompiledFunctions
      ? residualFn.evaluate(params).cost
      : (residualFn as (params: Value[]) => Value[])(params).reduce((sum, r) => sum + r.data * r.data, 0);

    if (newCost < currentCost) {
      return alpha;
    }

    alpha *= 0.5;
  }

  params.forEach((p, idx) => {
    p.data = originalData[idx];
  });

  return 0;
}

export function nonlinearLeastSquares(
  params: Value[],
  residualFn: ((params: Value[]) => Value[]) | CompiledFunctions,
  options: NonlinearLeastSquaresOptions = {}
): NonlinearLeastSquaresResult {
  const {
    maxIterations = 100,
    costTolerance = 1e-6,
    paramTolerance = 1e-6,
    gradientTolerance = 1e-6,
    lineSearchSteps = 10,
    initialDamping = 1e-3,
    adaptiveDamping = true,
    dampingIncreaseFactor = 10,
    dampingDecreaseFactor = 10,
    maxInnerIterations = 10,
    useQR = false,
    trustRegionRadius = Infinity,
    verbose = false,
  } = options;

  const startTime = performance.now();
  let prevCost = Infinity;
  let lambda = initialDamping;

  for (let iter = 0; iter < maxIterations; iter++) {
    const { residuals, J, cost } = residualFn instanceof CompiledFunctions
      ? residualFn.evaluate(params)
      : computeResidualsAndJacobian(params, residualFn as (params: Value[]) => Value[]);

    const Jtr = computeJtr(J, residuals);
    const gradientNorm = Math.sqrt(Jtr.reduce((sum, g) => sum + g * g, 0));

    if (verbose) {
      console.log(
        `Iteration ${iter}: cost=${cost.toFixed(6)}, ||∇||=${gradientNorm.toExponential(2)}${adaptiveDamping ? `, λ=${lambda.toExponential(2)}` : ''}`
      );
      if (iter === 0) {
        const m = J.length;  // number of constraints
        const n = J[0]?.length ?? 0;  // number of parameters
        console.log(`  Jacobian shape: ${m}×${n}`);

        // Detect underdetermined system (nullspace exists)
        if (m < n) {
          console.log(`  ⚠ Underdetermined: ${n - m} degrees of freedom (nullspace dimension ≥ ${n - m})`);
        }

        console.log(`  Jacobian:`);
        J.forEach((row, i) => {
          console.log(`    [${row.map(v => v.toFixed(4)).join(', ')}]`);
        });
        console.log(`  Residuals: [${residuals.map(r => r.toFixed(4)).join(', ')}]`);
      }
    }

    if (gradientNorm < gradientTolerance) {
      return {
        success: true,
        iterations: iter,
        finalCost: cost,
        convergenceReason: "Gradient tolerance reached",
        computationTime: performance.now() - startTime,
      };
    }

    if (Math.abs(prevCost - cost) < costTolerance) {
      return {
        success: true,
        iterations: iter,
        finalCost: cost,
        convergenceReason: "Cost tolerance reached",
        computationTime: performance.now() - startTime,
      };
    }

    if (cost < costTolerance) {
      return {
        success: true,
        iterations: iter,
        finalCost: cost,
        convergenceReason: "Cost below threshold",
        computationTime: performance.now() - startTime,
      };
    }

    let delta: number[];
    let accepted = false;
    let innerIterations = 0;

    while (!accepted && innerIterations < maxInnerIterations) {
      try {
        delta = solveNormalEquations(J, residuals, adaptiveDamping ? lambda : 0, useQR);
      } catch (e) {
        if (verbose) {
          console.log(`  Linear solver failed: ${e}`);
        }
        return {
          success: false,
          iterations: iter,
          finalCost: cost,
          convergenceReason: `Linear solver failed: ${e}`,
          computationTime: performance.now() - startTime,
        };
      }

      let deltaNorm = Math.sqrt(delta.reduce((sum, d) => sum + d * d, 0));

      if (verbose && iter === 0 && innerIterations === 0) {
        console.log(`  Initial delta: [${delta.map(d => d.toFixed(4)).join(', ')}]`);
        console.log(`  Delta norm: ${deltaNorm.toFixed(4)}`);
      }

      if (deltaNorm > trustRegionRadius) {
        const scale = trustRegionRadius / deltaNorm;
        delta = delta.map(d => d * scale);
        deltaNorm = trustRegionRadius;
      }

      if (deltaNorm < paramTolerance) {
        return {
          success: true,
          iterations: iter,
          finalCost: cost,
          convergenceReason: "Parameter tolerance reached",
          computationTime: performance.now() - startTime,
        };
      }

      if (adaptiveDamping) {
        const originalData = params.map(p => p.data);
        params.forEach((p, idx) => {
          p.data = originalData[idx] + delta[idx];
        });

        if (verbose && iter === 0 && innerIterations === 0) {
          console.log(`  New params: [${params.map(p => p.data.toFixed(4)).join(', ')}]`);
        }

        const newCost = residualFn instanceof CompiledFunctions
          ? residualFn.evaluate(params).cost
          : (residualFn as (params: Value[]) => Value[])(params).reduce((sum, r) => sum + r.data * r.data, 0);

        if (verbose && iter === 0) {
          console.log(`  Inner iteration ${innerIterations}: λ=${lambda.toExponential(2)}, cost=${cost.toFixed(4)} → ${newCost.toFixed(4)}, accepted=${newCost < cost}`);
        }

        if (newCost < cost) {
          lambda = Math.max(lambda / dampingDecreaseFactor, 1e-10);
          accepted = true;
        } else {
          params.forEach((p, idx) => {
            p.data = originalData[idx];
          });
          lambda = Math.min(lambda * dampingIncreaseFactor, 1e10);
          innerIterations++;
        }
      } else {
        const alpha = lineSearch(params, delta, residualFn, cost, lineSearchSteps);

        if (alpha === 0) {
          return {
            success: false,
            iterations: iter,
            finalCost: cost,
            convergenceReason: "Line search failed",
            computationTime: performance.now() - startTime,
          };
        }
        accepted = true;
      }
    }

    if (!accepted) {
      return {
        success: false,
        iterations: iter,
        finalCost: cost,
        convergenceReason: "Damping adjustment failed",
        computationTime: performance.now() - startTime,
      };
    }

    prevCost = cost;
  }

  const { cost } = residualFn instanceof CompiledFunctions
    ? residualFn.evaluate(params)
    : computeResidualsAndJacobian(params, residualFn as (params: Value[]) => Value[]);
  return {
    success: false,
    iterations: maxIterations,
    finalCost: cost,
    convergenceReason: "Max iterations reached",
    computationTime: performance.now() - startTime,
  };
}
