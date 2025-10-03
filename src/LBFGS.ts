import { Value } from "./Value";

/**
 * Configuration options for L-BFGS optimizer.
 * @public
 */
export interface LBFGSOptions {
  /** Maximum number of iterations (default: 100) */
  maxIterations?: number;
  /** Tolerance for cost function convergence (default: 1e-6) */
  costTolerance?: number;
  /** Tolerance for gradient norm (default: 1e-6) */
  gradientTolerance?: number;
  /** Number of correction pairs to store (default: 10) */
  historySize?: number;
  /** Maximum number of line search steps (default: 20) */
  maxLineSearchSteps?: number;
  /** Parameter for sufficient decrease (Armijo) condition, c1 in (0, 1) (default: 1e-4) */
  c1?: number;
  /** Parameter for curvature (Wolfe) condition, c2 in (c1, 1) (default: 0.9) */
  c2?: number;
  /** Initial step size (default: 1.0) */
  initialStepSize?: number;
  /** Print progress information (default: false) */
  verbose?: boolean;
}

/**
 * Result object returned by L-BFGS optimizer.
 * @public
 */
export interface LBFGSResult {
  /** Whether optimization converged successfully */
  success: boolean;
  /** Number of iterations performed */
  iterations: number;
  /** Final objective value */
  finalCost: number;
  /** Reason for termination */
  convergenceReason: string;
  /** Total computation time in milliseconds */
  computationTime: number;
  /** Number of function evaluations */
  functionEvaluations: number;
}

/**
 * Computes objective value and gradient.
 * Returns the objective as a Value (whose backward() gives gradients).
 */
function computeObjectiveAndGradient(
  params: Value[],
  objectiveFn: (params: Value[]) => Value
): { cost: number; gradient: number[] } {
  // Zero out gradients
  params.forEach((p) => (p.grad = 0));

  // Compute objective
  const objective = objectiveFn(params);
  const cost = objective.data;

  // Compute gradient via backpropagation
  Value.zeroGradTree(objective);
  params.forEach(p => p.grad = 0);
  objective.backward();

  const gradient = params.map((p) => p.grad);

  return { cost, gradient };
}

/**
 * Performs backtracking line search with strong Wolfe conditions.
 *
 * Strong Wolfe conditions:
 * 1. Sufficient decrease (Armijo): f(x + α*d) ≤ f(x) + c1*α*∇f(x)ᵀd
 * 2. Curvature condition: |∇f(x + α*d)ᵀd| ≤ c2*|∇f(x)ᵀd|
 */
function wolfeLineSearch(
  params: Value[],
  direction: number[],
  objectiveFn: (params: Value[]) => Value,
  currentCost: number,
  currentGradient: number[],
  options: {
    c1: number;
    c2: number;
    maxSteps: number;
    initialStepSize: number;
  }
): { stepSize: number; newCost: number; newGradient: number[]; evaluations: number } {
  const { c1, c2, maxSteps, initialStepSize } = options;
  const originalData = params.map((p) => p.data);

  // Compute directional derivative: ∇f(x)ᵀd
  const dg0 = currentGradient.reduce((sum, g, i) => sum + g * direction[i], 0);

  if (dg0 >= 0) {
    // Direction is not a descent direction - this shouldn't happen with L-BFGS
    // but we handle it gracefully
    return {
      stepSize: 0,
      newCost: currentCost,
      newGradient: currentGradient,
      evaluations: 0,
    };
  }

  let alpha = initialStepSize;
  let evaluations = 0;

  for (let i = 0; i < maxSteps; i++) {
    // Update parameters: x_new = x + α*d
    params.forEach((p, idx) => {
      p.data = originalData[idx] + alpha * direction[idx];
    });

    // Evaluate objective and gradient at new point
    const { cost: newCost, gradient: newGradient } = computeObjectiveAndGradient(params, objectiveFn);
    evaluations++;

    // Check for numerical issues
    if (!Number.isFinite(newCost) || newGradient.some(g => !Number.isFinite(g))) {
      // Numerical overflow/underflow - reduce step size
      alpha *= 0.1;
      continue;
    }

    // Check Armijo condition: f(x + α*d) ≤ f(x) + c1*α*∇f(x)ᵀd
    const armijoCondition = newCost <= currentCost + c1 * alpha * dg0;

    // Compute directional derivative at new point: ∇f(x + α*d)ᵀd
    const dgNew = newGradient.reduce((sum, g, i) => sum + g * direction[i], 0);

    // Check curvature condition (strong Wolfe): |∇f(x + α*d)ᵀd| ≤ c2*|∇f(x)ᵀd|
    const curvatureCondition = Math.abs(dgNew) <= c2 * Math.abs(dg0);

    if (armijoCondition && curvatureCondition) {
      // Both Wolfe conditions satisfied
      return { stepSize: alpha, newCost, newGradient, evaluations };
    }

    if (!armijoCondition) {
      // Armijo condition failed - step size too large
      alpha *= 0.5;
    } else if (dgNew > 0) {
      // Curvature condition failed and derivative became positive
      // We've gone too far - reduce step size
      alpha *= 0.5;
    } else {
      // Curvature condition failed but we can try increasing (cautiously)
      // However, in backtracking we only decrease, so reduce slightly
      alpha *= 0.8;
    }
  }

  // Line search failed - restore original parameters and return zero step
  params.forEach((p, idx) => {
    p.data = originalData[idx];
  });

  return {
    stepSize: 0,
    newCost: currentCost,
    newGradient: currentGradient,
    evaluations,
  };
}

/**
 * Computes search direction using L-BFGS two-loop recursion.
 *
 * This efficiently computes H_k * ∇f(x_k) where H_k is the approximate
 * inverse Hessian, using only stored vectors s and y from recent iterations.
 *
 * Algorithm from Nocedal & Wright (2006), Algorithm 7.4
 */
function twoLoopRecursion(
  gradient: number[],
  s_history: number[][],  // Position differences: x_{k+1} - x_k
  y_history: number[][],  // Gradient differences: ∇f(x_{k+1}) - ∇f(x_k)
  rho_history: number[]   // 1 / (y_k^T * s_k)
): number[] {
  const m = s_history.length;  // Number of stored correction pairs
  const n = gradient.length;    // Number of parameters

  if (m === 0) {
    // No history - use steepest descent (negative gradient)
    return gradient.map((g) => -g);
  }

  // First loop: backward pass
  const alpha: number[] = new Array(m);
  let q = [...gradient];  // Copy gradient

  for (let i = m - 1; i >= 0; i--) {
    alpha[i] = rho_history[i] * s_history[i].reduce((sum, s_val, j) => sum + s_val * q[j], 0);
    q = q.map((q_val, j) => q_val - alpha[i] * y_history[i][j]);
  }

  // Initial Hessian approximation: H_0 = γ * I
  // We use γ = (s_{k-1}^T * y_{k-1}) / (y_{k-1}^T * y_{k-1})
  // This is recommended in Nocedal & Wright (2006), Section 7.2
  const lastIdx = m - 1;
  const s_last = s_history[lastIdx];
  const y_last = y_history[lastIdx];
  const sy = s_last.reduce((sum, s_val, j) => sum + s_val * y_last[j], 0);
  const yy = y_last.reduce((sum, y_val) => sum + y_val * y_val, 0);
  const gamma = yy > 0 ? sy / yy : 1.0;

  // r = H_0 * q = γ * q
  let r = q.map((q_val) => gamma * q_val);

  // Second loop: forward pass
  for (let i = 0; i < m; i++) {
    const beta = rho_history[i] * y_history[i].reduce((sum, y_val, j) => sum + y_val * r[j], 0);
    r = r.map((r_val, j) => r_val + s_history[i][j] * (alpha[i] - beta));
  }

  // Return search direction (negative for minimization)
  return r.map((r_val) => -r_val);
}

/**
 * L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) optimizer.
 *
 * A quasi-Newton method that approximates the inverse Hessian using only
 * a limited history of gradient and position updates. Ideal for large-scale
 * unconstrained optimization problems where computing/storing the full Hessian
 * is impractical.
 *
 * References:
 * - Nocedal & Wright (2006), "Numerical Optimization", 2nd edition, Chapter 7
 * - Liu & Nocedal (1989), "On the limited memory BFGS method for large scale optimization"
 *
 * @param params - Array of Value parameters to optimize
 * @param objectiveFn - Function that computes the objective (returns a Value)
 * @param options - Configuration options
 * @returns Optimization result with convergence information
 *
 * @example
 * ```typescript
 * const x = V.W(1.0);
 * const y = V.W(2.0);
 * const params = [x, y];
 *
 * // Minimize Rosenbrock function: (1-x)² + 100(y-x²)²
 * const result = lbfgs(params, (p) => {
 *   const [x, y] = p;
 *   const a = V.sub(V.C(1), x);
 *   const b = V.sub(y, V.pow(x, 2));
 *   return V.add(V.pow(a, 2), V.mul(V.C(100), V.pow(b, 2)));
 * }, { verbose: true });
 *
 * console.log(`Solution: x=${x.data}, y=${y.data}`);
 * ```
 *
 * @public
 */
export function lbfgs(
  params: Value[],
  objectiveFn: (params: Value[]) => Value,
  options: LBFGSOptions = {}
): LBFGSResult {
  const {
    maxIterations = 100,
    costTolerance = 1e-6,
    gradientTolerance = 1e-6,
    historySize = 10,
    maxLineSearchSteps = 20,
    c1 = 1e-4,
    c2 = 0.9,
    initialStepSize = 1.0,
    verbose = false,
  } = options;

  // Validate parameters
  if (c1 <= 0 || c1 >= 1) {
    throw new Error(`c1 must be in (0, 1), got ${c1}`);
  }
  if (c2 <= c1 || c2 >= 1) {
    throw new Error(`c2 must be in (c1, 1), got ${c2}`);
  }

  const startTime = performance.now();
  let totalFunctionEvaluations = 0;

  // History storage for L-BFGS
  const s_history: number[][] = [];  // Position differences
  const y_history: number[][] = [];  // Gradient differences
  const rho_history: number[] = [];  // 1 / (y^T * s)

  // Initial evaluation
  let { cost, gradient } = computeObjectiveAndGradient(params, objectiveFn);
  totalFunctionEvaluations++;
  let gradientNorm = Math.sqrt(gradient.reduce((sum, g) => sum + g * g, 0));

  if (verbose) {
    console.log(`L-BFGS Optimization`);
    console.log(`Initial: cost=${cost.toFixed(6)}, ||∇||=${gradientNorm.toExponential(2)}`);
  }

  // Check initial gradient
  if (gradientNorm < gradientTolerance) {
    return {
      success: true,
      iterations: 0,
      finalCost: cost,
      convergenceReason: "Initial gradient below tolerance",
      computationTime: performance.now() - startTime,
      functionEvaluations: totalFunctionEvaluations,
    };
  }

  let prevCost = cost;
  let prevGradient = gradient;
  let prevParams = params.map((p) => p.data);

  for (let iter = 0; iter < maxIterations; iter++) {
    // Compute search direction using two-loop recursion
    const direction = twoLoopRecursion(gradient, s_history, y_history, rho_history);

    // Perform line search with Wolfe conditions
    const lineSearchResult = wolfeLineSearch(params, direction, objectiveFn, cost, gradient, {
      c1,
      c2,
      maxSteps: maxLineSearchSteps,
      initialStepSize: iter === 0 ? initialStepSize : 1.0,
    });

    totalFunctionEvaluations += lineSearchResult.evaluations;

    if (lineSearchResult.stepSize === 0) {
      // Line search failed
      return {
        success: false,
        iterations: iter,
        finalCost: cost,
        convergenceReason: "Line search failed to find acceptable step",
        computationTime: performance.now() - startTime,
        functionEvaluations: totalFunctionEvaluations,
      };
    }

    // Update with line search result
    cost = lineSearchResult.newCost;
    gradient = lineSearchResult.newGradient;
    gradientNorm = Math.sqrt(gradient.reduce((sum, g) => sum + g * g, 0));

    // Compute s_k = x_{k+1} - x_k and y_k = ∇f(x_{k+1}) - ∇f(x_k)
    const s_k = params.map((p, i) => p.data - prevParams[i]);
    const y_k = gradient.map((g, i) => g - prevGradient[i]);

    // Compute ρ_k = 1 / (y_k^T * s_k)
    const sTy = s_k.reduce((sum, s_val, i) => sum + s_val * y_k[i], 0);

    if (Math.abs(sTy) > 1e-10) {
      // Only update history if curvature condition is satisfied
      const rho_k = 1.0 / sTy;

      s_history.push(s_k);
      y_history.push(y_k);
      rho_history.push(rho_k);

      // Maintain history size limit
      if (s_history.length > historySize) {
        s_history.shift();
        y_history.shift();
        rho_history.shift();
      }
    }

    if (verbose) {
      console.log(
        `Iteration ${iter + 1}: cost=${cost.toFixed(6)}, ||∇||=${gradientNorm.toExponential(2)}, ` +
        `α=${lineSearchResult.stepSize.toFixed(4)}, history=${s_history.length}`
      );
    }

    // Check convergence criteria
    if (gradientNorm < gradientTolerance) {
      return {
        success: true,
        iterations: iter + 1,
        finalCost: cost,
        convergenceReason: "Gradient tolerance reached",
        computationTime: performance.now() - startTime,
        functionEvaluations: totalFunctionEvaluations,
      };
    }

    if (Math.abs(prevCost - cost) < costTolerance) {
      return {
        success: true,
        iterations: iter + 1,
        finalCost: cost,
        convergenceReason: "Cost tolerance reached",
        computationTime: performance.now() - startTime,
        functionEvaluations: totalFunctionEvaluations,
      };
    }

    if (cost < costTolerance) {
      return {
        success: true,
        iterations: iter + 1,
        finalCost: cost,
        convergenceReason: "Cost below threshold",
        computationTime: performance.now() - startTime,
        functionEvaluations: totalFunctionEvaluations,
      };
    }

    // Prepare for next iteration
    prevCost = cost;
    prevGradient = gradient;
    prevParams = params.map((p) => p.data);
  }

  return {
    success: false,
    iterations: maxIterations,
    finalCost: cost,
    convergenceReason: "Maximum iterations reached",
    computationTime: performance.now() - startTime,
    functionEvaluations: totalFunctionEvaluations,
  };
}
