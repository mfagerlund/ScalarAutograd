/**
 * Sketch solver using Adam optimizer for gradient descent.
 * Alternative to Levenberg-Marquardt for debugging.
 */

import { Adam, Losses, V, Value, Vec2 } from '../../../src';
import type { ValueMap } from './ConstraintResiduals';
import { computeConstraintResiduals } from './ConstraintResiduals';
import { LineConstraintType } from './types/Entities';
import type { Project } from './types/Project';
import type { SolverOptions, SolverResult } from './SketchSolver';

export class AdamSketchSolver {
  private tolerance: number;
  private maxIterations: number;
  private learningRate: number;

  constructor(options: SolverOptions & { learningRate?: number } = {}) {
    this.tolerance = options.tolerance ?? 1e-6;
    this.maxIterations = options.maxIterations ?? 1000; // Adam needs more iterations
    this.learningRate = options.learningRate ?? 0.1;
  }

  /**
   * Solve constraints using Adam optimizer.
   */
  solve(project: Project): SolverResult {
    // Collect all free variables (non-pinned points, free circle radii)
    const variables: Value[] = [];
    const valueMap: ValueMap = {
      points: new Map(),
      circleRadii: new Map(),
    };

    // Add point positions as variables
    for (const point of project.points) {
      if (point.pinned) {
        // Pinned points are constants
        valueMap.points.set(point, new Vec2(V.C(point.x), V.C(point.y)));
      } else {
        // Free points are variables to optimize
        const vecX = V.W(point.x);
        const vecY = V.W(point.y);
        valueMap.points.set(point, new Vec2(vecX, vecY));
        variables.push(vecX, vecY);
      }
    }

    // Add free circle radii as variables
    for (const circle of project.circles) {
      if (circle.fixedRadius) {
        // Fixed radii are constants
        valueMap.circleRadii.set(circle, V.C(circle.radius));
      } else {
        // Free radii are variables to optimize
        const radiusVar = V.W(circle.radius);
        valueMap.circleRadii.set(circle, radiusVar);
        variables.push(radiusVar);
      }
    }

    // Build loss function from all constraints
    const computeLoss = () => {
      const residuals: Value[] = [];

      // Line constraints
      for (const line of project.lines) {
        const start = valueMap.points.get(line.start)!;
        const end = valueMap.points.get(line.end)!;

        // Fixed length constraint
        if (line.fixedLength !== undefined) {
          const actualLength = end.sub(start).magnitude;
          const targetLength = V.C(line.fixedLength);
          const residual = V.sub(actualLength, targetLength);
          residuals.push(residual);
        }

        // Horizontal constraint (y1 = y2)
        if (line.constraintType === LineConstraintType.Horizontal) {
          const residual = V.sub(end.y, start.y);
          residuals.push(residual);
        }

        // Vertical constraint (x1 = x2)
        if (line.constraintType === LineConstraintType.Vertical) {
          const residual = V.sub(end.x, start.x);
          residuals.push(residual);
        }
      }

      // Inter-entity constraints
      for (const constraint of project.constraints) {
        const constraintResiduals = computeConstraintResiduals(constraint, valueMap);
        residuals.push(...constraintResiduals);
      }

      // Sum of squared residuals
      let loss = V.C(0);
      for (const r of residuals) {
        loss = V.add(loss, V.mul(r, r));
      }

      return loss;
    };

    // If no variables to optimize, check if constraints are satisfied
    if (variables.length === 0) {
      const loss = computeLoss();
      const residualMagnitude = Math.sqrt(loss.data);

      return {
        converged: residualMagnitude < this.tolerance,
        iterations: 0,
        residual: residualMagnitude,
        error: residualMagnitude < this.tolerance ? null : 'Over-constrained (no free variables)',
      };
    }

    try {
      // Create Adam optimizer
      const optimizer = new Adam(variables, { learningRate: this.learningRate });

      let bestLoss = Infinity;
      let bestResidual = Infinity;
      let iterations = 0;

      for (let i = 0; i < this.maxIterations; i++) {
        iterations = i + 1;

        // Compute loss
        const loss = computeLoss();

        // Zero gradients
        optimizer.zeroGrad();

        // Backward pass
        loss.backward();

        // Update parameters
        optimizer.step();

        // Track best solution
        const residualMagnitude = Math.sqrt(loss.data);
        if (residualMagnitude < bestResidual) {
          bestResidual = residualMagnitude;
          bestLoss = loss.data;
        }

        // Check convergence
        if (residualMagnitude < this.tolerance) {
          break;
        }

        // Early stopping if not improving
        if (i > 100 && residualMagnitude > bestResidual * 1.5) {
          break;
        }
      }

      // Update project with solved values
      for (const point of project.points) {
        if (!point.pinned) {
          const vec = valueMap.points.get(point)!;
          point.x = vec.x.data;
          point.y = vec.y.data;
        }
      }

      for (const circle of project.circles) {
        if (!circle.fixedRadius) {
          const radiusVal = valueMap.circleRadii.get(circle)!;
          circle.radius = radiusVal.data;
        }
      }

      return {
        converged: bestResidual < this.tolerance,
        iterations,
        residual: bestResidual,
        error: bestResidual < this.tolerance ? null : `Adam did not converge (best residual: ${bestResidual.toExponential(2)})`,
      };
    } catch (error) {
      return {
        converged: false,
        iterations: 0,
        residual: Infinity,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
}
