/**
 * Sketch solver using nonlinear least squares optimization.
 * Solves geometric constraints by minimizing residuals.
 */

import { Value, V, Vec2 } from '../../../src';
import { nonlinearLeastSquares } from '../../../src/NonlinearLeastSquares';
import type { Project } from './types/Project';
import type { Point, Line, Circle } from './types/Entities';
import { computeConstraintResiduals } from './ConstraintResiduals';
import type { ValueMap } from './ConstraintResiduals';

export interface SolverResult {
  converged: boolean;
  iterations: number;
  residual: number;
  error: string | null;
}

export interface SolverOptions {
  tolerance?: number;
  maxIterations?: number;
  damping?: number;
}

export class SketchSolver {
  private tolerance: number;
  private maxIterations: number;
  private damping: number;

  constructor(options: SolverOptions = {}) {
    this.tolerance = options.tolerance ?? 1e-6;
    this.maxIterations = options.maxIterations ?? 100;
    this.damping = options.damping ?? 1e-3;
  }

  /**
   * Solve constraints for a project.
   * Updates point positions and circle radii to satisfy constraints.
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

    // Add line fixed lengths as constraints (residuals)
    const lineResiduals: Value[] = [];
    for (const line of project.lines) {
      if (line.fixedLength !== undefined) {
        const start = valueMap.points.get(line.start)!;
        const end = valueMap.points.get(line.end)!;
        const actualLength = end.sub(start).magnitude;
        const targetLength = V.C(line.fixedLength);
        const residual = V.sub(actualLength, targetLength);
        lineResiduals.push(residual);
      }
    }

    // Build residual function from all constraints
    const residualFn = (vars: Value[]) => {
      const residuals: Value[] = [...lineResiduals];

      for (const constraint of project.constraints) {
        const constraintResiduals = computeConstraintResiduals(constraint, valueMap);
        residuals.push(...constraintResiduals);
      }

      return residuals;
    };

    // If no variables to optimize, check if constraints are satisfied
    if (variables.length === 0) {
      const residuals = residualFn([]);
      const residualSumSquared = residuals.reduce(
        (sum, r) => sum + r.data ** 2,
        0
      );
      const residualMagnitude = Math.sqrt(residualSumSquared);

      return {
        converged: residualMagnitude < this.tolerance,
        iterations: 0,
        residual: residualMagnitude,
        error: residualMagnitude < this.tolerance ? null : 'Over-constrained (no free variables)',
      };
    }

    try {
      // Solve using Levenberg-Marquardt
      const result = nonlinearLeastSquares(variables, residualFn, {
        costTolerance: this.tolerance,
        maxIterations: this.maxIterations,
        initialDamping: this.damping,
        adaptiveDamping: true,
      });

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

      // Compute final residual magnitude
      const finalResiduals = residualFn(variables);
      const residualSumSquared = finalResiduals.reduce(
        (sum, r) => sum + r.data ** 2,
        0
      );
      const residualMagnitude = Math.sqrt(residualSumSquared);

      return {
        converged: result.success,
        iterations: result.iterations,
        residual: residualMagnitude,
        error: result.success ? null : result.convergenceReason,
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

  /**
   * Check if a project's constraints are satisfied (within tolerance).
   */
  isValid(project: Project): boolean {
    const result = this.solve(project);
    return result.converged && result.residual < this.tolerance;
  }
}
