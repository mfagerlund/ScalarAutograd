/**
 * Tests for SketchSolver - constraint satisfaction via nonlinear least squares
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { SketchSolver } from '../SketchSolver';
import {
  Project,
  Point,
  Line,
  Circle,
  LineConstraintType,
  ConstraintType,
  ParallelConstraint,
  PerpendicularConstraint,
  PointOnLineConstraint,
  PointOnCircleConstraint,
  CollinearConstraint,
  AngleConstraint,
  TangentConstraint,
} from '../types';

describe('SketchSolver - Line Constraints', () => {
  let solver: SketchSolver;

  beforeEach(() => {
    solver = new SketchSolver({ tolerance: 1e-4, maxIterations: 200 });
  });

  it('should solve fixed length constraint', () => {
    const p1: Point = { x: 0, y: 0, pinned: true };
    const p2: Point = { x: 50, y: 0, pinned: true }; // Y is fixed, X will be optimized
    const p3: Point = { x: 50, y: 50 }; // Free point, wrong distance initially

    const line: Line = {
      start: p1,
      end: p3,
      constraintType: LineConstraintType.Free,
      fixedLength: 100, // Should be 100 units long
    };

    const project: Project = {
      name: 'Test',
      points: [p1, p2, p3],
      lines: [line],
      circles: [],
      constraints: [],
    };

    const result = solver.solve(project);

    expect(result.converged).toBe(true);
    expect(result.residual).toBeLessThan(1e-3);

    // Check that line length is now 100
    const dx = p3.x - p1.x;
    const dy = p3.y - p1.y;
    const length = Math.sqrt(dx * dx + dy * dy);
    expect(length).toBeCloseTo(100, 2);
  });

  it('should solve horizontal line constraint', () => {
    const p1: Point = { x: 0, y: 0, pinned: true };
    const p2: Point = { x: 100, y: 50 }; // Not horizontal initially

    // For horizontal constraint, we add a perpendicular constraint to Y axis
    const horizontalDirection: Line = {
      start: { x: 0, y: 0, pinned: true },
      end: { x: 1, y: 0, pinned: true },
      constraintType: LineConstraintType.Free,
    };

    const line: Line = {
      start: p1,
      end: p2,
      constraintType: LineConstraintType.Horizontal,
    };

    const perpendicularConstraint: PerpendicularConstraint = {
      type: ConstraintType.Perpendicular,
      line1: line,
      line2: {
        start: { x: 0, y: 0, pinned: true },
        end: { x: 0, y: 1, pinned: true },
        constraintType: LineConstraintType.Free,
      },
    };

    const project: Project = {
      name: 'Test',
      points: [p1, p2],
      lines: [line],
      circles: [],
      constraints: [perpendicularConstraint],
    };

    const result = solver.solve(project);

    expect(result.converged).toBe(true);
    expect(result.residual).toBeLessThan(1e-3);

    // Check that p2.y is approximately equal to p1.y (horizontal)
    expect(p2.y).toBeCloseTo(p1.y, 4);
  });
});

describe('SketchSolver - Parallel Constraint', () => {
  let solver: SketchSolver;

  beforeEach(() => {
    solver = new SketchSolver({ tolerance: 1e-4, maxIterations: 200 });
  });

  it('should solve parallel lines constraint', () => {
    const p1: Point = { x: 0, y: 0, pinned: true };
    const p2: Point = { x: 100, y: 0, pinned: true };
    const p3: Point = { x: 0, y: 50 };
    const p4: Point = { x: 100, y: 80 }; // Not parallel initially

    const line1: Line = {
      start: p1,
      end: p2,
      constraintType: LineConstraintType.Free,
    };

    const line2: Line = {
      start: p3,
      end: p4,
      constraintType: LineConstraintType.Free,
    };

    const parallelConstraint: ParallelConstraint = {
      type: ConstraintType.Parallel,
      line1,
      line2,
    };

    const project: Project = {
      name: 'Test',
      points: [p1, p2, p3, p4],
      lines: [line1, line2],
      circles: [],
      constraints: [parallelConstraint],
    };

    const result = solver.solve(project);

    expect(result.converged).toBe(true);
    expect(result.residual).toBeLessThan(1e-3);

    // Check that lines are parallel (cross product of directions is ~0)
    const dir1x = p2.x - p1.x;
    const dir1y = p2.y - p1.y;
    const dir2x = p4.x - p3.x;
    const dir2y = p4.y - p3.y;
    const cross = dir1x * dir2y - dir1y * dir2x;

    expect(Math.abs(cross)).toBeLessThan(1e-4);
  });
});

describe('SketchSolver - Perpendicular Constraint', () => {
  let solver: SketchSolver;

  beforeEach(() => {
    solver = new SketchSolver({ tolerance: 1e-4, maxIterations: 200 });
  });

  it('should solve perpendicular lines constraint', () => {
    const p1: Point = { x: 0, y: 0, pinned: true };
    const p2: Point = { x: 100, y: 0, pinned: true };
    const p3: Point = { x: 0, y: 0, pinned: true };
    const p4: Point = { x: 50, y: 50 }; // Not perpendicular initially

    const line1: Line = {
      start: p1,
      end: p2,
      constraintType: LineConstraintType.Free,
    };

    const line2: Line = {
      start: p3,
      end: p4,
      constraintType: LineConstraintType.Free,
    };

    const perpendicularConstraint: PerpendicularConstraint = {
      type: ConstraintType.Perpendicular,
      line1,
      line2,
    };

    const project: Project = {
      name: 'Test',
      points: [p1, p2, p3, p4],
      lines: [line1, line2],
      circles: [],
      constraints: [perpendicularConstraint],
    };

    const result = solver.solve(project);

    expect(result.converged).toBe(true);
    expect(result.residual).toBeLessThan(1e-3);

    // Check that lines are perpendicular (dot product is ~0)
    const dir1x = p2.x - p1.x;
    const dir1y = p2.y - p1.y;
    const dir2x = p4.x - p3.x;
    const dir2y = p4.y - p3.y;
    const dot = dir1x * dir2x + dir1y * dir2y;

    expect(Math.abs(dot)).toBeLessThan(1e-4);
  });
});

describe('SketchSolver - Point Constraints', () => {
  let solver: SketchSolver;

  beforeEach(() => {
    solver = new SketchSolver({ tolerance: 1e-4, maxIterations: 200 });
  });

  it('should solve point-on-line constraint', () => {
    const p1: Point = { x: 0, y: 0, pinned: true };
    const p2: Point = { x: 100, y: 0, pinned: true };
    const p3: Point = { x: 50, y: 25 }; // Not on line initially

    const line: Line = {
      start: p1,
      end: p2,
      constraintType: LineConstraintType.Free,
    };

    const pointOnLineConstraint: PointOnLineConstraint = {
      type: ConstraintType.PointOnLine,
      point: p3,
      line,
    };

    const project: Project = {
      name: 'Test',
      points: [p1, p2, p3],
      lines: [line],
      circles: [],
      constraints: [pointOnLineConstraint],
    };

    const result = solver.solve(project);

    expect(result.converged).toBe(true);
    expect(result.residual).toBeLessThan(1e-3);

    // Check that p3 is on the line (y should be ~0)
    expect(p3.y).toBeCloseTo(0, 4);
  });

  it('should solve collinear points constraint', () => {
    const p1: Point = { x: 0, y: 0, pinned: true };
    const p2: Point = { x: 100, y: 0, pinned: true };
    const p3: Point = { x: 50, y: 25 }; // Not collinear initially

    const collinearConstraint: CollinearConstraint = {
      type: ConstraintType.Collinear,
      point1: p1,
      point2: p2,
      point3: p3,
    };

    const project: Project = {
      name: 'Test',
      points: [p1, p2, p3],
      lines: [],
      circles: [],
      constraints: [collinearConstraint],
    };

    const result = solver.solve(project);

    expect(result.converged).toBe(true);
    expect(result.residual).toBeLessThan(1e-3);

    // Check that p3 is collinear (y should be ~0)
    expect(p3.y).toBeCloseTo(0, 4);
  });
});

describe('SketchSolver - Circle Constraints', () => {
  let solver: SketchSolver;

  beforeEach(() => {
    solver = new SketchSolver({ tolerance: 1e-4, maxIterations: 200 });
  });

  it('should solve point-on-circle constraint', () => {
    const center: Point = { x: 100, y: 100, pinned: true };
    const point: Point = { x: 150, y: 100 }; // Distance 50 from center

    const circle: Circle = {
      center,
      radius: 75, // Want point at distance 75
      fixedRadius: true,
    };

    const pointOnCircleConstraint: PointOnCircleConstraint = {
      type: ConstraintType.PointOnCircle,
      point,
      circle,
    };

    const project: Project = {
      name: 'Test',
      points: [center, point],
      lines: [],
      circles: [circle],
      constraints: [pointOnCircleConstraint],
    };

    const result = solver.solve(project);

    expect(result.converged).toBe(true);
    expect(result.residual).toBeLessThan(1e-3);

    // Check that point is on circle perimeter
    const dx = point.x - center.x;
    const dy = point.y - center.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    expect(distance).toBeCloseTo(75, 4);
  });

  it('should solve tangent line-circle constraint', () => {
    const center: Point = { x: 100, y: 100, pinned: true };
    const circle: Circle = {
      center,
      radius: 50,
      fixedRadius: true,
    };

    const p1: Point = { x: 0, y: 0, pinned: true };
    const p2: Point = { x: 200, y: 50 }; // Line not tangent initially

    const line: Line = {
      start: p1,
      end: p2,
      constraintType: LineConstraintType.Free,
    };

    const tangentConstraint: TangentConstraint = {
      type: ConstraintType.Tangent,
      line,
      circle,
    };

    const project: Project = {
      name: 'Test',
      points: [center, p1, p2],
      lines: [line],
      circles: [circle],
      constraints: [tangentConstraint],
    };

    const result = solver.solve(project);

    expect(result.converged).toBe(true);
    expect(result.residual).toBeLessThan(1e-3);

    // Check that distance from center to line equals radius
    const dirX = p2.x - p1.x;
    const dirY = p2.y - p1.y;
    const lineLen = Math.sqrt(dirX * dirX + dirY * dirY);
    const toCenter = { x: center.x - p1.x, y: center.y - p1.y };
    const cross = Math.abs(dirX * toCenter.y - dirY * toCenter.x);
    const distance = cross / lineLen;

    expect(distance).toBeCloseTo(50, 3);
  });
});

describe('SketchSolver - Complex Scenarios', () => {
  let solver: SketchSolver;

  beforeEach(() => {
    solver = new SketchSolver({ tolerance: 1e-4, maxIterations: 200 });
  });

  it('should solve rectangle with parallel and perpendicular constraints', () => {
    const p1: Point = { x: 0, y: 0, pinned: true };
    const p2: Point = { x: 100, y: 5 }; // Slight error
    const p3: Point = { x: 95, y: 55 }; // Slight error
    const p4: Point = { x: -5, y: 50 }; // Slight error

    const line1: Line = { start: p1, end: p2, constraintType: LineConstraintType.Free };
    const line2: Line = { start: p2, end: p3, constraintType: LineConstraintType.Free };
    const line3: Line = { start: p3, end: p4, constraintType: LineConstraintType.Free };
    const line4: Line = { start: p4, end: p1, constraintType: LineConstraintType.Free };

    const parallel1: ParallelConstraint = {
      type: ConstraintType.Parallel,
      line1: line1,
      line2: line3,
    };

    const parallel2: ParallelConstraint = {
      type: ConstraintType.Parallel,
      line1: line2,
      line2: line4,
    };

    const perp1: PerpendicularConstraint = {
      type: ConstraintType.Perpendicular,
      line1: line1,
      line2: line2,
    };

    const project: Project = {
      name: 'Rectangle',
      points: [p1, p2, p3, p4],
      lines: [line1, line2, line3, line4],
      circles: [],
      constraints: [parallel1, parallel2, perp1],
    };

    const result = solver.solve(project);

    expect(result.converged).toBe(true);
    expect(result.residual).toBeLessThan(1e-4);

    // Verify rectangle properties
    expect(p2.y).toBeCloseTo(0, 3); // Horizontal top edge
    expect(p4.x).toBeCloseTo(0, 3); // Vertical left edge
  });

  it('should handle pinned points correctly', () => {
    const p1: Point = { x: 0, y: 0, pinned: true };
    const p2: Point = { x: 100, y: 100, pinned: true };
    const p3: Point = { x: 50, y: 200 }; // Free to move

    const collinearConstraint: CollinearConstraint = {
      type: ConstraintType.Collinear,
      point1: p1,
      point2: p2,
      point3: p3,
    };

    const project: Project = {
      name: 'Pinned',
      points: [p1, p2, p3],
      lines: [],
      circles: [],
      constraints: [collinearConstraint],
    };

    const result = solver.solve(project);

    expect(result.converged).toBe(true);

    // p1 and p2 should not have moved
    expect(p1.x).toBe(0);
    expect(p1.y).toBe(0);
    expect(p2.x).toBe(100);
    expect(p2.y).toBe(100);

    // p3 should be on the line from p1 to p2
    expect(p3.x).toBeCloseTo(p3.y, 3); // Line is y=x
  });

  it('should handle over-constrained system gracefully', () => {
    const p1: Point = { x: 0, y: 0, pinned: true };
    const p2: Point = { x: 100, y: 0, pinned: true };

    const line: Line = {
      start: p1,
      end: p2,
      constraintType: LineConstraintType.Free,
      fixedLength: 50, // Impossible: pinned points are 100 apart
    };

    const project: Project = {
      name: 'Over-constrained',
      points: [p1, p2],
      lines: [line],
      circles: [],
      constraints: [],
    };

    const result = solver.solve(project);

    // Solver should either not converge or have high residual
    if (result.converged) {
      expect(result.residual).toBeGreaterThan(1e-3);
    } else {
      expect(result.converged).toBe(false);
    }
  });
});
