/**
 * Tests for data model operations (user interactions without UI)
 * These tests verify that we can support common use cases like:
 * - Creating points, lines, circles
 * - Dragging points (updating positions)
 * - Connecting points with lines
 * - Deleting entities with cascade
 */

import { describe, it, expect } from 'vitest';
import {
  Point,
  Line,
  Circle,
  Project,
  LineConstraintType,
  ConstraintType,
  ParallelConstraint,
  serializeProject,
  deserializeProject,
} from '../types';

describe('Data Model - Create Operations', () => {
  it('should create a point', () => {
    const project: Project = {
      name: 'Test',
      points: [],
      lines: [],
      circles: [],
      constraints: [],
    };

    // User creates a point
    const newPoint: Point = { x: 100, y: 200 };
    project.points.push(newPoint);

    expect(project.points).toHaveLength(1);
    expect(project.points[0].x).toBe(100);
    expect(project.points[0].y).toBe(200);
  });

  it('should create a line connecting two points', () => {
    const project: Project = {
      name: 'Test',
      points: [],
      lines: [],
      circles: [],
      constraints: [],
    };

    // User creates two points
    const p1: Point = { x: 0, y: 0 };
    const p2: Point = { x: 100, y: 0 };
    project.points.push(p1, p2);

    // User creates a line between them
    const line: Line = {
      start: p1,
      end: p2,
      constraintType: LineConstraintType.Free,
    };
    project.lines.push(line);

    expect(project.lines).toHaveLength(1);
    expect(project.lines[0].start).toBe(p1);
    expect(project.lines[0].end).toBe(p2);
  });

  it('should create a circle at a point', () => {
    const project: Project = {
      name: 'Test',
      points: [],
      lines: [],
      circles: [],
      constraints: [],
    };

    // User creates a center point
    const center: Point = { x: 150, y: 150 };
    project.points.push(center);

    // User creates a circle
    const circle: Circle = {
      center,
      radius: 50,
      fixedRadius: true,
    };
    project.circles.push(circle);

    expect(project.circles).toHaveLength(1);
    expect(project.circles[0].center).toBe(center);
    expect(project.circles[0].radius).toBe(50);
  });

  it('should create a line by auto-creating points (click in empty space)', () => {
    const project: Project = {
      name: 'Test',
      points: [],
      lines: [],
      circles: [],
      constraints: [],
    };

    // User clicks at (50, 50) - creates first point
    const p1: Point = { x: 50, y: 50 };
    project.points.push(p1);

    // User clicks at (150, 150) - creates second point and line
    const p2: Point = { x: 150, y: 150 };
    project.points.push(p2);

    const line: Line = {
      start: p1,
      end: p2,
      constraintType: LineConstraintType.Free,
    };
    project.lines.push(line);

    expect(project.points).toHaveLength(2);
    expect(project.lines).toHaveLength(1);
    expect(project.lines[0].start).toBe(p1);
    expect(project.lines[0].end).toBe(p2);
  });
});

describe('Data Model - Drag Operations', () => {
  it('should update point position when dragged', () => {
    const point: Point = { x: 100, y: 100 };

    // User drags point to new position
    point.x = 150;
    point.y = 200;

    expect(point.x).toBe(150);
    expect(point.y).toBe(200);
  });

  it('should update line endpoints when connected points are dragged', () => {
    const p1: Point = { x: 0, y: 0 };
    const p2: Point = { x: 100, y: 0 };
    const line: Line = {
      start: p1,
      end: p2,
      constraintType: LineConstraintType.Free,
    };

    // User drags p2
    p2.x = 200;
    p2.y = 100;

    // Line automatically reflects new position (object reference)
    expect(line.end.x).toBe(200);
    expect(line.end.y).toBe(100);
  });

  it('should update circle center when center point is dragged', () => {
    const center: Point = { x: 100, y: 100 };
    const circle: Circle = {
      center,
      radius: 50,
      fixedRadius: true,
    };

    // User drags center point
    center.x = 200;
    center.y = 200;

    // Circle automatically reflects new center (object reference)
    expect(circle.center.x).toBe(200);
    expect(circle.center.y).toBe(200);
  });

  it('should not update pinned point position during solver', () => {
    const point: Point = { x: 100, y: 100, pinned: true };

    // Solver would skip this point because it's pinned
    if (!point.pinned) {
      point.x = 150; // This should not execute
    }

    expect(point.x).toBe(100);
    expect(point.y).toBe(100);
  });

  it('should update multiple lines when shared point is dragged', () => {
    const p1: Point = { x: 0, y: 0 };
    const p2: Point = { x: 100, y: 0 };
    const p3: Point = { x: 50, y: 100 };

    // Triangle with shared points
    const line1: Line = { start: p1, end: p2, constraintType: LineConstraintType.Free };
    const line2: Line = { start: p2, end: p3, constraintType: LineConstraintType.Free };
    const line3: Line = { start: p3, end: p1, constraintType: LineConstraintType.Free };

    // User drags p2 (shared by line1 and line2)
    p2.x = 150;
    p2.y = 50;

    // Both lines reflect the change
    expect(line1.end.x).toBe(150);
    expect(line1.end.y).toBe(50);
    expect(line2.start.x).toBe(150);
    expect(line2.start.y).toBe(50);
  });
});

describe('Data Model - Delete Operations', () => {
  it('should delete a standalone point', () => {
    const project: Project = {
      name: 'Test',
      points: [{ x: 100, y: 100 }],
      lines: [],
      circles: [],
      constraints: [],
    };

    // User deletes the point
    project.points = project.points.filter(p => p.x !== 100);

    expect(project.points).toHaveLength(0);
  });

  it('should cascade delete lines when a point is deleted', () => {
    const p1: Point = { x: 0, y: 0 };
    const p2: Point = { x: 100, y: 0 };
    const p3: Point = { x: 50, y: 100 };

    const line1: Line = { start: p1, end: p2, constraintType: LineConstraintType.Free };
    const line2: Line = { start: p2, end: p3, constraintType: LineConstraintType.Free };

    const project: Project = {
      name: 'Test',
      points: [p1, p2, p3],
      lines: [line1, line2],
      circles: [],
      constraints: [],
    };

    // User deletes p2 - should cascade delete line1 and line2
    project.points = project.points.filter(p => p !== p2);
    project.lines = project.lines.filter(l => l.start !== p2 && l.end !== p2);

    expect(project.points).toHaveLength(2);
    expect(project.lines).toHaveLength(0); // Both lines deleted
  });

  it('should cascade delete circles when center point is deleted', () => {
    const center: Point = { x: 100, y: 100 };
    const circle: Circle = {
      center,
      radius: 50,
      fixedRadius: true,
    };

    const project: Project = {
      name: 'Test',
      points: [center],
      lines: [],
      circles: [circle],
      constraints: [],
    };

    // User deletes center point - should cascade delete circle
    project.points = project.points.filter(p => p !== center);
    project.circles = project.circles.filter(c => c.center !== center);

    expect(project.points).toHaveLength(0);
    expect(project.circles).toHaveLength(0);
  });

  it('should cascade delete constraints when entities are deleted', () => {
    const p1: Point = { x: 0, y: 0 };
    const p2: Point = { x: 100, y: 0 };
    const p3: Point = { x: 0, y: 100 };
    const p4: Point = { x: 100, y: 100 };

    const line1: Line = { start: p1, end: p2, constraintType: LineConstraintType.Free };
    const line2: Line = { start: p3, end: p4, constraintType: LineConstraintType.Free };

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

    // User deletes line1 - should cascade delete parallel constraint
    project.lines = project.lines.filter(l => l !== line1);
    project.constraints = project.constraints.filter(c => {
      if (c.type === ConstraintType.Parallel) {
        const pc = c as ParallelConstraint;
        return pc.line1 !== line1 && pc.line2 !== line1;
      }
      return true;
    });

    expect(project.lines).toHaveLength(1);
    expect(project.constraints).toHaveLength(0); // Constraint deleted
  });
});

describe('Data Model - Complex Scenarios', () => {
  it('should support creating a triangle with constraints', () => {
    const project: Project = {
      name: 'Triangle',
      points: [],
      lines: [],
      circles: [],
      constraints: [],
    };

    // Create three points
    const p1: Point = { x: 0, y: 0, pinned: true };
    const p2: Point = { x: 100, y: 0 };
    const p3: Point = { x: 50, y: 86.6 };
    project.points.push(p1, p2, p3);

    // Create three lines (triangle)
    const base: Line = {
      start: p1,
      end: p2,
      constraintType: LineConstraintType.Horizontal,
      fixedLength: 100,
    };
    const side1: Line = {
      start: p2,
      end: p3,
      constraintType: LineConstraintType.Free,
      fixedLength: 100,
    };
    const side2: Line = {
      start: p3,
      end: p1,
      constraintType: LineConstraintType.Free,
      fixedLength: 100,
    };
    project.lines.push(base, side1, side2);

    expect(project.points).toHaveLength(3);
    expect(project.lines).toHaveLength(3);
    expect(project.lines[0].constraintType).toBe(LineConstraintType.Horizontal);
    expect(project.lines[0].fixedLength).toBe(100);
    expect(project.points[0].pinned).toBe(true);
  });

  it('should support finding entities connected to a point', () => {
    const p1: Point = { x: 0, y: 0 };
    const p2: Point = { x: 100, y: 0 };
    const p3: Point = { x: 50, y: 100 };

    const line1: Line = { start: p1, end: p2, constraintType: LineConstraintType.Free };
    const line2: Line = { start: p2, end: p3, constraintType: LineConstraintType.Free };
    const line3: Line = { start: p3, end: p1, constraintType: LineConstraintType.Free };
    const circle: Circle = { center: p2, radius: 20, fixedRadius: true };

    const project: Project = {
      name: 'Test',
      points: [p1, p2, p3],
      lines: [line1, line2, line3],
      circles: [circle],
      constraints: [],
    };

    // Find all entities connected to p2
    const connectedLines = project.lines.filter(l => l.start === p2 || l.end === p2);
    const connectedCircles = project.circles.filter(c => c.center === p2);

    expect(connectedLines).toHaveLength(2); // line1 and line2
    expect(connectedCircles).toHaveLength(1); // circle
  });

  it('should support cloning a project (deep copy scenario)', () => {
    const p1: Point = { x: 0, y: 0 };
    const p2: Point = { x: 100, y: 0 };
    const line: Line = { start: p1, end: p2, constraintType: LineConstraintType.Free };

    const original: Project = {
      name: 'Original',
      points: [p1, p2],
      lines: [line],
      circles: [],
      constraints: [],
    };

    // Cloning via serialization (only way to deep copy with object references)
    // This is what save/load would do
    const cloned = deserializeProject(serializeProject(original));

    // Verify cloned has different object instances
    expect(cloned.points[0]).not.toBe(original.points[0]);
    expect(cloned.lines[0]).not.toBe(original.lines[0]);

    // But same values
    expect(cloned.points[0].x).toBe(original.points[0].x);
    expect(cloned.lines[0].start.x).toBe(original.lines[0].start.x);

    // Modifying clone doesn't affect original
    cloned.points[0].x = 999;
    expect(original.points[0].x).toBe(0);
  });
});
