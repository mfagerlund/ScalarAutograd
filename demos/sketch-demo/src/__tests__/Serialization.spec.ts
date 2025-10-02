/**
 * Tests for project serialization and deserialization
 */

import { describe, it, expect } from 'vitest';
import {
  Point,
  Line,
  Circle,
  LineConstraintType,
  ConstraintType,
  Project,
  serializeProject,
  deserializeProject,
  ParallelConstraint,
  PointOnCircleConstraint,
} from '../types';

describe('Project Serialization', () => {
  it('should serialize and deserialize a simple project with points', () => {
    const p1: Point = { x: 10, y: 20, pinned: true, label: 'P1' };
    const p2: Point = { x: 30, y: 40 };

    const project: Project = {
      name: 'Test Project',
      points: [p1, p2],
      lines: [],
      circles: [],
      constraints: [],
    };

    const serialized = serializeProject(project);
    const deserialized = deserializeProject(serialized);

    expect(deserialized.name).toBe('Test Project');
    expect(deserialized.points).toHaveLength(2);
    expect(deserialized.points[0].x).toBe(10);
    expect(deserialized.points[0].y).toBe(20);
    expect(deserialized.points[0].pinned).toBe(true);
    expect(deserialized.points[0].label).toBe('P1');
    expect(deserialized.points[1].x).toBe(30);
    expect(deserialized.points[1].y).toBe(40);
  });

  it('should serialize and deserialize lines with point references', () => {
    const p1: Point = { x: 0, y: 0 };
    const p2: Point = { x: 100, y: 0 };

    const line: Line = {
      start: p1,
      end: p2,
      constraintType: LineConstraintType.Horizontal,
      fixedLength: 100,
      label: 'Base Line',
    };

    const project: Project = {
      name: 'Line Project',
      points: [p1, p2],
      lines: [line],
      circles: [],
      constraints: [],
    };

    const serialized = serializeProject(project);
    const deserialized = deserializeProject(serialized);

    expect(deserialized.lines).toHaveLength(1);
    expect(deserialized.lines[0].start.x).toBe(0);
    expect(deserialized.lines[0].end.x).toBe(100);
    expect(deserialized.lines[0].constraintType).toBe(LineConstraintType.Horizontal);
    expect(deserialized.lines[0].fixedLength).toBe(100);
    expect(deserialized.lines[0].label).toBe('Base Line');

    // Verify that start and end reference the actual point objects
    expect(deserialized.lines[0].start).toBe(deserialized.points[0]);
    expect(deserialized.lines[0].end).toBe(deserialized.points[1]);
  });

  it('should serialize and deserialize circles with center point references', () => {
    const center: Point = { x: 50, y: 50 };

    const circle: Circle = {
      center,
      radius: 25,
      fixedRadius: true,
      label: 'Circle 1',
    };

    const project: Project = {
      name: 'Circle Project',
      points: [center],
      lines: [],
      circles: [circle],
      constraints: [],
    };

    const serialized = serializeProject(project);
    const deserialized = deserializeProject(serialized);

    expect(deserialized.circles).toHaveLength(1);
    expect(deserialized.circles[0].center.x).toBe(50);
    expect(deserialized.circles[0].center.y).toBe(50);
    expect(deserialized.circles[0].radius).toBe(25);
    expect(deserialized.circles[0].fixedRadius).toBe(true);
    expect(deserialized.circles[0].label).toBe('Circle 1');

    // Verify that center references the actual point object
    expect(deserialized.circles[0].center).toBe(deserialized.points[0]);
  });

  it('should serialize and deserialize constraints with entity references', () => {
    const p1: Point = { x: 0, y: 0 };
    const p2: Point = { x: 100, y: 0 };
    const p3: Point = { x: 0, y: 100 };
    const p4: Point = { x: 100, y: 100 };
    const center: Point = { x: 200, y: 200 };

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

    const circle: Circle = {
      center,
      radius: 50,
      fixedRadius: true,
    };

    const parallelConstraint: ParallelConstraint = {
      type: ConstraintType.Parallel,
      line1,
      line2,
      label: 'Parallel Lines',
    };

    const pointOnCircleConstraint: PointOnCircleConstraint = {
      type: ConstraintType.PointOnCircle,
      point: p1,
      circle,
    };

    const project: Project = {
      name: 'Constraint Project',
      points: [p1, p2, p3, p4, center],
      lines: [line1, line2],
      circles: [circle],
      constraints: [parallelConstraint, pointOnCircleConstraint],
    };

    const serialized = serializeProject(project);
    const deserialized = deserializeProject(serialized);

    expect(deserialized.constraints).toHaveLength(2);

    // Check parallel constraint
    const deserializedParallel = deserialized.constraints[0] as ParallelConstraint;
    expect(deserializedParallel.type).toBe(ConstraintType.Parallel);
    expect(deserializedParallel.label).toBe('Parallel Lines');
    expect(deserializedParallel.line1).toBe(deserialized.lines[0]);
    expect(deserializedParallel.line2).toBe(deserialized.lines[1]);

    // Check point-on-circle constraint
    const deserializedPointOnCircle = deserialized.constraints[1] as PointOnCircleConstraint;
    expect(deserializedPointOnCircle.type).toBe(ConstraintType.PointOnCircle);
    expect(deserializedPointOnCircle.point).toBe(deserialized.points[0]);
    expect(deserializedPointOnCircle.circle).toBe(deserialized.circles[0]);
  });

  it('should handle JSON round-trip', () => {
    const p1: Point = { x: 10, y: 20 };
    const p2: Point = { x: 30, y: 40 };
    const line: Line = {
      start: p1,
      end: p2,
      constraintType: LineConstraintType.Vertical,
    };

    const project: Project = {
      name: 'JSON Test',
      points: [p1, p2],
      lines: [line],
      circles: [],
      constraints: [],
    };

    // Serialize to JSON string
    const json = JSON.stringify(serializeProject(project));

    // Deserialize from JSON string
    const deserialized = deserializeProject(JSON.parse(json));

    expect(deserialized.name).toBe('JSON Test');
    expect(deserialized.points).toHaveLength(2);
    expect(deserialized.lines).toHaveLength(1);
    expect(deserialized.lines[0].start).toBe(deserialized.points[0]);
    expect(deserialized.lines[0].end).toBe(deserialized.points[1]);
  });

  it('should preserve object references across multiple lines sharing points', () => {
    const p1: Point = { x: 0, y: 0 };
    const p2: Point = { x: 100, y: 0 };
    const p3: Point = { x: 50, y: 100 };

    // Triangle: three lines sharing points
    const line1: Line = { start: p1, end: p2, constraintType: LineConstraintType.Free };
    const line2: Line = { start: p2, end: p3, constraintType: LineConstraintType.Free };
    const line3: Line = { start: p3, end: p1, constraintType: LineConstraintType.Free };

    const project: Project = {
      name: 'Triangle',
      points: [p1, p2, p3],
      lines: [line1, line2, line3],
      circles: [],
      constraints: [],
    };

    const serialized = serializeProject(project);
    const deserialized = deserializeProject(serialized);

    // Verify that shared points are the same object reference
    expect(deserialized.lines[0].end).toBe(deserialized.lines[1].start); // p2
    expect(deserialized.lines[1].end).toBe(deserialized.lines[2].start); // p3
    expect(deserialized.lines[2].end).toBe(deserialized.lines[0].start); // p1
  });
});
