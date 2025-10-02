/**
 * Project container and serialization
 */

import type {
    AngleConstraint,
    CollinearConstraint,
    Constraint,
    ParallelConstraint,
    PerpendicularConstraint,
    PointOnCircleConstraint,
    PointOnLineConstraint,
    RadialAlignmentConstraint,
    TangentConstraint,
} from './Constraints';
import { ConstraintType } from './Constraints';
import type {
    Circle,
    Entity,
    Line,
    Point,
} from './Entities';
import { LineConstraintType } from './Entities';

/**
 * Project - the serializable container for a sketch
 */
export interface Project {
  name: string;
  points: Point[];
  lines: Line[];
  circles: Circle[];
  constraints: Constraint[];
}

/**
 * Serialized entity types (with IDs for storage only)
 */
export interface SerializedPoint {
  id: string;
  x: number;
  y: number;
  pinned?: boolean;
  label?: string;
}

export interface SerializedLine {
  id: string;
  startPointId: string;
  endPointId: string;
  constraintType: LineConstraintType;
  fixedLength?: number;
  label?: string;
}

export interface SerializedCircle {
  id: string;
  centerPointId: string;
  radius: number;
  fixedRadius: boolean;
  label?: string;
}

export interface SerializedConstraint {
  id: string;
  type: ConstraintType;
  label?: string;
  // Entity references as IDs
  line1Id?: string;
  line2Id?: string;
  pointId?: string;
  point1Id?: string;
  point2Id?: string;
  lineId?: string;
  circleId?: string;
  // Constraint parameters
  angleDegrees?: number;
}

/**
 * Serialized project (with IDs)
 */
export interface SerializedProject {
  name: string;
  points: SerializedPoint[];
  lines: SerializedLine[];
  circles: SerializedCircle[];
  constraints: SerializedConstraint[];
}

/**
 * Serialize a project to JSON-compatible format with IDs
 */
export function serializeProject(project: Project): SerializedProject {
  // Create ID maps for entities
  const pointIds = new Map<Point, string>();
  const lineIds = new Map<Line, string>();
  const circleIds = new Map<Circle, string>();

  // Assign IDs to all entities
  project.points.forEach((p, i) => pointIds.set(p, `p${i}`));
  project.lines.forEach((l, i) => lineIds.set(l, `l${i}`));
  project.circles.forEach((c, i) => circleIds.set(c, `c${i}`));

  // Serialize points
  const points: SerializedPoint[] = project.points.map(p => ({
    id: pointIds.get(p)!,
    x: p.x,
    y: p.y,
    pinned: p.pinned,
    label: p.label,
  }));

  // Serialize lines
  const lines: SerializedLine[] = project.lines.map(l => ({
    id: lineIds.get(l)!,
    startPointId: pointIds.get(l.start)!,
    endPointId: pointIds.get(l.end)!,
    constraintType: l.constraintType,
    fixedLength: l.fixedLength,
    label: l.label,
  }));

  // Serialize circles
  const circles: SerializedCircle[] = project.circles.map(c => ({
    id: circleIds.get(c)!,
    centerPointId: pointIds.get(c.center)!,
    radius: c.radius,
    fixedRadius: c.fixedRadius,
    label: c.label,
  }));

  // Serialize constraints
  const constraints: SerializedConstraint[] = project.constraints.map(c => {
    const base = {
      id: `con${project.constraints.indexOf(c)}`,
      type: c.type,
      label: c.label,
    };

    switch (c.type) {
      case ConstraintType.Collinear:
      case ConstraintType.Parallel:
      case ConstraintType.Perpendicular:
        return {
          ...base,
          line1Id: lineIds.get((c as CollinearConstraint | ParallelConstraint | PerpendicularConstraint).line1)!,
          line2Id: lineIds.get((c as CollinearConstraint | ParallelConstraint | PerpendicularConstraint).line2)!,
        };
      case ConstraintType.Angle:
        return {
          ...base,
          line1Id: lineIds.get((c as AngleConstraint).line1)!,
          line2Id: lineIds.get((c as AngleConstraint).line2)!,
          angleDegrees: (c as AngleConstraint).angleDegrees,
        };
      case ConstraintType.PointOnLine:
        return {
          ...base,
          pointId: pointIds.get((c as PointOnLineConstraint).point)!,
          lineId: lineIds.get((c as PointOnLineConstraint).line)!,
        };
      case ConstraintType.PointOnCircle:
        return {
          ...base,
          pointId: pointIds.get((c as PointOnCircleConstraint).point)!,
          circleId: circleIds.get((c as PointOnCircleConstraint).circle)!,
        };
      case ConstraintType.Tangent:
        return {
          ...base,
          lineId: lineIds.get((c as TangentConstraint).line)!,
          circleId: circleIds.get((c as TangentConstraint).circle)!,
        };
      case ConstraintType.RadialAlignment:
        return {
          ...base,
          point1Id: pointIds.get((c as RadialAlignmentConstraint).point1)!,
          circleId: circleIds.get((c as RadialAlignmentConstraint).circle)!,
          point2Id: pointIds.get((c as RadialAlignmentConstraint).point2)!,
        };
      case ConstraintType.EqualLength:
        return {
          ...base,
          lineIds: (c as import('./Constraints').EqualLengthConstraint).lines.map(l => lineIds.get(l)!),
        };
      case ConstraintType.EqualRadius:
        return {
          ...base,
          circleIds: (c as import('./Constraints').EqualRadiusConstraint).circles.map(circle => circleIds.get(circle)!),
        };
      case ConstraintType.CirclesTangent:
      case ConstraintType.Concentric:
        return {
          ...base,
          circle1Id: circleIds.get((c as import('./Constraints').CirclesTangentConstraint | import('./Constraints').ConcentricConstraint).circle1)!,
          circle2Id: circleIds.get((c as import('./Constraints').CirclesTangentConstraint | import('./Constraints').ConcentricConstraint).circle2)!,
        };
      default:
        return base as any;
    }
  });

  return {
    name: project.name,
    points,
    lines,
    circles,
    constraints,
  };
}

/**
 * Deserialize a project from JSON format (resolves IDs to object references)
 */
export function deserializeProject(serialized: SerializedProject): Project {
  // Create lookup maps
  const pointMap = new Map<string, Point>();
  const lineMap = new Map<string, Line>();
  const circleMap = new Map<string, Circle>();

  // Deserialize points first (no dependencies)
  const points: Point[] = serialized.points.map(sp => {
    const point: Point = {
      x: sp.x,
      y: sp.y,
      pinned: sp.pinned,
      label: sp.label,
    };
    pointMap.set(sp.id, point);
    return point;
  });

  // Deserialize lines (depend on points)
  const lines: Line[] = serialized.lines.map(sl => {
    const line: Line = {
      start: pointMap.get(sl.startPointId)!,
      end: pointMap.get(sl.endPointId)!,
      constraintType: sl.constraintType,
      fixedLength: sl.fixedLength,
      label: sl.label,
    };
    lineMap.set(sl.id, line);
    return line;
  });

  // Deserialize circles (depend on points)
  const circles: Circle[] = serialized.circles.map(sc => {
    const circle: Circle = {
      center: pointMap.get(sc.centerPointId)!,
      radius: sc.radius,
      fixedRadius: sc.fixedRadius,
      label: sc.label,
    };
    circleMap.set(sc.id, circle);
    return circle;
  });

  // Deserialize constraints (depend on everything)
  const constraints: Constraint[] = serialized.constraints.map(sc => {
    switch (sc.type) {
      case ConstraintType.Collinear:
        return {
          type: sc.type,
          line1: lineMap.get(sc.line1Id!)!,
          line2: lineMap.get(sc.line2Id!)!,
          label: sc.label,
        } as CollinearConstraint;
      case ConstraintType.Parallel:
        return {
          type: sc.type,
          line1: lineMap.get(sc.line1Id!)!,
          line2: lineMap.get(sc.line2Id!)!,
          label: sc.label,
        } as ParallelConstraint;
      case ConstraintType.Perpendicular:
        return {
          type: sc.type,
          line1: lineMap.get(sc.line1Id!)!,
          line2: lineMap.get(sc.line2Id!)!,
          label: sc.label,
        } as PerpendicularConstraint;
      case ConstraintType.Angle:
        return {
          type: sc.type,
          line1: lineMap.get(sc.line1Id!)!,
          line2: lineMap.get(sc.line2Id!)!,
          angleDegrees: sc.angleDegrees!,
          label: sc.label,
        } as AngleConstraint;
      case ConstraintType.PointOnLine:
        return {
          type: sc.type,
          point: pointMap.get(sc.pointId!)!,
          line: lineMap.get(sc.lineId!)!,
          label: sc.label,
        } as PointOnLineConstraint;
      case ConstraintType.PointOnCircle:
        return {
          type: sc.type,
          point: pointMap.get(sc.pointId!)!,
          circle: circleMap.get(sc.circleId!)!,
          label: sc.label,
        } as PointOnCircleConstraint;
      case ConstraintType.Tangent:
        return {
          type: sc.type,
          line: lineMap.get(sc.lineId!)!,
          circle: circleMap.get(sc.circleId!)!,
          label: sc.label,
        } as TangentConstraint;
      case ConstraintType.RadialAlignment:
        return {
          type: sc.type,
          point1: pointMap.get(sc.point1Id!)!,
          circle: circleMap.get(sc.circleId!)!,
          point2: pointMap.get(sc.point2Id!)!,
          label: sc.label,
        } as RadialAlignmentConstraint;
      default:
        throw new Error(`Unknown constraint type: ${sc.type}`);
    }
  });

  return {
    name: serialized.name,
    points,
    lines,
    circles,
    constraints,
  };
}

/**
 * Helper to get all entities as an array
 */
export function getAllEntities(project: Project): Entity[] {
  return [...project.points, ...project.lines, ...project.circles];
}

/**
 * Create a demo project with points, lines, and circles
 */
export function createDemoProject(): Project {
  // Create some points
  const p1: Point = { x: 100, y: 100, pinned: true };
  const p2: Point = { x: 300, y: 100 };
  const p3: Point = { x: 200, y: 250 };
  const p4: Point = { x: 400, y: 200 };
  const p5: Point = { x: 400, y: 350 };

  // Create lines connecting points
  const line1: Line = {
    start: p1,
    end: p2,
    constraintType: LineConstraintType.Horizontal,
    fixedLength: 200,
  };

  const line2: Line = {
    start: p2,
    end: p3,
    constraintType: LineConstraintType.Free,
  };

  const line3: Line = {
    start: p3,
    end: p1,
    constraintType: LineConstraintType.Free,
  };

  const line4: Line = {
    start: p4,
    end: p5,
    constraintType: LineConstraintType.Vertical,
  };

  // Create circles
  const circle1: Circle = {
    center: p3,
    radius: 50,
    fixedRadius: false,
  };

  const circle2: Circle = {
    center: p4,
    radius: 75,
    fixedRadius: false,
  };

  // Add some example constraints
  const constraints: Constraint[] = [
    // Make line2 and line3 equal length
    {
      type: ConstraintType.Parallel,
      line1: line3,
      line2: line4,
    } as ParallelConstraint,
  ];

  return {
    name: 'Demo Project',
    points: [p1, p2, p3, p4, p5],
    lines: [line1, line2, line3, line4],
    circles: [circle1, circle2],
    constraints,
  };
}
