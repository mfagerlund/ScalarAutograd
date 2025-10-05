/**
 * Residual functions for geometric constraints.
 * Each function returns an array of Value objects representing constraint violations.
 * When constraints are satisfied, all residuals should be zero.
 */

import { V, Value, Vec2 } from '../../../src';
import type {
    AngleConstraint,
    CirclesTangentConstraint,
    CoincidentConstraint,
    CollinearConstraint,
    ConcentricConstraint,
    Constraint,
    EqualLengthConstraint,
    EqualRadiusConstraint,
    MidpointConstraint,
    ParallelConstraint,
    PerpendicularConstraint,
    PointOnCircleConstraint,
    PointOnLineConstraint,
    RadialAlignmentConstraint,
    SymmetryConstraint,
    TangentConstraint,
} from './types/Constraints';
import { ConstraintType } from './types/Constraints';

/**
 * Map from entity objects to their corresponding Value objects for optimization.
 * Points map to Vec2, circles map to their radius Value.
 */
export interface ValueMap {
  points: Map<any, Vec2>;
  circleRadii: Map<any, Value>;
}

/**
 * Compute residuals for a constraint.
 * Returns array of Value objects (residuals) that should be zero when constraint is satisfied.
 */
export function computeConstraintResiduals(
  constraint: Constraint,
  valueMap: ValueMap
): Value[] {
  switch (constraint.type) {
    case ConstraintType.Coincident:
      return computeCoincidentResiduals(constraint, valueMap);
    case ConstraintType.Midpoint:
      return computeMidpointResiduals(constraint, valueMap);
    case ConstraintType.Collinear:
      return computeCollinearResiduals(constraint, valueMap);
    case ConstraintType.Parallel:
      return computeParallelResiduals(constraint, valueMap);
    case ConstraintType.Perpendicular:
      return computePerpendicularResiduals(constraint, valueMap);
    case ConstraintType.Angle:
      return computeAngleResiduals(constraint, valueMap);
    case ConstraintType.PointOnLine:
      return computePointOnLineResiduals(constraint, valueMap);
    case ConstraintType.PointOnCircle:
      return computePointOnCircleResiduals(constraint, valueMap);
    case ConstraintType.Tangent:
      return computeTangentResiduals(constraint, valueMap);
    case ConstraintType.RadialAlignment:
      return computeRadialAlignmentResiduals(constraint, valueMap);
    case ConstraintType.EqualLength:
      return computeEqualLengthResiduals(constraint, valueMap);
    case ConstraintType.EqualRadius:
      return computeEqualRadiusResiduals(constraint, valueMap);
    case ConstraintType.CirclesTangent:
      return computeCirclesTangentResiduals(constraint, valueMap);
    case ConstraintType.Concentric:
      return computeConcentricResiduals(constraint, valueMap);
    case ConstraintType.Symmetry:
      return computeSymmetryResiduals(constraint, valueMap);
    default:
      return [];
  }
}

/**
 * Collinear constraint: Two lines lie on the same infinite line.
 * Residual: Cross product of direction vectors should be zero (parallel)
 * AND one line's start point should lie on the other line's infinite extension.
 */
function computeCollinearResiduals(
  constraint: CollinearConstraint,
  valueMap: ValueMap
): Value[] {
  const l1Start = valueMap.points.get(constraint.line1.start)!;
  const l1End = valueMap.points.get(constraint.line1.end)!;
  const l2Start = valueMap.points.get(constraint.line2.start)!;
  const l2End = valueMap.points.get(constraint.line2.end)!;

  const dir1 = l1End.sub(l1Start);
  const dir2 = l2End.sub(l2Start);

  // Lines must be parallel (cross product = 0)
  const cross = Vec2.cross(dir1, dir2);

  // Line2's start point must lie on line1's infinite extension
  const distance = Vec2.distanceToLine(l2Start, l1Start, l1End);

  return [cross, distance]; // Both should be 0 when collinear
}

/**
 * Parallel constraint: Two lines are parallel.
 * Residual: Cross product of direction vectors should be zero.
 */
function computeParallelResiduals(
  constraint: ParallelConstraint,
  valueMap: ValueMap
): Value[] {
  const l1Start = valueMap.points.get(constraint.line1.start)!;
  const l1End = valueMap.points.get(constraint.line1.end)!;
  const l2Start = valueMap.points.get(constraint.line2.start)!;
  const l2End = valueMap.points.get(constraint.line2.end)!;

  const dir1 = l1End.sub(l1Start);
  const dir2 = l2End.sub(l2Start);

  const cross = Vec2.cross(dir1, dir2);

  return [cross]; // Should be 0 when parallel
}

/**
 * Perpendicular constraint: Two lines are perpendicular.
 * Residual: Dot product of direction vectors should be zero.
 */
function computePerpendicularResiduals(
  constraint: PerpendicularConstraint,
  valueMap: ValueMap
): Value[] {
  const l1Start = valueMap.points.get(constraint.line1.start)!;
  const l1End = valueMap.points.get(constraint.line1.end)!;
  const l2Start = valueMap.points.get(constraint.line2.start)!;
  const l2End = valueMap.points.get(constraint.line2.end)!;

  const dir1 = l1End.sub(l1Start);
  const dir2 = l2End.sub(l2Start);

  const dot = Vec2.dot(dir1, dir2);

  return [dot]; // Should be 0 when perpendicular
}

/**
 * Angle constraint: Two lines meet at a specific angle.
 * Residual: Difference between actual angle and target angle.
 */
function computeAngleResiduals(
  constraint: AngleConstraint,
  valueMap: ValueMap
): Value[] {
  const l1Start = valueMap.points.get(constraint.line1.start)!;
  const l1End = valueMap.points.get(constraint.line1.end)!;
  const l2Start = valueMap.points.get(constraint.line2.start)!;
  const l2End = valueMap.points.get(constraint.line2.end)!;

  const dir1 = l1End.sub(l1Start);
  const dir2 = l2End.sub(l2Start);

  const actualAngle = Vec2.angleBetween(dir1, dir2);
  // Convert target angle from degrees to radians
  const targetAngleRadians = V.C(constraint.angleDegrees * Math.PI / 180);

  const residual = V.sub(actualAngle, targetAngleRadians);

  return [residual]; // Should be 0 when angle matches
}

/**
 * Point on line constraint: A point lies on a line (ray).
 * Residual: Perpendicular distance from point to line should be zero.
 */
function computePointOnLineResiduals(
  constraint: PointOnLineConstraint,
  valueMap: ValueMap
): Value[] {
  const point = valueMap.points.get(constraint.point)!;
  const lineStart = valueMap.points.get(constraint.line.start)!;
  const lineEnd = valueMap.points.get(constraint.line.end)!;

  const distance = Vec2.distanceToLine(point, lineStart, lineEnd);

  return [distance]; // Should be 0 when point is on line
}

/**
 * Point on circle constraint: A point lies on a circle's perimeter.
 * Residual: Distance from center to point minus radius should be zero.
 */
function computePointOnCircleResiduals(
  constraint: PointOnCircleConstraint,
  valueMap: ValueMap
): Value[] {
  const point = valueMap.points.get(constraint.point)!;
  const center = valueMap.points.get(constraint.circle.center)!;
  const radius = valueMap.circleRadii.get(constraint.circle) ?? V.C(constraint.circle.radius);

  const distance = point.sub(center).magnitude;
  const residual = V.sub(distance, radius);

  return [residual]; // Should be 0 when point is on circle
}

/**
 * Tangent constraint: A line is tangent to a circle.
 * Residual: Distance from circle center to line should equal radius.
 */
function computeTangentResiduals(
  constraint: TangentConstraint,
  valueMap: ValueMap
): Value[] {
  const lineStart = valueMap.points.get(constraint.line.start)!;
  const lineEnd = valueMap.points.get(constraint.line.end)!;
  const center = valueMap.points.get(constraint.circle.center)!;
  const radius = valueMap.circleRadii.get(constraint.circle) ?? V.C(constraint.circle.radius);

  const distance = Vec2.distanceToLine(center, lineStart, lineEnd);
  const residual = V.sub(distance, radius);

  return [residual]; // Should be 0 when line is tangent
}

/**
 * Radial alignment constraint: Three entities (point1, circle center, point2) are collinear.
 * This is the same as collinear constraint but specifically for radial patterns.
 * Residual: Cross product should be zero.
 */
function computeRadialAlignmentResiduals(
  constraint: RadialAlignmentConstraint,
  valueMap: ValueMap
): Value[] {
  const p1 = valueMap.points.get(constraint.point1)!;
  const center = valueMap.points.get(constraint.circle.center)!;
  const p2 = valueMap.points.get(constraint.point2)!;

  const v1 = center.sub(p1);
  const v2 = p2.sub(p1);
  const cross = Vec2.cross(v1, v2);

  return [cross]; // Should be 0 when collinear
}

/**
 * Equal length constraint: Multiple lines have equal length.
 * Residual: Difference between each line's length and the first line's length.
 * Creates N-1 residuals for N lines.
 */
function computeEqualLengthResiduals(
  constraint: EqualLengthConstraint,
  valueMap: ValueMap
): Value[] {
  if (constraint.lines.length < 2) return [];

  const residuals: Value[] = [];

  // Get first line's length as reference
  const firstStart = valueMap.points.get(constraint.lines[0].start)!;
  const firstEnd = valueMap.points.get(constraint.lines[0].end)!;
  const firstLength = firstEnd.sub(firstStart).magnitude;

  // Compare all other lines to first line
  for (let i = 1; i < constraint.lines.length; i++) {
    const start = valueMap.points.get(constraint.lines[i].start)!;
    const end = valueMap.points.get(constraint.lines[i].end)!;
    const length = end.sub(start).magnitude;

    residuals.push(V.sub(length, firstLength));
  }

  return residuals; // All should be 0 when lengths are equal
}

/**
 * Equal radius constraint: Multiple circles have equal radius.
 * Residual: Difference between each circle's radius and the first circle's radius.
 * Creates N-1 residuals for N circles.
 */
function computeEqualRadiusResiduals(
  constraint: EqualRadiusConstraint,
  valueMap: ValueMap
): Value[] {
  if (constraint.circles.length < 2) return [];

  const residuals: Value[] = [];

  // Get first circle's radius as reference
  const firstRadius = valueMap.circleRadii.get(constraint.circles[0]) ?? V.C(constraint.circles[0].radius);

  // Compare all other circles to first circle
  for (let i = 1; i < constraint.circles.length; i++) {
    const radius = valueMap.circleRadii.get(constraint.circles[i]) ?? V.C(constraint.circles[i].radius);

    residuals.push(V.sub(radius, firstRadius));
  }

  return residuals; // All should be 0 when radii are equal
}

/**
 * Circles tangent constraint: Two circles are tangent (touch at exactly one point).
 * Residual: Distance between centers equals sum (external tangent) or difference (internal tangent) of radii.
 * We use external tangent: distance = r1 + r2
 */
function computeCirclesTangentResiduals(
  constraint: CirclesTangentConstraint,
  valueMap: ValueMap
): Value[] {
  const center1 = valueMap.points.get(constraint.circle1.center)!;
  const center2 = valueMap.points.get(constraint.circle2.center)!;
  const radius1 = valueMap.circleRadii.get(constraint.circle1) ?? V.C(constraint.circle1.radius);
  const radius2 = valueMap.circleRadii.get(constraint.circle2) ?? V.C(constraint.circle2.radius);

  const distance = center2.sub(center1).magnitude;

  // External tangent: distance = r1 + r2
  const targetDistance = V.add(radius1, radius2);
  const residual = V.sub(distance, targetDistance);

  return [residual]; // Should be 0 when circles are tangent
}

/**
 * Coincident constraint: Two points share the same position.
 * Residual: Distance between points should be zero (both x and y coordinates match).
 */
function computeCoincidentResiduals(
  constraint: CoincidentConstraint,
  valueMap: ValueMap
): Value[] {
  const p1 = valueMap.points.get(constraint.point1)!;
  const p2 = valueMap.points.get(constraint.point2)!;

  const dx = V.sub(p1.x, p2.x);
  const dy = V.sub(p1.y, p2.y);

  return [dx, dy]; // Both should be 0 when points are coincident
}

/**
 * Midpoint constraint: Point is at the midpoint of a line.
 * Residual: Point position should equal average of line endpoints.
 */
function computeMidpointResiduals(
  constraint: MidpointConstraint,
  valueMap: ValueMap
): Value[] {
  const point = valueMap.points.get(constraint.point)!;
  const lineStart = valueMap.points.get(constraint.line.start)!;
  const lineEnd = valueMap.points.get(constraint.line.end)!;

  const midpoint = lineStart.add(lineEnd).mul(V.C(0.5));

  const dx = V.sub(point.x, midpoint.x);
  const dy = V.sub(point.y, midpoint.y);

  return [dx, dy]; // Both should be 0 when point is at midpoint
}

/**
 * Concentric constraint: Two circles share the same center.
 * Residual: Distance between centers should be zero.
 */
function computeConcentricResiduals(
  constraint: ConcentricConstraint,
  valueMap: ValueMap
): Value[] {
  const center1 = valueMap.points.get(constraint.circle1.center)!;
  const center2 = valueMap.points.get(constraint.circle2.center)!;

  const dx = V.sub(center1.x, center2.x);
  const dy = V.sub(center1.y, center2.y);

  return [dx, dy]; // Both should be 0 when centers are coincident
}

/**
 * Symmetry constraint: Two points are symmetric about a line (mirror reflection).
 * Residual:
 * 1. Midpoint of p1-p2 should lie on symmetry line
 * 2. Line p1-p2 should be perpendicular to symmetry line
 */
function computeSymmetryResiduals(
  constraint: SymmetryConstraint,
  valueMap: ValueMap
): Value[] {
  const p1 = valueMap.points.get(constraint.point1)!;
  const p2 = valueMap.points.get(constraint.point2)!;
  const lineStart = valueMap.points.get(constraint.symmetryLine.start)!;
  const lineEnd = valueMap.points.get(constraint.symmetryLine.end)!;

  // Midpoint of p1-p2
  const midpoint = p1.add(p2).mul(V.C(0.5));

  // Residual 1: Midpoint should lie on symmetry line
  const distanceToLine = Vec2.distanceToLine(midpoint, lineStart, lineEnd);

  // Residual 2: Vector p1-p2 should be perpendicular to symmetry line
  const p1p2 = p2.sub(p1);
  const lineDir = lineEnd.sub(lineStart);
  const dotProduct = Vec2.dot(p1p2, lineDir);

  return [distanceToLine, dotProduct]; // Both should be 0 when symmetric
}
