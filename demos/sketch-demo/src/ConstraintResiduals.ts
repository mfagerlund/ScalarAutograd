/**
 * Residual functions for geometric constraints.
 * Each function returns an array of Value objects representing constraint violations.
 * When constraints are satisfied, all residuals should be zero.
 */

import { Value, V, Vec2 } from '../../../src';
import type {
  Constraint,
  CollinearConstraint,
  ParallelConstraint,
  PerpendicularConstraint,
  AngleConstraint,
  PointOnLineConstraint,
  PointOnCircleConstraint,
  TangentConstraint,
  RadialAlignmentConstraint,
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
    default:
      return [];
  }
}

/**
 * Collinear constraint: Three points lie on the same line.
 * Residual: Area of triangle formed by three points should be zero.
 * Area = 0.5 * |cross product of two edge vectors|
 */
function computeCollinearResiduals(
  constraint: CollinearConstraint,
  valueMap: ValueMap
): Value[] {
  const p1 = valueMap.points.get(constraint.point1)!;
  const p2 = valueMap.points.get(constraint.point2)!;
  const p3 = valueMap.points.get(constraint.point3)!;

  const v1 = p2.sub(p1);
  const v2 = p3.sub(p1);
  const cross = Vec2.cross(v1, v2);

  return [cross]; // Should be 0 when collinear
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
