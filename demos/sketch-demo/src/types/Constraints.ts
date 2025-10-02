/**
 * Constraint types for geometric relationships
 */

import type { Line, Point, Circle } from './Entities';

/**
 * Base constraint interface
 */
export interface BaseConstraint {
  type: ConstraintType;
  label?: string;
}

/**
 * All constraint types
 */
export enum ConstraintType {
  // Line-to-line constraints
  Collinear = 'collinear',
  Parallel = 'parallel',
  Perpendicular = 'perpendicular',
  Angle = 'angle',

  // Point-entity constraints
  PointOnLine = 'point-on-line',
  PointOnCircle = 'point-on-circle',

  // Line-circle constraints
  Tangent = 'tangent',
  RadialAlignment = 'radial-alignment',  // Point-circle-point collinearity
}

/**
 * Two lines are collinear (lie on the same infinite line)
 */
export interface CollinearConstraint extends BaseConstraint {
  type: ConstraintType.Collinear;
  line1: Line;
  line2: Line;
}

/**
 * Two lines are parallel
 */
export interface ParallelConstraint extends BaseConstraint {
  type: ConstraintType.Parallel;
  line1: Line;
  line2: Line;
}

/**
 * Two lines are perpendicular (90 degrees)
 */
export interface PerpendicularConstraint extends BaseConstraint {
  type: ConstraintType.Perpendicular;
  line1: Line;
  line2: Line;
}

/**
 * Two lines meet at a specific angle (in degrees)
 */
export interface AngleConstraint extends BaseConstraint {
  type: ConstraintType.Angle;
  line1: Line;
  line2: Line;
  angleDegrees: number;
}

/**
 * Point lies anywhere on the line (not just between endpoints)
 */
export interface PointOnLineConstraint extends BaseConstraint {
  type: ConstraintType.PointOnLine;
  point: Point;
  line: Line;
}

/**
 * Point lies on the circle perimeter
 */
export interface PointOnCircleConstraint extends BaseConstraint {
  type: ConstraintType.PointOnCircle;
  point: Point;
  circle: Circle;
}

/**
 * Line is tangent to circle (perpendicular to radius at intersection)
 */
export interface TangentConstraint extends BaseConstraint {
  type: ConstraintType.Tangent;
  line: Line;
  circle: Circle;
}

/**
 * Point, circle center, and another point are collinear
 * Used for constraining tangent normals
 */
export interface RadialAlignmentConstraint extends BaseConstraint {
  type: ConstraintType.RadialAlignment;
  point1: Point;
  circle: Circle;
  point2: Point;
}

/**
 * Union type for all constraints
 */
export type Constraint =
  | CollinearConstraint
  | ParallelConstraint
  | PerpendicularConstraint
  | AngleConstraint
  | PointOnLineConstraint
  | PointOnCircleConstraint
  | TangentConstraint
  | RadialAlignmentConstraint;
