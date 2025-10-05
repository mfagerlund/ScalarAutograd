/**
 * Constraint types for geometric relationships
 */

import type { Circle, Line, Point } from './Entities';

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
  // Point constraints
  Coincident = 'coincident',           // Two points share same position
  Midpoint = 'midpoint',               // Point is midpoint of line

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

  // Equality constraints
  EqualLength = 'equal-length',
  EqualRadius = 'equal-radius',

  // Circle-circle constraints
  CirclesTangent = 'circles-tangent',
  Concentric = 'concentric',           // Circles share same center

  // Symmetry
  Symmetry = 'symmetry',               // Two points symmetric about line
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
 * Multiple lines have equal length
 */
export interface EqualLengthConstraint extends BaseConstraint {
  type: ConstraintType.EqualLength;
  lines: Line[];
}

/**
 * Multiple circles have equal radius
 */
export interface EqualRadiusConstraint extends BaseConstraint {
  type: ConstraintType.EqualRadius;
  circles: Circle[];
}

/**
 * Two circles are tangent (difference or sum of radii equals distance between centers)
 */
export interface CirclesTangentConstraint extends BaseConstraint {
  type: ConstraintType.CirclesTangent;
  circle1: Circle;
  circle2: Circle;
}

/**
 * Two circles share the same center point (concentric)
 */
export interface ConcentricConstraint extends BaseConstraint {
  type: ConstraintType.Concentric;
  circle1: Circle;
  circle2: Circle;
}

/**
 * Two points share the same position (coincident)
 */
export interface CoincidentConstraint extends BaseConstraint {
  type: ConstraintType.Coincident;
  point1: Point;
  point2: Point;
}

/**
 * Point is at the midpoint of a line
 */
export interface MidpointConstraint extends BaseConstraint {
  type: ConstraintType.Midpoint;
  point: Point;
  line: Line;
}

/**
 * Two points are symmetric about a line (mirror symmetry)
 */
export interface SymmetryConstraint extends BaseConstraint {
  type: ConstraintType.Symmetry;
  point1: Point;
  point2: Point;
  symmetryLine: Line;
}

/**
 * Union type for all constraints
 */
export type Constraint =
  | CoincidentConstraint
  | MidpointConstraint
  | CollinearConstraint
  | ParallelConstraint
  | PerpendicularConstraint
  | AngleConstraint
  | PointOnLineConstraint
  | PointOnCircleConstraint
  | TangentConstraint
  | RadialAlignmentConstraint
  | EqualLengthConstraint
  | EqualRadiusConstraint
  | CirclesTangentConstraint
  | ConcentricConstraint
  | SymmetryConstraint;
