/**
 * Core entity types: Point, Line, Circle
 */

/**
 * Represents a point in 2D space.
 * Points are the fundamental building blocks - lines and circles reference points.
 */
export interface Point {
  x: number;
  y: number;
  pinned?: boolean;  // If true, point position is fixed (not optimized by solver)
  label?: string;
}

/**
 * Line orientation/constraint type
 */
export enum LineConstraintType {
  Free = 'free',           // No orientation constraint
  Horizontal = 'horizontal', // Forces y1 = y2
  Vertical = 'vertical'      // Forces x1 = x2
}

/**
 * Represents a line segment connecting two points.
 * Lines can have orientation constraints (horizontal/vertical) and optional fixed length.
 */
export interface Line {
  start: Point;  // Direct reference to Point object
  end: Point;    // Direct reference to Point object
  constraintType: LineConstraintType;
  fixedLength?: number;  // If undefined, length is free
  label?: string;
}

/**
 * Represents a circle with a center point and radius.
 */
export interface Circle {
  center: Point;  // Direct reference to Point object
  radius: number;
  fixedRadius: boolean;   // If false, radius can be optimized by solver
  label?: string;
}

/**
 * Union type for all entity types
 */
export type Entity = Point | Line | Circle;

/**
 * Type guards for entity discrimination
 */
export function isPoint(entity: Entity): entity is Point {
  return 'x' in entity && 'y' in entity;
}

export function isLine(entity: Entity): entity is Line {
  return 'start' in entity && 'end' in entity;
}

export function isCircle(entity: Entity): entity is Circle {
  return 'center' in entity && 'radius' in entity;
}
