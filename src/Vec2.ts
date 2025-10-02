import { V } from './V';
import { Value } from './Value';

/**
 * 2D vector class with differentiable operations.
 * @public
 */
export class Vec2 {
  x: Value;
  y: Value;

  constructor(x: Value | number, y: Value | number) {
    this.x = typeof x === 'number' ? new Value(x) : x;
    this.y = typeof y === 'number' ? new Value(y) : y;
  }

  static W(x: number, y: number): Vec2 {
    return new Vec2(V.W(x), V.W(y));
  }

  static C(x: number, y: number): Vec2 {
    return new Vec2(V.C(x), V.C(y));
  }

  static zero(): Vec2 {
    return new Vec2(V.C(0), V.C(0));
  }

  static one(): Vec2 {
    return new Vec2(V.C(1), V.C(1));
  }

  get magnitude(): Value {
    return V.sqrt(V.add(V.square(this.x), V.square(this.y)));
  }

  get sqrMagnitude(): Value {
    return V.add(V.square(this.x), V.square(this.y));
  }

  get normalized(): Vec2 {
    const mag = this.magnitude;
    return new Vec2(V.div(this.x, mag), V.div(this.y, mag));
  }

  static dot(a: Vec2, b: Vec2): Value {
    return V.add(V.mul(a.x, b.x), V.mul(a.y, b.y));
  }

  add(other: Vec2): Vec2 {
    return new Vec2(V.add(this.x, other.x), V.add(this.y, other.y));
  }

  sub(other: Vec2): Vec2 {
    return new Vec2(V.sub(this.x, other.x), V.sub(this.y, other.y));
  }

  mul(scalar: Value | number): Vec2 {
    return new Vec2(V.mul(this.x, scalar), V.mul(this.y, scalar));
  }

  div(scalar: Value | number): Vec2 {
    return new Vec2(V.div(this.x, scalar), V.div(this.y, scalar));
  }

  /**
   * 2D cross product (returns scalar z-component of 3D cross product).
   * Also known as the "perpendicular dot product" or "wedge product".
   * Returns positive if b is counter-clockwise from a, negative if clockwise.
   */
  static cross(a: Vec2, b: Vec2): Value {
    return V.sub(V.mul(a.x, b.y), V.mul(a.y, b.x));
  }

  /**
   * Get perpendicular vector (rotated 90° counter-clockwise).
   * Useful for computing distances to lines.
   */
  get perpendicular(): Vec2 {
    return new Vec2(V.neg(this.y), this.x);
  }

  /**
   * Distance from this point to a line defined by two points.
   * Uses the perpendicular distance formula.
   */
  static distanceToLine(point: Vec2, lineStart: Vec2, lineEnd: Vec2): Value {
    const lineDir = lineEnd.sub(lineStart);
    const pointDir = point.sub(lineStart);
    const lineLength = lineDir.magnitude;
    const cross = Vec2.cross(lineDir, pointDir);
    return V.div(V.abs(cross), lineLength);
  }

  /**
   * Angle between two vectors in radians.
   * Returns value in range [0, π].
   */
  static angleBetween(a: Vec2, b: Vec2): Value {
    const dotProd = Vec2.dot(a, b);
    const magProduct = V.mul(a.magnitude, b.magnitude);
    const cosAngle = V.div(dotProd, magProduct);
    return V.acos(cosAngle);
  }

  get trainables(): Value[] {
    return [this.x, this.y];
  }

  toString(): string {
    return `Vec2(${this.x.data}, ${this.y.data})`;
  }
}
