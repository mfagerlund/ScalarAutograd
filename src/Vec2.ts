import { Value } from './Value';
import { V } from './V';

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

  get trainables(): Value[] {
    return [this.x, this.y];
  }

  toString(): string {
    return `Vec2(${this.x.data}, ${this.y.data})`;
  }
}
