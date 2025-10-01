import { Value } from './Value';
import { V } from './V';

export class Vec3 {
  x: Value;
  y: Value;
  z: Value;

  constructor(x: Value | number, y: Value | number, z: Value | number) {
    this.x = typeof x === 'number' ? new Value(x) : x;
    this.y = typeof y === 'number' ? new Value(y) : y;
    this.z = typeof z === 'number' ? new Value(z) : z;
  }

  static W(x: number, y: number, z: number): Vec3 {
    return new Vec3(V.W(x), V.W(y), V.W(z));
  }

  static C(x: number, y: number, z: number): Vec3 {
    return new Vec3(V.C(x), V.C(y), V.C(z));
  }

  static zero(): Vec3 {
    return new Vec3(V.C(0), V.C(0), V.C(0));
  }

  static one(): Vec3 {
    return new Vec3(V.C(1), V.C(1), V.C(1));
  }

  get magnitude(): Value {
    return V.sqrt(V.add(V.add(V.square(this.x), V.square(this.y)), V.square(this.z)));
  }

  get sqrMagnitude(): Value {
    return V.add(V.add(V.square(this.x), V.square(this.y)), V.square(this.z));
  }

  get normalized(): Vec3 {
    const mag = this.magnitude;
    return new Vec3(V.div(this.x, mag), V.div(this.y, mag), V.div(this.z, mag));
  }

  static dot(a: Vec3, b: Vec3): Value {
    return V.add(V.add(V.mul(a.x, b.x), V.mul(a.y, b.y)), V.mul(a.z, b.z));
  }

  static cross(a: Vec3, b: Vec3): Vec3 {
    return new Vec3(
      V.sub(V.mul(a.y, b.z), V.mul(a.z, b.y)),
      V.sub(V.mul(a.z, b.x), V.mul(a.x, b.z)),
      V.sub(V.mul(a.x, b.y), V.mul(a.y, b.x))
    );
  }

  add(other: Vec3): Vec3 {
    return new Vec3(V.add(this.x, other.x), V.add(this.y, other.y), V.add(this.z, other.z));
  }

  sub(other: Vec3): Vec3 {
    return new Vec3(V.sub(this.x, other.x), V.sub(this.y, other.y), V.sub(this.z, other.z));
  }

  mul(scalar: Value | number): Vec3 {
    return new Vec3(V.mul(this.x, scalar), V.mul(this.y, scalar), V.mul(this.z, scalar));
  }

  div(scalar: Value | number): Vec3 {
    return new Vec3(V.div(this.x, scalar), V.div(this.y, scalar), V.div(this.z, scalar));
  }

  get trainables(): Value[] {
    return [this.x, this.y, this.z];
  }

  toString(): string {
    return `Vec3(${this.x.data}, ${this.y.data}, ${this.z.data})`;
  }
}
