import { V } from 'scalar-autograd';
import { Value } from 'scalar-autograd';

/**
 * 4D vector class with differentiable operations.
 * Primarily used for quaternions (w, x, y, z) where w is the scalar part.
 */
export class Vec4 {
  w: Value;
  x: Value;
  y: Value;
  z: Value;

  constructor(
    w: Value | number,
    x: Value | number,
    y: Value | number,
    z: Value | number
  ) {
    this.w = typeof w === 'number' ? new Value(w) : w;
    this.x = typeof x === 'number' ? new Value(x) : x;
    this.y = typeof y === 'number' ? new Value(y) : y;
    this.z = typeof z === 'number' ? new Value(z) : z;
  }

  static W(w: number, x: number, y: number, z: number): Vec4 {
    return new Vec4(V.W(w), V.W(x), V.W(y), V.W(z));
  }

  static C(w: number, x: number, y: number, z: number): Vec4 {
    return new Vec4(V.C(w), V.C(x), V.C(y), V.C(z));
  }

  static zero(): Vec4 {
    return new Vec4(V.C(0), V.C(0), V.C(0), V.C(0));
  }

  static identity(): Vec4 {
    // Identity quaternion: (1, 0, 0, 0)
    return new Vec4(V.C(1), V.C(0), V.C(0), V.C(0));
  }

  get magnitude(): Value {
    return V.sqrt(
      V.add(
        V.add(V.square(this.w), V.square(this.x)),
        V.add(V.square(this.y), V.square(this.z))
      )
    );
  }

  get sqrMagnitude(): Value {
    return V.add(
      V.add(V.square(this.w), V.square(this.x)),
      V.add(V.square(this.y), V.square(this.z))
    );
  }

  get normalized(): Vec4 {
    const mag = this.magnitude;
    return new Vec4(
      V.div(this.w, mag),
      V.div(this.x, mag),
      V.div(this.y, mag),
      V.div(this.z, mag)
    );
  }

  /**
   * Normalized quaternion with custom analytical gradients.
   * More efficient than autodiff through magnitude and division.
   *
   * Gradient formula: ∂(q/|q|)/∂q = (I - nn^T)/|q| where n = q/|q|
   */
  normalizedCustomGrad(): Vec4 {
    // Forward pass: compute normalized quaternion using .data (no autodiff graph)
    const magSq =
      this.w.data * this.w.data +
      this.x.data * this.x.data +
      this.y.data * this.y.data +
      this.z.data * this.z.data;
    const mag = Math.sqrt(magSq);
    const nw = this.w.data / mag;
    const nx = this.x.data / mag;
    const ny = this.y.data / mag;
    const nz = this.z.data / mag;
    const invMag = 1.0 / mag;

    // Create output Values with custom backward pass
    // Gradient: ∂(qi/|q|)/∂qj = δij/|q| - qi*qj/|q|³
    const outW = Value.makeNary(
      nw,
      [this.w, this.x, this.y, this.z],
      (out: Value) => () => {
        this.w.grad += out.grad * (1 - nw * nw) * invMag;
        this.x.grad += out.grad * (-nw * nx) * invMag;
        this.y.grad += out.grad * (-nw * ny) * invMag;
        this.z.grad += out.grad * (-nw * nz) * invMag;
      },
      'normalize_w',
      'normalize_w'
    );

    const outX = Value.makeNary(
      nx,
      [this.w, this.x, this.y, this.z],
      (out: Value) => () => {
        this.w.grad += out.grad * (-nx * nw) * invMag;
        this.x.grad += out.grad * (1 - nx * nx) * invMag;
        this.y.grad += out.grad * (-nx * ny) * invMag;
        this.z.grad += out.grad * (-nx * nz) * invMag;
      },
      'normalize_x',
      'normalize_x'
    );

    const outY = Value.makeNary(
      ny,
      [this.w, this.x, this.y, this.z],
      (out: Value) => () => {
        this.w.grad += out.grad * (-ny * nw) * invMag;
        this.x.grad += out.grad * (-ny * nx) * invMag;
        this.y.grad += out.grad * (1 - ny * ny) * invMag;
        this.z.grad += out.grad * (-ny * nz) * invMag;
      },
      'normalize_y',
      'normalize_y'
    );

    const outZ = Value.makeNary(
      nz,
      [this.w, this.x, this.y, this.z],
      (out: Value) => () => {
        this.w.grad += out.grad * (-nz * nw) * invMag;
        this.x.grad += out.grad * (-nz * nx) * invMag;
        this.y.grad += out.grad * (-nz * ny) * invMag;
        this.z.grad += out.grad * (1 - nz * nz) * invMag;
      },
      'normalize_z',
      'normalize_z'
    );

    return new Vec4(outW, outX, outY, outZ);
  }

  static dot(a: Vec4, b: Vec4): Value {
    return V.add(
      V.add(V.mul(a.w, b.w), V.mul(a.x, b.x)),
      V.add(V.mul(a.y, b.y), V.mul(a.z, b.z))
    );
  }

  add(other: Vec4): Vec4 {
    return new Vec4(
      V.add(this.w, other.w),
      V.add(this.x, other.x),
      V.add(this.y, other.y),
      V.add(this.z, other.z)
    );
  }

  sub(other: Vec4): Vec4 {
    return new Vec4(
      V.sub(this.w, other.w),
      V.sub(this.x, other.x),
      V.sub(this.y, other.y),
      V.sub(this.z, other.z)
    );
  }

  mul(scalar: Value | number): Vec4 {
    return new Vec4(
      V.mul(this.w, scalar),
      V.mul(this.x, scalar),
      V.mul(this.y, scalar),
      V.mul(this.z, scalar)
    );
  }

  div(scalar: Value | number): Vec4 {
    return new Vec4(
      V.div(this.w, scalar),
      V.div(this.x, scalar),
      V.div(this.y, scalar),
      V.div(this.z, scalar)
    );
  }

  get trainables(): Value[] {
    return [this.w, this.x, this.y, this.z];
  }

  toString(): string {
    return `Vec4(${this.w.data}, ${this.x.data}, ${this.y.data}, ${this.z.data})`;
  }

  /**
   * Clone this vector (create new Vec4 with same values but independent graph).
   */
  clone(): Vec4 {
    return new Vec4(
      new Value(this.w.data),
      new Value(this.x.data),
      new Value(this.y.data),
      new Value(this.z.data)
    );
  }

  /**
   * Create Vec4 from raw data (no gradients).
   */
  static fromData(w: number, x: number, y: number, z: number): Vec4 {
    return Vec4.C(w, x, y, z);
  }

  /**
   * Extract raw data as array [w, x, y, z].
   */
  toArray(): [number, number, number, number] {
    return [this.w.data, this.x.data, this.y.data, this.z.data];
  }
}
