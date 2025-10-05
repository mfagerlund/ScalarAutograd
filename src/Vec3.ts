import { V } from './V';
import { Value } from './Value';

/**
 * 3D vector class with differentiable operations.
 * @public
 */
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

  /**
   * Normalized vector with custom analytical gradients.
   * More efficient than autodiff through magnitude and division.
   *
   * Gradient formula: ∂(v/|v|)/∂v = (I - nn^T)/|v| where n = v/|v|
   */
  normalizedCustomGrad(): Vec3 {
    // Forward pass: compute normalized vector using .data (no autodiff graph)
    const mag = Math.sqrt(this.x.data * this.x.data + this.y.data * this.y.data + this.z.data * this.z.data);
    const nx = this.x.data / mag;
    const ny = this.y.data / mag;
    const nz = this.z.data / mag;
    const invMag = 1.0 / mag;

    // Create output Values with custom backward pass
    const outX = Value.makeNary(
      nx,
      [this.x, this.y, this.z],
      (out) => () => {
        // ∂nx/∂vx = (1 - nx*nx)/|v|, ∂nx/∂vy = -nx*ny/|v|, ∂nx/∂vz = -nx*nz/|v|
        this.x.grad += out.grad * (1 - nx * nx) * invMag;
        this.y.grad += out.grad * (-nx * ny) * invMag;
        this.z.grad += out.grad * (-nx * nz) * invMag;
      },
      'normalize_x',
      'normalize_x'
    );

    const outY = Value.makeNary(
      ny,
      [this.x, this.y, this.z],
      (out) => () => {
        // ∂ny/∂vx = -ny*nx/|v|, ∂ny/∂vy = (1 - ny*ny)/|v|, ∂ny/∂vz = -ny*nz/|v|
        this.x.grad += out.grad * (-ny * nx) * invMag;
        this.y.grad += out.grad * (1 - ny * ny) * invMag;
        this.z.grad += out.grad * (-ny * nz) * invMag;
      },
      'normalize_y',
      'normalize_y'
    );

    const outZ = Value.makeNary(
      nz,
      [this.x, this.y, this.z],
      (out) => () => {
        // ∂nz/∂vx = -nz*nx/|v|, ∂nz/∂vy = -nz*ny/|v|, ∂nz/∂vz = (1 - nz*nz)/|v|
        this.x.grad += out.grad * (-nz * nx) * invMag;
        this.y.grad += out.grad * (-nz * ny) * invMag;
        this.z.grad += out.grad * (1 - nz * nz) * invMag;
      },
      'normalize_z',
      'normalize_z'
    );

    return new Vec3(outX, outY, outZ);
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

  /**
   * Compute angle between two vectors in radians.
   * Returns value in range [0, π].
   */
  static angleBetween(a: Vec3, b: Vec3): Value {
    const dotProd = Vec3.dot(a, b);
    const magProduct = V.mul(a.magnitude, b.magnitude);
    const cosAngle = V.div(dotProd, magProduct);
    return V.acos(V.clamp(cosAngle, -1, 1)); // Clamp for numerical stability
  }

  /**
   * Project vector a onto vector b.
   */
  static project(a: Vec3, b: Vec3): Vec3 {
    const bMagSq = b.sqrMagnitude;
    const scale = V.div(Vec3.dot(a, b), bMagSq);
    return b.mul(scale);
  }

  /**
   * Reject vector a from vector b (component of a perpendicular to b).
   */
  static reject(a: Vec3, b: Vec3): Vec3 {
    return a.sub(Vec3.project(a, b));
  }

  /**
   * Linear interpolation between two vectors.
   */
  static lerp(a: Vec3, b: Vec3, t: Value | number): Vec3 {
    const oneMinusT = V.sub(1, t);
    return new Vec3(
      V.add(V.mul(a.x, oneMinusT), V.mul(b.x, t)),
      V.add(V.mul(a.y, oneMinusT), V.mul(b.y, t)),
      V.add(V.mul(a.z, oneMinusT), V.mul(b.z, t))
    );
  }

  /**
   * Component-wise minimum.
   */
  static min(a: Vec3, b: Vec3): Vec3 {
    return new Vec3(V.min(a.x, b.x), V.min(a.y, b.y), V.min(a.z, b.z));
  }

  /**
   * Component-wise maximum.
   */
  static max(a: Vec3, b: Vec3): Vec3 {
    return new Vec3(V.max(a.x, b.x), V.max(a.y, b.y), V.max(a.z, b.z));
  }

  /**
   * Clone this vector (create new Vec3 with same values but independent graph).
   */
  clone(): Vec3 {
    return new Vec3(new Value(this.x.data), new Value(this.y.data), new Value(this.z.data));
  }

  /**
   * Create Vec3 from raw data (no gradients).
   */
  static fromData(x: number, y: number, z: number): Vec3 {
    return Vec3.C(x, y, z);
  }

  /**
   * Extract raw data as array [x, y, z].
   */
  toArray(): number[] {
    return [this.x.data, this.y.data, this.z.data];
  }

  /**
   * Distance between two points.
   */
  static distance(a: Vec3, b: Vec3): Value {
    return a.sub(b).magnitude;
  }

  /**
   * Squared distance (faster, no sqrt).
   */
  static sqrDistance(a: Vec3, b: Vec3): Value {
    return a.sub(b).sqrMagnitude;
  }
}
