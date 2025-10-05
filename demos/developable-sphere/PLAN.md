# Developable Sphere Demo - Implementation Plan

## Overview

This demo implements the **Developability of Triangle Meshes** algorithm from Stein, Grinspun, and Crane (SIGGRAPH 2018). Starting with a sphere (which cannot be flattened without distortion), we use gradient descent to evolve it toward a **piecewise developable surface** - one that can be cut into flat pieces and manufactured from sheet material.

**Key Insight**: By minimizing a local "developability energy" at each vertex, the mesh naturally evolves to concentrate all curvature onto a sparse collection of seam curves, leaving the rest of the surface perfectly flattenable.

---

## Mathematical Foundation

### What is a Developable Surface?

A **developable surface** is one that can be flattened into a plane without stretching or shearing. Think of a cylinder, cone, or a bent piece of paper. Mathematically, a smooth developable surface has:
- **Zero Gaussian curvature** (K = 0) everywhere
- **Straight ruling lines** passing through each point
- C² smoothness (twice differentiable)

For triangle meshes, we need a discrete version that captures these properties.

### Discrete Developability (Definition 3.1 from paper)

A vertex star St(i) is called a **hinge** if its triangles can be partitioned into two edge-connected flat regions with constant normals. A mesh is **discrete developable** if every vertex is a hinge.

**Visual**: Imagine two flat pieces of paper meeting at a fold. The triangles on each side of the fold are coplanar.

```
     Hinge vertex               Seam vertex (not developable)

        N₁                              N₁
       /|\                             /|\
      / | \                           / | \
     /  |  \                         /  |  \
    ----i----  N₂                   ----i----
        |                               |
    (all triangles                  (normals not
     coplanar on                     coplanar - has
     each side)                      Gaussian curvature)
```

**Theorem 3.3**: A vertex star is a hinge **if and only if** all its triangle normals lie in a common plane.

This is the key result we'll exploit for optimization!

### Combinatorial Energy (E^P)

For a vertex i, partition its triangles into two regions F₁, F₂. The "hinge-likeness" of partition P is:

```
π(P) = Σ_{p=1,2} (1/n_p²) Σ_{σ₁,σ₂ ∈ Fₚ} |N_{σ₁} - N_{σ₂}|²
```

This measures how flat each region is. The local energy is:

```
E^P_i = min_{P ∈ Pᵢ} π(P)
```

where Pᵢ is the set of all edge-connected bipartitions of St(i).

**Intuition**: Try all ways to split the vertex star into two parts. Pick the split that makes each part most planar.

**Total Energy**: E^P = Σᵢ E^P_i

---

## Algorithm Overview

1. **Initialize**: Create an icosphere (subdivided icosahedron) with ~500-2000 vertices
2. **Energy Computation**: For each vertex, find the best bipartition and compute π(P)
3. **Gradient Descent**: Use L-BFGS to minimize E^P by adjusting vertex positions
4. **Convergence**: Stop when gradient norm is small or energy plateaus
5. **Visualization**: Render the evolving mesh with seam curves highlighted

**Key Property**: The algorithm naturally creates piecewise developable surfaces without explicitly detecting patches or seams. Seam curves emerge automatically where the energy concentrates.

---

## Extensions to Core Library

Before implementing the demo, we'll extend `Vec3` and create a `Geometry` utility class in the main ScalarAutograd library.

### Vec3 Extensions

**File**: `src/Vec3.ts` (main library)

Add the following methods to support mesh operations:

```typescript
export class Vec3 {
  // ... existing methods ...

  /**
   * Compute angle between two vectors in radians.
   * Returns value in range [0, π].
   */
  static angleBetween(a: Vec3, b: Vec3): Value {
    const dotProd = Vec3.dot(a, b);
    const magProduct = V.mul(a.magnitude, b.magnitude);
    const cosAngle = V.div(dotProd, magProduct);
    return V.acos(V.clamp(cosAngle, -1, 1));  // Clamp for numerical stability
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
    return new Vec3(
      V.min(a.x, b.x),
      V.min(a.y, b.y),
      V.min(a.z, b.z)
    );
  }

  /**
   * Component-wise maximum.
   */
  static max(a: Vec3, b: Vec3): Vec3 {
    return new Vec3(
      V.max(a.x, b.x),
      V.max(a.y, b.y),
      V.max(a.z, b.z)
    );
  }

  /**
   * Clone this vector (create new Vec3 with same values but independent graph).
   */
  clone(): Vec3 {
    return new Vec3(
      new Value(this.x.data),
      new Value(this.y.data),
      new Value(this.z.data)
    );
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
```

### Geometry Utility Class

**File**: `src/Geometry.ts` (main library)

Create a new utility class for geometric computations specific to triangle meshes:

```typescript
import { Vec3 } from './Vec3';
import { V } from './V';
import { Value } from './Value';

/**
 * Geometric utility functions for triangle mesh operations.
 * All methods are differentiable and support automatic gradient computation.
 * @public
 */
export class Geometry {
  /**
   * Compute the normal of a triangle defined by three vertices.
   * Uses right-hand rule: normal points in direction of (b-a) × (c-a).
   * Returns a UNIT normal vector.
   *
   * @param a - First vertex
   * @param b - Second vertex
   * @param c - Third vertex
   * @returns Unit normal vector
   */
  static triangleNormal(a: Vec3, b: Vec3, c: Vec3): Vec3 {
    const edge1 = b.sub(a);
    const edge2 = c.sub(a);
    const cross = Vec3.cross(edge1, edge2);
    return cross.normalized;
  }

  /**
   * Compute the area of a triangle defined by three vertices.
   * Uses the cross product formula: Area = 0.5 * ||(b-a) × (c-a)||
   *
   * @param a - First vertex
   * @param b - Second vertex
   * @param c - Third vertex
   * @returns Triangle area
   */
  static triangleArea(a: Vec3, b: Vec3, c: Vec3): Value {
    const edge1 = b.sub(a);
    const edge2 = c.sub(a);
    const cross = Vec3.cross(edge1, edge2);
    return V.mul(0.5, cross.magnitude);
  }

  /**
   * Compute the interior angle at vertex b in triangle abc.
   * Returns angle in radians, range [0, π].
   *
   * Formula: angle = acos(dot(normalize(a-b), normalize(c-b)))
   *
   * @param a - First neighboring vertex
   * @param b - Vertex where angle is measured
   * @param c - Second neighboring vertex
   * @returns Interior angle in radians
   */
  static interiorAngle(a: Vec3, b: Vec3, c: Vec3): Value {
    const ba = a.sub(b).normalized;
    const bc = c.sub(b).normalized;
    return Vec3.angleBetween(ba, bc);
  }

  /**
   * Compute the angle defect at a vertex.
   * Angle defect = 2π - sum of interior angles around the vertex.
   * For flat regions, angle defect = 0.
   * For positive Gaussian curvature (sphere-like), angle defect > 0.
   * For negative Gaussian curvature (saddle-like), angle defect < 0.
   *
   * @param center - The vertex at which to compute angle defect
   * @param neighbors - Ordered list of neighboring vertices (CCW or CW)
   * @returns Angle defect in radians
   */
  static angleDefect(center: Vec3, neighbors: Vec3[]): Value {
    let angleSum = V.C(0);

    const n = neighbors.length;
    for (let i = 0; i < n; i++) {
      const prev = neighbors[i];
      const next = neighbors[(i + 1) % n];
      const angle = Geometry.interiorAngle(prev, center, next);
      angleSum = V.add(angleSum, angle);
    }

    return V.sub(2 * Math.PI, angleSum);
  }

  /**
   * Compute the centroid (center of mass) of a set of points.
   *
   * @param points - Array of points
   * @returns Centroid
   */
  static centroid(points: Vec3[]): Vec3 {
    if (points.length === 0) {
      return Vec3.zero();
    }

    let sum = Vec3.zero();
    for (const p of points) {
      sum = sum.add(p);
    }
    return sum.div(points.length);
  }

  /**
   * Compute area-weighted normal at a vertex.
   * This is the sum of face normals weighted by face areas,
   * then normalized. Provides a smooth approximation of the
   * surface normal at a vertex.
   *
   * @param center - The vertex
   * @param triangles - Array of triangles as [Vec3, Vec3, Vec3] where center is one of the vertices
   * @returns Area-weighted unit normal
   */
  static vertexNormal(center: Vec3, triangles: [Vec3, Vec3, Vec3][]): Vec3 {
    let weightedSum = Vec3.zero();

    for (const [a, b, c] of triangles) {
      const normal = Geometry.triangleNormal(a, b, c);
      const area = Geometry.triangleArea(a, b, c);
      weightedSum = weightedSum.add(normal.mul(area));
    }

    return weightedSum.normalized;
  }

  /**
   * Project point onto a plane defined by a point and normal.
   *
   * @param point - Point to project
   * @param planePoint - A point on the plane
   * @param planeNormal - Unit normal of the plane
   * @returns Projected point
   */
  static projectToPlane(point: Vec3, planePoint: Vec3, planeNormal: Vec3): Vec3 {
    const toPoint = point.sub(planePoint);
    const distance = Vec3.dot(toPoint, planeNormal);
    const offset = planeNormal.mul(distance);
    return point.sub(offset);
  }

  /**
   * Compute signed distance from point to plane.
   * Positive if point is on the side the normal points to.
   *
   * @param point - Point to measure
   * @param planePoint - A point on the plane
   * @param planeNormal - Unit normal of the plane
   * @returns Signed distance
   */
  static distanceToPlane(point: Vec3, planePoint: Vec3, planeNormal: Vec3): Value {
    const toPoint = point.sub(planePoint);
    return Vec3.dot(toPoint, planeNormal);
  }

  /**
   * Check if a set of normals are coplanar (lie in a common plane).
   * Returns a measure of non-coplanarity (0 = perfectly coplanar).
   * Uses the smallest eigenvalue of the normal covariance matrix.
   *
   * This is a simplified version - the full implementation would
   * compute eigenvalues. For now, returns sum of cross products.
   *
   * @param normals - Array of unit normal vectors
   * @returns Measure of non-coplanarity (0 = coplanar)
   */
  static normalCoplanarity(normals: Vec3[]): Value {
    if (normals.length < 2) return V.C(0);

    // Simple measure: sum of squared cross products between all pairs
    // If all normals are coplanar, all cross products lie along the same axis
    let deviation = V.C(0);

    for (let i = 0; i < normals.length; i++) {
      for (let j = i + 1; j < normals.length; j++) {
        const cross = Vec3.cross(normals[i], normals[j]);
        deviation = V.add(deviation, cross.sqrMagnitude);
      }
    }

    return deviation;
  }

  /**
   * Compute the dihedral angle between two triangles sharing an edge.
   * Returns angle in radians, range [0, π].
   * Angle = 0 means triangles are coplanar.
   * Angle = π means triangles fold completely back on themselves.
   *
   * @param n1 - Normal of first triangle
   * @param n2 - Normal of second triangle
   * @returns Dihedral angle in radians
   */
  static dihedralAngle(n1: Vec3, n2: Vec3): Value {
    const dotProd = Vec3.dot(n1, n2);
    const clampedDot = V.clamp(dotProd, -1, 1);
    return V.acos(clampedDot);
  }

  /**
   * Compute the average of multiple vectors.
   *
   * @param vectors - Array of vectors
   * @returns Average vector (not normalized)
   */
  static average(vectors: Vec3[]): Vec3 {
    if (vectors.length === 0) return Vec3.zero();

    let sum = Vec3.zero();
    for (const v of vectors) {
      sum = sum.add(v);
    }
    return sum.div(vectors.length);
  }

  /**
   * Compute bounding box of a set of points.
   *
   * @param points - Array of points
   * @returns Object with min and max corners
   */
  static boundingBox(points: Vec3[]): { min: Vec3; max: Vec3 } {
    if (points.length === 0) {
      return { min: Vec3.zero(), max: Vec3.zero() };
    }

    let min = points[0].clone();
    let max = points[0].clone();

    for (let i = 1; i < points.length; i++) {
      min = Vec3.min(min, points[i]);
      max = Vec3.max(max, points[i]);
    }

    return { min, max };
  }
}
```

**Export from index.ts**:

Add to `src/index.ts`:
```typescript
export { Geometry } from './Geometry';
```

**Tests**: `test/Geometry.spec.ts`

```typescript
import { describe, it, expect } from 'vitest';
import { Geometry } from '../src/Geometry';
import { Vec3 } from '../src/Vec3';
import { V } from '../src/V';

describe('Geometry', () => {
  it('computes triangle normal correctly', () => {
    const a = Vec3.C(0, 0, 0);
    const b = Vec3.C(1, 0, 0);
    const c = Vec3.C(0, 1, 0);

    const normal = Geometry.triangleNormal(a, b, c);

    expect(normal.x.data).toBeCloseTo(0);
    expect(normal.y.data).toBeCloseTo(0);
    expect(normal.z.data).toBeCloseTo(1);
  });

  it('computes triangle area correctly', () => {
    const a = Vec3.C(0, 0, 0);
    const b = Vec3.C(1, 0, 0);
    const c = Vec3.C(0, 1, 0);

    const area = Geometry.triangleArea(a, b, c);

    expect(area.data).toBeCloseTo(0.5);
  });

  it('computes interior angle correctly', () => {
    const a = Vec3.C(1, 0, 0);
    const b = Vec3.C(0, 0, 0);
    const c = Vec3.C(0, 1, 0);

    const angle = Geometry.interiorAngle(a, b, c);

    expect(angle.data).toBeCloseTo(Math.PI / 2);  // 90 degrees
  });

  it('computes zero angle defect for flat patch', () => {
    const center = Vec3.C(0, 0, 0);
    const neighbors = [
      Vec3.C(1, 0, 0),
      Vec3.C(0.5, 0.866, 0),
      Vec3.C(-0.5, 0.866, 0),
      Vec3.C(-1, 0, 0),
      Vec3.C(-0.5, -0.866, 0),
      Vec3.C(0.5, -0.866, 0),
    ];

    const defect = Geometry.angleDefect(center, neighbors);

    expect(defect.data).toBeCloseTo(0, 3);
  });

  it('computes dihedral angle correctly', () => {
    const n1 = Vec3.C(0, 0, 1).normalized;
    const n2 = Vec3.C(1, 0, 0).normalized;

    const angle = Geometry.dihedralAngle(n1, n2);

    expect(angle.data).toBeCloseTo(Math.PI / 2);
  });
});
```

---

## Technical Specification

### Phase 1: Triangle Mesh Infrastructure

#### 1.1 TriangleMesh Class

**File**: `src/mesh/TriangleMesh.ts`

```typescript
interface Triangle {
  vertices: [number, number, number];  // Vertex indices
}

class TriangleMesh {
  vertices: Vec3[];        // Vertex positions
  faces: Triangle[];       // Triangle connectivity

  // Computed properties (cached)
  private faceNormals: Vec3[];
  private faceAreas: Value[];
  private vertexStars: Map<number, number[]>;  // vertex -> incident faces

  // Core methods (delegate to Geometry utilities)
  getFaceNormal(faceIdx: number): Vec3 {
    // Uses Geometry.triangleNormal()
  }

  getFaceArea(faceIdx: number): Value {
    // Uses Geometry.triangleArea()
  }

  getVertexStar(vertexIdx: number): number[] {
    // Returns face indices
  }

  getInteriorAngle(faceIdx: number, vertexIdx: number): Value {
    // Uses Geometry.interiorAngle()
  }

  // Update geometry
  setVertexPosition(idx: number, pos: Vec3): void;
  invalidateCache(): void;
}
```

**Implementation Note**:
All geometric computations delegate to the `Geometry` utility class:
- Face normal: `Geometry.triangleNormal(v_i, v_j, v_k)`
- Face area: `Geometry.triangleArea(v_i, v_j, v_k)`
- Interior angle: `Geometry.interiorAngle(v_j, v_i, v_k)` where i is the vertex

**Test Strategy**:
- Create a simple tetrahedron, verify normals point outward
- Check that face areas sum to expected value
- Verify vertex stars contain correct faces

#### 1.2 IcoSphere Generator

**File**: `src/mesh/IcoSphere.ts`

**Algorithm**:
1. Start with regular icosahedron (12 vertices, 20 faces)
2. Subdivide each triangle into 4 by adding midpoint vertices
3. Project new vertices onto unit sphere
4. Repeat subdivision to desired resolution

**Subdivision levels**:
- Level 0: 12 vertices, 20 faces
- Level 1: 42 vertices, 80 faces
- Level 2: 162 vertices, 320 faces
- Level 3: 642 vertices, 1280 faces ← **Target**
- Level 4: 2562 vertices, 5120 faces

```typescript
class IcoSphere {
  static generate(subdivisions: number, radius: number = 1.0): TriangleMesh;

  private static icosahedronVertices(): Vec3[];
  private static icosahedronFaces(): Triangle[];
  private static subdivide(mesh: TriangleMesh): TriangleMesh;
  private static projectToSphere(mesh: TriangleMesh, radius: number): void;
}
```

**Test Strategy**:
- Verify level 0 has exactly 12 vertices, 20 faces
- Check that all vertices lie on sphere (distance = radius)
- Verify Euler characteristic: V - E + F = 2 for sphere

---

### Phase 2: Developability Energy

#### 2.1 Partition Enumeration

**File**: `src/energy/PartitionEnumerator.ts`

Given a vertex star with k triangles, enumerate all edge-connected bipartitions.

**Algorithm**:
1. Build adjacency graph of triangles in star (two triangles are adjacent if they share an edge)
2. For each triangle t, start a BFS/DFS to grow region F₁
3. Remaining triangles form F₂
4. Check that both regions are edge-connected
5. Avoid duplicate partitions (P and its complement are the same)

**Complexity**: O(2^k) partitions in worst case, but typically k ≈ 6 and many partitions are invalid.

```typescript
interface Partition {
  region1: number[];  // Face indices
  region2: number[];  // Face indices
}

class PartitionEnumerator {
  static enumerate(star: number[], mesh: TriangleMesh): Partition[];

  private static buildAdjacency(faces: number[], mesh: TriangleMesh): Map<number, number[]>;
  private static isConnected(faces: number[], adjacency: Map<number, number[]>): boolean;
}
```

**Test Strategy**:
- For a vertex with valence 3, expect limited partitions
- Verify all partitions are edge-connected
- Check that region1 ∪ region2 = full star

#### 2.2 Energy Computation

**File**: `src/energy/DevelopableEnergy.ts`

**Combinatorial Energy per Vertex** (Equation 1):

```typescript
function computePartitionEnergy(
  partition: Partition,
  mesh: TriangleMesh
): Value {
  let energy = V.C(0);

  for (const region of [partition.region1, partition.region2]) {
    const n = region.length;

    // Compute average normal
    let avgNormal = Vec3.zero();
    for (const faceIdx of region) {
      avgNormal = avgNormal.add(mesh.getFaceNormal(faceIdx));
    }
    avgNormal = avgNormal.div(n);

    // Sum squared deviations
    for (const faceIdx of region) {
      const N = mesh.getFaceNormal(faceIdx);
      const diff = N.sub(avgNormal);
      energy = V.add(energy, diff.sqrMagnitude);
    }
    energy = V.div(energy, n * n);
  }

  return energy;
}
```

**Vertex Energy**:

```typescript
function computeVertexEnergy(
  vertexIdx: number,
  mesh: TriangleMesh
): Value {
  const star = mesh.getVertexStar(vertexIdx);
  const partitions = PartitionEnumerator.enumerate(star, mesh);

  let minEnergy = V.C(Infinity);

  for (const P of partitions) {
    const energy = computePartitionEnergy(P, mesh);
    minEnergy = V.min(minEnergy, energy);
  }

  return minEnergy;
}
```

**Total Energy**:

```typescript
class DevelopableEnergy {
  static compute(mesh: TriangleMesh): Value {
    let total = V.C(0);

    for (let i = 0; i < mesh.vertices.length; i++) {
      total = V.add(total, computeVertexEnergy(i, mesh));
    }

    return total;
  }
}
```

**Test Strategy**:
- **Flat patch**: Create coplanar triangles, verify energy = 0
- **Single hinge**: Create two flat regions meeting at fold, verify energy ≈ 0
- **Sphere patch**: Curved vertex star, verify energy > 0
- **Gradient check**: Finite difference validation against autodiff

---

### Phase 3: L-BFGS Optimization

#### 3.1 Objective Function

**File**: `src/optimization/DevelopableObjective.ts`

L-BFGS expects:
- `params: Value[]` - flat array of optimizable parameters
- `objectiveFn: (params: Value[]) => Value` - returns scalar cost

We need to convert between mesh representation and flat parameter array:

```typescript
class DevelopableObjective {
  private mesh: TriangleMesh;
  private numVertices: number;

  // Convert mesh to flat parameter array
  meshToParams(): Value[] {
    const params: Value[] = [];
    for (const v of this.mesh.vertices) {
      params.push(v.x, v.y, v.z);
    }
    return params;
  }

  // Update mesh from parameters
  paramsToMesh(params: Value[]): void {
    for (let i = 0; i < this.numVertices; i++) {
      const x = params[3 * i];
      const y = params[3 * i + 1];
      const z = params[3 * i + 2];
      this.mesh.setVertexPosition(i, new Vec3(x, y, z));
    }
  }

  // Objective function for L-BFGS
  objective(params: Value[]): Value {
    this.paramsToMesh(params);
    return DevelopableEnergy.compute(this.mesh);
  }
}
```

#### 3.2 Optimization Loop

**File**: `src/optimization/DevelopableOptimizer.ts`

```typescript
import { lbfgs } from 'scalar-autograd';

class DevelopableOptimizer {
  private mesh: TriangleMesh;
  private history: TriangleMesh[] = [];

  optimize(options: {
    maxIterations?: number;
    gradientTolerance?: number;
    verbose?: boolean;
    captureInterval?: number;  // Save mesh every N iterations
  }): OptimizationResult {
    const objective = new DevelopableObjective(this.mesh);
    const params = objective.meshToParams();

    // Wrap objective to capture snapshots
    let iteration = 0;
    const wrappedObjective = (p: Value[]) => {
      if (iteration % (options.captureInterval ?? 10) === 0) {
        this.captureSnapshot();
      }
      iteration++;
      return objective.objective(p);
    };

    // Run L-BFGS
    const result = lbfgs(params, wrappedObjective, {
      maxIterations: options.maxIterations ?? 200,
      gradientTolerance: options.gradientTolerance ?? 1e-5,
      verbose: options.verbose ?? true,
    });

    // Update mesh with final parameters
    objective.paramsToMesh(params);

    return {
      success: result.success,
      iterations: result.iterations,
      finalEnergy: result.finalCost,
      history: this.history,
    };
  }

  private captureSnapshot(): void {
    // Deep copy mesh for history
    this.history.push(this.mesh.clone());
  }
}
```

**Test Strategy**:
- **Convergence test**: Start with noisy sphere, verify energy decreases monotonically
- **Known solution**: Start with already-developable surface, verify it doesn't change much
- **Gradient norm**: Verify that final gradient norm is below tolerance

---

### Phase 4: Analysis & Metrics

#### 4.1 Developability Analysis

**File**: `src/analysis/DevelopabilityMetrics.ts`

```typescript
interface VertexClassification {
  hingeVertices: number[];    // Energy ≈ 0
  seamVertices: number[];     // Energy > threshold
}

interface DevelopabilityMetrics {
  classification: VertexClassification;
  averageEnergy: number;
  maxEnergy: number;
  seamCurveCount: number;
  developableRatio: number;  // Fraction of hinge vertices
}

class DevelopabilityAnalyzer {
  static analyze(
    mesh: TriangleMesh,
    hingeThreshold: number = 1e-3
  ): DevelopabilityMetrics {
    const classification = this.classifyVertices(mesh, hingeThreshold);
    const seamCurves = this.extractSeamCurves(mesh, classification);

    return {
      classification,
      averageEnergy: this.computeAverageEnergy(mesh),
      maxEnergy: this.computeMaxEnergy(mesh),
      seamCurveCount: seamCurves.length,
      developableRatio: classification.hingeVertices.length / mesh.vertices.length,
    };
  }

  private static classifyVertices(
    mesh: TriangleMesh,
    threshold: number
  ): VertexClassification {
    const hingeVertices: number[] = [];
    const seamVertices: number[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      const energy = computeVertexEnergy(i, mesh).data;
      if (energy < threshold) {
        hingeVertices.push(i);
      } else {
        seamVertices.push(i);
      }
    }

    return { hingeVertices, seamVertices };
  }

  private static extractSeamCurves(
    mesh: TriangleMesh,
    classification: VertexClassification
  ): number[][] {
    // Build graph of seam vertices and extract connected components
    // Each component is a seam curve
    // ... (graph traversal implementation)
  }
}
```

**Test Strategy**:
- Create a cone (developable), verify all vertices classified as hinges
- Create a sphere, verify many seam vertices
- After optimization, verify developableRatio increases significantly

---

### Phase 5: Three.js Visualization

#### 5.1 Core Rendering

**File**: `src/visualization/MeshRenderer.ts`

```typescript
import * as THREE from 'three';

class MeshRenderer {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private meshObject: THREE.Mesh;

  constructor(canvas: HTMLCanvasElement) {
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, canvas.width / canvas.height, 0.1, 1000);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 7.5);
    this.scene.add(ambientLight, directionalLight);

    this.camera.position.z = 3;
  }

  updateMesh(mesh: TriangleMesh, metrics: DevelopabilityMetrics): void {
    // Remove old mesh
    if (this.meshObject) {
      this.scene.remove(this.meshObject);
    }

    // Create THREE.js geometry
    const geometry = new THREE.BufferGeometry();

    // Positions
    const positions = new Float32Array(mesh.vertices.length * 3);
    mesh.vertices.forEach((v, i) => {
      positions[3 * i] = v.x.data;
      positions[3 * i + 1] = v.y.data;
      positions[3 * i + 2] = v.z.data;
    });
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // Indices
    const indices: number[] = [];
    mesh.faces.forEach(f => {
      indices.push(f.vertices[0], f.vertices[1], f.vertices[2]);
    });
    geometry.setIndex(indices);

    // Vertex colors (blue = hinge, red = seam)
    const colors = new Float32Array(mesh.vertices.length * 3);
    for (let i = 0; i < mesh.vertices.length; i++) {
      const isHinge = metrics.classification.hingeVertices.includes(i);
      colors[3 * i] = isHinge ? 0.3 : 1.0;      // R
      colors[3 * i + 1] = isHinge ? 0.5 : 0.2;  // G
      colors[3 * i + 2] = isHinge ? 1.0 : 0.3;  // B
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Material
    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      flatShading: false,
    });

    this.meshObject = new THREE.Mesh(geometry, material);
    this.scene.add(this.meshObject);

    geometry.computeVertexNormals();
  }

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }
}
```

#### 5.2 Animation System

**File**: `src/visualization/AnimationController.ts`

```typescript
class AnimationController {
  private meshHistory: TriangleMesh[];
  private currentFrame: number = 0;
  private playing: boolean = false;
  private renderer: MeshRenderer;

  constructor(history: TriangleMesh[], renderer: MeshRenderer) {
    this.meshHistory = history;
    this.renderer = renderer;
  }

  play(): void {
    this.playing = true;
    this.animate();
  }

  pause(): void {
    this.playing = false;
  }

  seekToFrame(frame: number): void {
    this.currentFrame = Math.max(0, Math.min(frame, this.meshHistory.length - 1));
    this.updateDisplay();
  }

  private animate(): void {
    if (!this.playing) return;

    this.currentFrame = (this.currentFrame + 1) % this.meshHistory.length;
    this.updateDisplay();

    requestAnimationFrame(() => this.animate());
  }

  private updateDisplay(): void {
    const mesh = this.meshHistory[this.currentFrame];
    const metrics = DevelopabilityAnalyzer.analyze(mesh);
    this.renderer.updateMesh(mesh, metrics);
    this.renderer.render();
  }
}
```

#### 5.3 React UI

**File**: `src/App.tsx`

```typescript
import React, { useEffect, useRef, useState } from 'react';
import { IcoSphere } from './mesh/IcoSphere';
import { DevelopableOptimizer } from './optimization/DevelopableOptimizer';
import { MeshRenderer } from './visualization/MeshRenderer';
import { AnimationController } from './visualization/AnimationController';

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [metrics, setMetrics] = useState<DevelopabilityMetrics | null>(null);

  const [renderer, setRenderer] = useState<MeshRenderer | null>(null);
  const [animation, setAnimation] = useState<AnimationController | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    const r = new MeshRenderer(canvasRef.current);
    setRenderer(r);

    // Initial sphere
    const sphere = IcoSphere.generate(3, 1.0);
    const initialMetrics = DevelopabilityAnalyzer.analyze(sphere);
    r.updateMesh(sphere, initialMetrics);
    r.render();
    setMetrics(initialMetrics);
  }, []);

  const handleOptimize = async () => {
    if (!renderer) return;

    setIsOptimizing(true);

    const sphere = IcoSphere.generate(3, 1.0);
    const optimizer = new DevelopableOptimizer(sphere);

    const result = await optimizer.optimize({
      maxIterations: 200,
      gradientTolerance: 1e-5,
      verbose: true,
      captureInterval: 5,
    });

    setIsOptimizing(false);

    const anim = new AnimationController(result.history, renderer);
    setAnimation(anim);

    const finalMetrics = DevelopabilityAnalyzer.analyze(sphere);
    setMetrics(finalMetrics);
  };

  return (
    <div className="app">
      <div className="canvas-container">
        <canvas ref={canvasRef} width={800} height={600} />
      </div>

      <div className="controls">
        <button onClick={handleOptimize} disabled={isOptimizing}>
          {isOptimizing ? 'Optimizing...' : 'Run Optimization'}
        </button>

        {animation && (
          <>
            <button onClick={() => {
              if (isPlaying) animation.pause();
              else animation.play();
              setIsPlaying(!isPlaying);
            }}>
              {isPlaying ? 'Pause' : 'Play'}
            </button>
          </>
        )}
      </div>

      {metrics && (
        <div className="metrics">
          <h3>Developability Metrics</h3>
          <p>Hinge vertices: {metrics.classification.hingeVertices.length}</p>
          <p>Seam vertices: {metrics.classification.seamVertices.length}</p>
          <p>Developable ratio: {(metrics.developableRatio * 100).toFixed(1)}%</p>
          <p>Average energy: {metrics.averageEnergy.toExponential(2)}</p>
          <p>Seam curves: {metrics.seamCurveCount}</p>
        </div>
      )}
    </div>
  );
}
```

---

## Project Structure

```
demos/developable-sphere/
├── src/
│   ├── mesh/
│   │   ├── TriangleMesh.ts         # Core mesh data structure
│   │   ├── IcoSphere.ts            # Sphere generation
│   │   └── MeshUtils.ts            # Helper utilities
│   ├── energy/
│   │   ├── PartitionEnumerator.ts  # Enumerate bipartitions
│   │   └── DevelopableEnergy.ts    # E^P computation
│   ├── optimization/
│   │   ├── DevelopableObjective.ts # L-BFGS objective wrapper
│   │   └── DevelopableOptimizer.ts # Optimization loop
│   ├── analysis/
│   │   └── DevelopabilityMetrics.ts # Hinge/seam classification
│   ├── visualization/
│   │   ├── MeshRenderer.ts         # Three.js rendering
│   │   └── AnimationController.ts  # Evolution playback
│   ├── App.tsx                     # React UI
│   ├── main.tsx                    # Entry point
│   └── styles.css                  # Styling
├── test/
│   ├── TriangleMesh.test.ts
│   ├── IcoSphere.test.ts
│   ├── DevelopableEnergy.test.ts
│   └── DevelopableOptimizer.test.ts
├── public/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
└── PLAN.md (this file)
```

---

## Testing Strategy

### Unit Tests

1. **TriangleMesh.test.ts**
   - Test face normal computation on known geometry
   - Test face area computation
   - Test vertex star extraction
   - Test interior angle computation

2. **IcoSphere.test.ts**
   - Verify vertex count: 10n² + 2 where n = 2^subdivisions
   - Verify all vertices on unit sphere
   - Verify Euler characteristic V - E + F = 2

3. **DevelopableEnergy.test.ts**
   - **Flat patch**: Energy = 0
   - **Hinge**: Two flat regions, energy ≈ 0
   - **Sphere patch**: Energy > 0
   - **Gradient check**: Finite differences vs. autodiff

4. **DevelopableOptimizer.test.ts**
   - Verify energy decreases monotonically
   - Test convergence on noisy developable surface
   - Verify final gradient norm below tolerance

### Integration Tests

5. **Full optimization**
   - Run on sphere, verify significant energy reduction
   - Verify developable ratio increases from ~0% to >80%
   - Verify seam curves are sparse and connected

---

## Expected Results

### Initial Sphere (Subdivision Level 3)
- Vertices: 642
- Faces: 1280
- Average energy: ~0.05 - 0.1 (high)
- Developable ratio: ~0% (all vertices have curvature)

### After Optimization
- Average energy: ~0.001 - 0.01 (low)
- Developable ratio: 80-95% (most vertices are hinges)
- Seam curves: 3-10 connected curves where curvature concentrates
- Visual: Distinct flat "patches" separated by sharp creases

### Performance
- Initial energy computation: ~0.5-2s (depends on partition enumeration)
- L-BFGS iteration: ~1-5s per iteration
- Total optimization: 3-15 minutes for 200 iterations
- Rendering: 60 FPS for smooth interaction

---

## Known Challenges & Mitigations

### 1. Partition Enumeration Complexity
**Problem**: Exponential number of bipartitions for high-valence vertices
**Mitigation**:
- Most vertices have valence 5-6 on icosphere
- Prune invalid (non-edge-connected) partitions early
- Cache partition results where possible

### 2. Gradient Computation
**Problem**: Min operation in E^P_i is non-smooth
**Mitigation**:
- Subgradient is well-defined (gradient of minimal partition)
- L-BFGS handles non-smooth objectives reasonably well
- Paper confirms this approach works in practice

### 3. Local Minima
**Problem**: Non-convex optimization may get stuck
**Mitigation**:
- Paper shows algorithm reliably finds good solutions
- Icosphere tessellation provides reasonable initialization
- Can try multiple random perturbations if needed

### 4. Mesh Degeneracy
**Problem**: Optimization might create degenerate/self-intersecting triangles
**Mitigation**:
- Monitor triangle quality (minimum angle, area)
- Add soft constraints to prevent degeneracy if needed
- Paper doesn't report this as a major issue

### 5. Three.js Performance
**Problem**: Large meshes (2500+ vertices) may render slowly
**Mitigation**:
- Use BufferGeometry (fast)
- Enable frustum culling
- Option to reduce mesh resolution for preview

---

## Extensions & Future Work

1. **Covariance Energy**: Implement E^λ for comparison (faster but different behavior)
2. **Mesh Subdivision**: Implement 4-1 subdivision between optimization phases
3. **Export**: OBJ file export for 3D printing / manufacturing
4. **Flattening**: UV unwrapping of developable patches to verify flattenability
5. **Different Initial Shapes**: Bunny, teapot, arbitrary models
6. **Interactive Editing**: Allow user to guide seam placement
7. **Ruling Lines**: Visualize straight ruling directions

---

## Dependencies

```json
{
  "dependencies": {
    "react": "^19.1.1",
    "react-dom": "^19.1.1",
    "three": "^0.170.0",
    "@types/three": "^0.170.0",
    "scalar-autograd": "workspace:*"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^5.0.3",
    "vite": "^7.1.7",
    "typescript": "~5.8.3",
    "vitest": "^2.1.8"
  }
}
```

---

## Implementation Order

### Sprint 0: Library Extensions (Day 1)
- ⬜ Add Vec3 extensions to `src/Vec3.ts`
- ⬜ Create `src/Geometry.ts` utility class
- ⬜ Add tests for Geometry utilities
- ⬜ Export Geometry from `src/index.ts`
- ⬜ Update API docs

### Sprint 1: Mesh Foundation (Days 2-3)
- ⬜ Create demo project structure
- ⬜ TriangleMesh class (uses Geometry utilities)
- ⬜ IcoSphere generator
- ⬜ Tests for mesh operations

### Sprint 2: Energy Implementation (Days 3-4)
- ✅ Partition enumeration
- ✅ Combinatorial energy E^P
- ✅ Gradient validation tests
- ✅ Energy computation tests

### Sprint 3: Optimization (Days 5-6)
- ✅ L-BFGS objective wrapper
- ✅ Optimization loop with snapshots
- ✅ Convergence tests
- ✅ CLI demo runner

### Sprint 4: Visualization (Days 7-9)
- ✅ Three.js mesh rendering
- ✅ Vertex color-coding (hinge/seam)
- ✅ Animation system
- ✅ React UI integration

### Sprint 5: Polish & Analysis (Days 10-11)
- ✅ Metrics dashboard
- ✅ Performance optimization
- ✅ Documentation
- ✅ Final testing & demo video

---

## Success Criteria

✅ **Mesh Infrastructure**: Icosphere generates valid, manifold meshes
✅ **Energy Computation**: Returns 0 for flat patches, >0 for curved patches
✅ **Optimization**: Sphere energy reduces by 95%+ from initial value
✅ **Developability**: Final mesh has 80%+ hinge vertices
✅ **Visualization**: Smooth 60fps rendering with color-coded vertices
✅ **Animation**: Clear visual evolution from sphere to piecewise developable
✅ **Tests**: All unit tests pass, gradient checks within 1e-4 relative error

---

## References

- **Paper**: Stein, O., Grinspun, E., & Crane, K. (2018). Developability of Triangle Meshes. ACM SIGGRAPH
- **L-BFGS**: `docs/L-BFGS.md` in ScalarAutograd
- **Vec3**: `src/Vec3.ts` in ScalarAutograd (differentiable 3D vectors)

---

**Total Estimated Time**: 10-15 days for full implementation and polish
**Minimum Viable Demo**: 5-7 days (mesh + energy + basic optimization + simple visualization)
