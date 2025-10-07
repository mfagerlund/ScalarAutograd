import { Vec3, Geometry, Value } from 'scalar-autograd';

export interface Triangle {
  vertices: [number, number, number]; // Vertex indices
}

export class TriangleMesh {
  vertices: Vec3[];
  faces: Triangle[];

  // Cached properties
  private _faceNormals: Map<number, Vec3> = new Map();
  private _faceAreas: Map<number, Value> = new Map();
  private _vertexStars: Map<number, number[]> = new Map();
  private _cacheValid: boolean = false;

  constructor(vertices: Vec3[], faces: Triangle[]) {
    this.vertices = vertices;
    this.faces = faces;
    this.buildCache();
  }

  private buildCache(): void {
    this._vertexStars.clear();

    // Build vertex stars (incident faces, spatially sorted)
    for (let faceIdx = 0; faceIdx < this.faces.length; faceIdx++) {
      const face = this.faces[faceIdx];
      for (const vertexIdx of face.vertices) {
        if (!this._vertexStars.has(vertexIdx)) {
          this._vertexStars.set(vertexIdx, []);
        }
        this._vertexStars.get(vertexIdx)!.push(faceIdx);
      }
    }

    // Sort all vertex stars spatially (ordered around vertex)
    for (const [vertexIdx, star] of this._vertexStars) {
      this._vertexStars.set(vertexIdx, this.sortVertexStar(vertexIdx, star));
    }

    this._cacheValid = true;
  }

  invalidateCache(): void {
    this._faceNormals.clear();
    this._faceAreas.clear();
    this._cacheValid = false;
  }

  getFaceNormal(faceIdx: number): Vec3 {
    // NEVER cache computation graphs! Always recompute for gradient flow
    const face = this.faces[faceIdx];
    const [i, j, k] = face.vertices;
    const normal = Geometry.triangleNormal(this.vertices[i], this.vertices[j], this.vertices[k]);
    return normal;
  }

  getFaceArea(faceIdx: number): Value {
    // NEVER cache computation graphs! Always recompute for gradient flow
    const face = this.faces[faceIdx];
    const [i, j, k] = face.vertices;
    const area = Geometry.triangleArea(this.vertices[i], this.vertices[j], this.vertices[k]);
    return area;
  }

  /**
   * Get vertex star (faces incident to the vertex, spatially sorted around the vertex).
   * Faces are ordered by edge connectivity for contiguous spatial ordering.
   */
  getVertexStar(vertexIdx: number): number[] {
    if (!this._cacheValid) {
      this.buildCache();
    }
    return this._vertexStars.get(vertexIdx) || [];
  }

  /**
   * Sort vertex star faces in spatial order around the vertex.
   * Uses edge connectivity to traverse the star in circular order.
   */
  private sortVertexStar(_vertexIdx: number, star: number[]): number[] {
    if (star.length <= 1) return [...star];

    // Build adjacency: which faces share edges with each face
    const adjacency = new Map<number, number[]>();
    for (const faceIdx of star) {
      adjacency.set(faceIdx, []);
      for (const otherIdx of star) {
        if (faceIdx !== otherIdx && this.facesShareEdge(faceIdx, otherIdx)) {
          adjacency.get(faceIdx)!.push(otherIdx);
        }
      }
    }

    // Start with first face and traverse by adjacency
    const sorted: number[] = [star[0]];
    const visited = new Set<number>([star[0]]);

    while (sorted.length < star.length) {
      const current = sorted[sorted.length - 1];
      const neighbors = adjacency.get(current) || [];

      // Find first unvisited neighbor
      let found = false;
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          sorted.push(neighbor);
          visited.add(neighbor);
          found = true;
          break;
        }
      }

      // If no unvisited neighbor found, the star may not be fully connected
      // (boundary vertex or non-manifold) - just append remaining unsorted
      if (!found) {
        for (const faceIdx of star) {
          if (!visited.has(faceIdx)) {
            sorted.push(faceIdx);
            visited.add(faceIdx);
          }
        }
        break;
      }
    }

    return sorted;
  }

  getInteriorAngle(faceIdx: number, vertexIdx: number): Value {
    const face = this.faces[faceIdx];
    const localIdx = face.vertices.indexOf(vertexIdx);
    if (localIdx === -1) {
      throw new Error(`Vertex ${vertexIdx} not in face ${faceIdx}`);
    }

    const prev = face.vertices[(localIdx + 2) % 3];
    const curr = vertexIdx;
    const next = face.vertices[(localIdx + 1) % 3];

    return Geometry.interiorAngle(this.vertices[prev], this.vertices[curr], this.vertices[next]);
  }

  setVertexPosition(idx: number, pos: Vec3): void {
    if (!isFinite(pos.x.data) || !isFinite(pos.y.data) || !isFinite(pos.z.data)) {
      console.warn(`Attempted to set invalid position at vertex ${idx}: (${pos.x.data}, ${pos.y.data}, ${pos.z.data})`);
      return;
    }
    this.vertices[idx] = pos;
    this.invalidateCache();
  }

  clone(): TriangleMesh {
    const clonedVertices = this.vertices.map((v) => v.clone());
    const clonedFaces = this.faces.map((f) => ({ vertices: [...f.vertices] as [number, number, number] }));
    return new TriangleMesh(clonedVertices, clonedFaces);
  }

  /**
   * Check if two faces share an edge (are adjacent)
   */
  facesShareEdge(faceIdx1: number, faceIdx2: number): boolean {
    const f1 = this.faces[faceIdx1].vertices;
    const f2 = this.faces[faceIdx2].vertices;

    let sharedVertices = 0;
    for (const v1 of f1) {
      if (f2.includes(v1)) {
        sharedVertices++;
      }
    }

    return sharedVertices === 2; // Share exactly 2 vertices = share an edge
  }

  /**
   * Detect degenerate triangles (very small angles or near-zero area)
   */
  detectDegenerateTriangles(minAngleDegrees: number = 5): Array<{faceIdx: number, angles: number[]}> {
    const degenerate: Array<{faceIdx: number, angles: number[]}> = [];
    const minAngleRad = minAngleDegrees * Math.PI / 180;

    for (let faceIdx = 0; faceIdx < this.faces.length; faceIdx++) {
      const face = this.faces[faceIdx];
      const angles = [
        this.getInteriorAngle(faceIdx, face.vertices[0]).data,
        this.getInteriorAngle(faceIdx, face.vertices[1]).data,
        this.getInteriorAngle(faceIdx, face.vertices[2]).data,
      ];

      if (angles.some(a => a < minAngleRad || !isFinite(a))) {
        degenerate.push({ faceIdx, angles: angles.map(a => a * 180 / Math.PI) });
      }
    }

    return degenerate;
  }

  /**
   * Get mesh statistics
   */
  getStats(): {
    vertices: number;
    faces: number;
    minAngle: number;
    maxAngle: number;
    avgAngle: number;
    degenerateCount: number;
  } {
    let minAngle = Infinity;
    let maxAngle = -Infinity;
    let sumAngle = 0;
    let angleCount = 0;

    for (let faceIdx = 0; faceIdx < this.faces.length; faceIdx++) {
      const face = this.faces[faceIdx];
      for (const v of face.vertices) {
        const angle = this.getInteriorAngle(faceIdx, v).data * 180 / Math.PI;
        if (isFinite(angle)) {
          minAngle = Math.min(minAngle, angle);
          maxAngle = Math.max(maxAngle, angle);
          sumAngle += angle;
          angleCount++;
        }
      }
    }

    const degenerate = this.detectDegenerateTriangles();

    return {
      vertices: this.vertices.length,
      faces: this.faces.length,
      minAngle: minAngle === Infinity ? 0 : minAngle,
      maxAngle: maxAngle === -Infinity ? 0 : maxAngle,
      avgAngle: angleCount > 0 ? sumAngle / angleCount : 0,
      degenerateCount: degenerate.length,
    };
  }
}
