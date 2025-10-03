import { Vec3 } from 'scalar-autograd';
import { TriangleMesh, Triangle } from './TriangleMesh';

export class IcoSphere {
  /**
   * Generate an icosphere with the given number of subdivisions.
   * @param subdivisions - Number of subdivision levels (0-4 recommended)
   * @param radius - Radius of the sphere
   * @returns Generated triangle mesh
   */
  static generate(subdivisions: number, radius: number = 1.0): TriangleMesh {
    const mesh = this.createIcosahedron();

    for (let i = 0; i < subdivisions; i++) {
      this.subdivide(mesh);
    }

    this.projectToSphere(mesh, radius);

    return mesh;
  }

  private static createIcosahedron(): TriangleMesh {
    const t = (1.0 + Math.sqrt(5.0)) / 2.0; // Golden ratio

    const vertices = [
      Vec3.C(-1, t, 0),
      Vec3.C(1, t, 0),
      Vec3.C(-1, -t, 0),
      Vec3.C(1, -t, 0),
      Vec3.C(0, -1, t),
      Vec3.C(0, 1, t),
      Vec3.C(0, -1, -t),
      Vec3.C(0, 1, -t),
      Vec3.C(t, 0, -1),
      Vec3.C(t, 0, 1),
      Vec3.C(-t, 0, -1),
      Vec3.C(-t, 0, 1),
    ];

    const faces: Triangle[] = [
      // 5 faces around point 0
      { vertices: [0, 11, 5] },
      { vertices: [0, 5, 1] },
      { vertices: [0, 1, 7] },
      { vertices: [0, 7, 10] },
      { vertices: [0, 10, 11] },
      // 5 adjacent faces
      { vertices: [1, 5, 9] },
      { vertices: [5, 11, 4] },
      { vertices: [11, 10, 2] },
      { vertices: [10, 7, 6] },
      { vertices: [7, 1, 8] },
      // 5 faces around point 3
      { vertices: [3, 9, 4] },
      { vertices: [3, 4, 2] },
      { vertices: [3, 2, 6] },
      { vertices: [3, 6, 8] },
      { vertices: [3, 8, 9] },
      // 5 adjacent faces
      { vertices: [4, 9, 5] },
      { vertices: [2, 4, 11] },
      { vertices: [6, 2, 10] },
      { vertices: [8, 6, 7] },
      { vertices: [9, 8, 1] },
    ];

    return new TriangleMesh(vertices, faces);
  }

  private static subdivide(mesh: TriangleMesh): void {
    const newFaces: Triangle[] = [];
    const midpointCache = new Map<string, number>();

    const getMidpoint = (v1: number, v2: number): number => {
      const key = v1 < v2 ? `${v1},${v2}` : `${v2},${v1}`;

      if (midpointCache.has(key)) {
        return midpointCache.get(key)!;
      }

      const pos1 = mesh.vertices[v1];
      const pos2 = mesh.vertices[v2];
      const mid = Vec3.lerp(pos1, pos2, 0.5);

      const newIdx = mesh.vertices.length;
      mesh.vertices.push(mid);
      midpointCache.set(key, newIdx);

      return newIdx;
    };

    for (const face of mesh.faces) {
      const [v1, v2, v3] = face.vertices;

      const a = getMidpoint(v1, v2);
      const b = getMidpoint(v2, v3);
      const c = getMidpoint(v3, v1);

      // Create 4 new triangles
      newFaces.push({ vertices: [v1, a, c] });
      newFaces.push({ vertices: [v2, b, a] });
      newFaces.push({ vertices: [v3, c, b] });
      newFaces.push({ vertices: [a, b, c] });
    }

    mesh.faces = newFaces;
    mesh.invalidateCache();
  }

  private static projectToSphere(mesh: TriangleMesh, radius: number): void {
    for (let i = 0; i < mesh.vertices.length; i++) {
      const v = mesh.vertices[i];
      const normalized = v.normalized;
      mesh.vertices[i] = normalized.mul(radius);
    }
    mesh.invalidateCache();
  }
}
