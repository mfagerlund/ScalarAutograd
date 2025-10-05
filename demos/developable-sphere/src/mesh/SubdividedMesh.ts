import { Vec3 } from 'scalar-autograd';
import { TriangleMesh, Triangle } from './TriangleMesh';

/**
 * Represents the parent of a vertex in a subdivided mesh.
 * - Original vertices have a single parent (themselves)
 * - Edge midpoint vertices have two parents (the edge endpoints)
 */
export type VertexParent =
  | { type: 'original'; vertexIdx: number }
  | { type: 'edge'; v0: number; v1: number };

/**
 * A TriangleMesh with subdivision hierarchy tracking.
 *
 * Supports:
 * - Forward subdivision (4-1 regular subdivision)
 * - Backward projection (extract coarse mesh from fine mesh)
 * - Parent tracking for all vertices
 */
export class SubdividedMesh {
  mesh: TriangleMesh;
  parentMesh?: SubdividedMesh;
  vertexParents: VertexParent[];
  subdivisionLevel: number;

  constructor(
    mesh: TriangleMesh,
    parentMesh?: SubdividedMesh,
    vertexParents?: VertexParent[]
  ) {
    this.mesh = mesh;
    this.parentMesh = parentMesh;
    this.subdivisionLevel = parentMesh ? parentMesh.subdivisionLevel + 1 : 0;

    // If no parent info provided, all vertices are original
    if (!vertexParents) {
      this.vertexParents = mesh.vertices.map((_, i) => ({
        type: 'original' as const,
        vertexIdx: i,
      }));
    } else {
      this.vertexParents = vertexParents;
    }
  }

  /**
   * Create a SubdividedMesh from a regular TriangleMesh (starting point).
   */
  static fromMesh(mesh: TriangleMesh): SubdividedMesh {
    return new SubdividedMesh(mesh);
  }

  /**
   * Perform regular 4-1 subdivision: each triangle splits into 4.
   *
   * Before:           After:
   *     v0                v0
   *    /  \              /  \
   *   /    \           m01--m02
   *  /      \         / \  / \
   * v1------v2      v1--m12--v2
   *
   * Returns a new SubdividedMesh with parent tracking.
   */
  subdivide(): SubdividedMesh {
    const coarseMesh = this.mesh;
    const newVertices: Vec3[] = [...coarseMesh.vertices.map((v) => v.clone())];
    const newFaces: Triangle[] = [];
    const newVertexParents: VertexParent[] = [
      ...this.vertexParents.map((p) => ({ ...p })),
    ];

    // Edge map: stores midpoint vertex index for each edge
    // Key: `${min(v0,v1)},${max(v0,v1)}`
    const edgeMidpoints = new Map<string, number>();

    const getOrCreateMidpoint = (v0: number, v1: number): number => {
      const key = v0 < v1 ? `${v0},${v1}` : `${v1},${v0}`;

      if (edgeMidpoints.has(key)) {
        return edgeMidpoints.get(key)!;
      }

      // Create new midpoint vertex
      const p0 = coarseMesh.vertices[v0];
      const p1 = coarseMesh.vertices[v1];
      const midpoint = p0.add(p1).div(2);

      const newIdx = newVertices.length;
      newVertices.push(midpoint);

      // Track parent: this vertex comes from edge (v0, v1)
      newVertexParents.push({
        type: 'edge',
        v0: v0 < v1 ? v0 : v1,
        v1: v0 < v1 ? v1 : v0,
      });

      edgeMidpoints.set(key, newIdx);
      return newIdx;
    };

    // Subdivide each triangle into 4
    for (const face of coarseMesh.faces) {
      const [v0, v1, v2] = face.vertices;

      // Get or create midpoint vertices
      const m01 = getOrCreateMidpoint(v0, v1);
      const m12 = getOrCreateMidpoint(v1, v2);
      const m02 = getOrCreateMidpoint(v0, v2);

      // Create 4 new triangles
      newFaces.push({ vertices: [v0, m01, m02] }); // Corner at v0
      newFaces.push({ vertices: [v1, m12, m01] }); // Corner at v1
      newFaces.push({ vertices: [v2, m02, m12] }); // Corner at v2
      newFaces.push({ vertices: [m01, m12, m02] }); // Center
    }

    const newMesh = new TriangleMesh(newVertices, newFaces);
    return new SubdividedMesh(newMesh, this, newVertexParents);
  }

  /**
   * Project the current mesh back to its parent (coarse) level.
   *
   * - Original vertices: copy position directly
   * - Edge vertices: averaged from parent edge endpoints
   *
   * Returns a new SubdividedMesh at the parent level with updated positions.
   * Throws if this mesh has no parent.
   */
  projectToParent(): SubdividedMesh {
    if (!this.parentMesh) {
      throw new Error('Cannot project to parent: no parent mesh exists');
    }

    const parentMesh = this.parentMesh.mesh.clone();

    // Update parent vertex positions based on fine mesh
    for (let i = 0; i < parentMesh.vertices.length; i++) {
      // Find this vertex in the fine mesh (it should be an original vertex)
      const fineIdx = this.vertexParents.findIndex(
        (p) => p.type === 'original' && p.vertexIdx === i
      );

      if (fineIdx !== -1) {
        parentMesh.setVertexPosition(i, this.mesh.vertices[fineIdx].clone());
      }
    }

    return new SubdividedMesh(
      parentMesh,
      this.parentMesh.parentMesh,
      this.parentMesh.vertexParents
    );
  }

  /**
   * Project back to a specific subdivision level.
   *
   * @param targetLevel - The subdivision level to project to (0 = base mesh)
   * @returns SubdividedMesh at the target level
   */
  projectToLevel(targetLevel: number): SubdividedMesh {
    if (targetLevel < 0) {
      throw new Error('Target level must be non-negative');
    }

    if (targetLevel > this.subdivisionLevel) {
      throw new Error(
        `Target level ${targetLevel} is higher than current level ${this.subdivisionLevel}`
      );
    }

    if (targetLevel === this.subdivisionLevel) {
      return this;
    }

    // Walk up the parent chain
    let current: SubdividedMesh = this;
    while (current.subdivisionLevel > targetLevel) {
      current = current.projectToParent();
    }

    return current;
  }

  /**
   * Clone this mesh at the current subdivision level.
   */
  clone(): SubdividedMesh {
    return new SubdividedMesh(
      this.mesh.clone(),
      this.parentMesh,
      this.vertexParents.map((p) => ({ ...p }))
    );
  }

  /**
   * Get the base (level 0) mesh.
   */
  getBaseMesh(): SubdividedMesh {
    return this.projectToLevel(0);
  }
}
