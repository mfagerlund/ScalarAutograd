import { Geometry } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';

/**
 * Objective curvature-based classification using angle defect.
 *
 * This provides ground truth for developability independent of any energy function.
 * Uses discrete Gaussian curvature (angle defect) as the objective measure:
 *
 * - Angle defect = 2π - Σ(interior angles at vertex)
 * - Flat (developable): angle defect ≈ 0
 * - Spherical (positive curvature): angle defect > 0
 * - Saddle (negative curvature): angle defect < 0
 *
 * This is **objectively measurable** and doesn't depend on which energy function
 * was used for optimization.
 */
export class CurvatureClassifier {
  /**
   * Classify vertices by absolute angle defect (curvature).
   *
   * Uses adaptive threshold that scales with mesh resolution.
   * For a sphere, expected angle defect = 4π / num_vertices.
   * We use threshold = 0.1 * (4π / num_vertices) to ensure spheres are correctly
   * classified as curved regardless of subdivision level.
   *
   * For truly flat surfaces, angle defect ≈ 0, which is always below this threshold.
   *
   * @param mesh - The mesh to classify
   * @param thresholdMultiplier - Multiplier for adaptive threshold (default 0.1)
   * @returns Classification with flat and curved vertices
   */
  static classifyVertices(
    mesh: TriangleMesh,
    thresholdMultiplier: number = 0.1
  ): { hingeVertices: number[]; seamVertices: number[] } {
    const hingeVertices: number[] = [];
    const seamVertices: number[] = [];

    // Adaptive threshold based on mesh resolution
    // For a sphere: angle defect per vertex = 4π / num_vertices
    // Threshold = multiplier * expected_sphere_defect
    const expectedSphereDefect = (4 * Math.PI) / mesh.vertices.length;
    const threshold = thresholdMultiplier * expectedSphereDefect;

    for (let i = 0; i < mesh.vertices.length; i++) {
      const curvature = Math.abs(this.computeAngleDefect(i, mesh));

      if (curvature < threshold) {
        // Low curvature = developable (flat)
        hingeVertices.push(i);
      } else {
        // High curvature = curved
        seamVertices.push(i);
      }
    }

    return { hingeVertices, seamVertices };
  }

  /**
   * Compute angle defect (discrete Gaussian curvature) at a vertex.
   */
  private static computeAngleDefect(vertexIdx: number, mesh: TriangleMesh): number {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length === 0) return 0;

    // Get ordered neighbors around the vertex
    const neighbors = this.getOrderedNeighbors(vertexIdx, mesh);
    if (neighbors.length < 3) return 0;

    const center = mesh.vertices[vertexIdx];
    const neighborVerts = neighbors.map(idx => mesh.vertices[idx]);

    const angleDefect = Geometry.angleDefect(center, neighborVerts);
    return angleDefect.data;
  }

  /**
   * Get neighbors ordered spatially around the vertex (CCW or CW).
   * This is needed for angle defect calculation.
   */
  private static getOrderedNeighbors(vertexIdx: number, mesh: TriangleMesh): number[] {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length === 0) return [];

    // Build edge connectivity
    const edgeMap = new Map<number, Set<number>>();

    for (const faceIdx of star) {
      const face = mesh.faces[faceIdx];
      const verts = face.vertices;

      // Find where vertexIdx appears in this face
      const localIdx = verts.indexOf(vertexIdx);
      if (localIdx === -1) continue;

      // Get the two neighbors in this face
      const next = verts[(localIdx + 1) % 3];
      const prev = verts[(localIdx + 2) % 3];

      // Add bidirectional edges
      if (!edgeMap.has(next)) edgeMap.set(next, new Set());
      if (!edgeMap.has(prev)) edgeMap.set(prev, new Set());
      edgeMap.get(next)!.add(prev);
      edgeMap.get(prev)!.add(next);
    }

    // Walk the ring to get ordered neighbors
    const ordered: number[] = [];
    const visited = new Set<number>();
    let current = edgeMap.keys().next().value;
    if (current === undefined) return [];

    ordered.push(current);
    visited.add(current);

    while (ordered.length < edgeMap.size) {
      const neighbors = edgeMap.get(current);
      if (!neighbors) break;

      let found = false;
      for (const next of neighbors) {
        if (!visited.has(next)) {
          ordered.push(next);
          visited.add(next);
          current = next;
          found = true;
          break;
        }
      }

      if (!found) break;
    }

    return ordered;
  }

  /**
   * Count the number of connected developable regions in the mesh.
   * Uses face-based connectivity to count contiguous flat regions.
   */
  static countDevelopableRegions(
    mesh: TriangleMesh,
    thresholdMultiplier: number = 0.1
  ): number {
    const { hingeVertices } = this.classifyVertices(mesh, thresholdMultiplier);
    const hingeSet = new Set(hingeVertices);

    // Build face-to-face adjacency
    const faceAdjacency = new Map<number, number[]>();
    for (let i = 0; i < mesh.faces.length; i++) {
      faceAdjacency.set(i, []);
    }

    // Find adjacent faces (share an edge)
    for (let i = 0; i < mesh.faces.length; i++) {
      for (let j = i + 1; j < mesh.faces.length; j++) {
        if (mesh.facesShareEdge(i, j)) {
          faceAdjacency.get(i)!.push(j);
          faceAdjacency.get(j)!.push(i);
        }
      }
    }

    // Check if a face is developable (all vertices are hinges)
    const isDevelopableFace = (faceIdx: number): boolean => {
      const face = mesh.faces[faceIdx];
      return face.vertices.every(v => hingeSet.has(v));
    };

    // Count connected components of developable faces
    const visited = new Set<number>();
    let regionCount = 0;

    const dfs = (faceIdx: number) => {
      if (visited.has(faceIdx) || !isDevelopableFace(faceIdx)) return;
      visited.add(faceIdx);

      const neighbors = faceAdjacency.get(faceIdx) || [];
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor) && isDevelopableFace(neighbor)) {
          dfs(neighbor);
        }
      }
    };

    for (let i = 0; i < mesh.faces.length; i++) {
      if (!visited.has(i) && isDevelopableFace(i)) {
        regionCount++;
        dfs(i);
      }
    }

    return regionCount;
  }

  /**
   * Get curvature statistics for the mesh.
   */
  static getCurvatureStats(mesh: TriangleMesh): {
    min: number;
    max: number;
    mean: number;
    median: number;
    p10: number;
    p30: number;
    p50: number;
    p70: number;
    p90: number;
  } {
    const curvatures: number[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      curvatures.push(Math.abs(this.computeAngleDefect(i, mesh)));
    }

    curvatures.sort((a, b) => a - b);

    const getPercentile = (p: number) => {
      const index = Math.floor((curvatures.length - 1) * (p / 100));
      return curvatures[index];
    };

    const sum = curvatures.reduce((acc, c) => acc + c, 0);

    return {
      min: curvatures[0],
      max: curvatures[curvatures.length - 1],
      mean: sum / curvatures.length,
      median: getPercentile(50),
      p10: getPercentile(10),
      p30: getPercentile(30),
      p50: getPercentile(50),
      p70: getPercentile(70),
      p90: getPercentile(90),
    };
  }
}
