import { TriangleMesh } from '../mesh/TriangleMesh';

export interface Partition {
  region1: number[]; // Face indices
  region2: number[]; // Face indices
}

export class PartitionEnumerator {
  /**
   * Enumerate all valid edge-connected bipartitions of a vertex star.
   * @param star - Array of face indices forming the vertex star
   * @param mesh - The triangle mesh
   * @returns Array of valid bipartitions
   */
  static enumerate(star: number[], mesh: TriangleMesh): Partition[] {
    if (star.length < 2) {
      return [];
    }

    // Build adjacency graph of faces in the star
    const adjacency = this.buildAdjacency(star, mesh);

    const partitions: Partition[] = [];
    const visited = new Set<string>();

    // Try all possible subsets for region1 (except empty and full set)
    const numSubsets = 1 << star.length; // 2^n

    for (let mask = 1; mask < numSubsets - 1; mask++) {
      const region1: number[] = [];
      const region2: number[] = [];

      for (let i = 0; i < star.length; i++) {
        if ((mask & (1 << i)) !== 0) {
          region1.push(star[i]);
        } else {
          region2.push(star[i]);
        }
      }

      // Check if both regions are non-empty and edge-connected
      if (
        region1.length > 0 &&
        region2.length > 0 &&
        this.isConnected(region1, adjacency) &&
        this.isConnected(region2, adjacency)
      ) {
        // Create canonical form (smaller region first) to avoid duplicates
        const canonical = this.canonicalForm(region1, region2);
        const key = this.partitionKey(canonical);

        if (!visited.has(key)) {
          visited.add(key);
          partitions.push(canonical);
        }
      }
    }

    return partitions;
  }

  /**
   * Build adjacency graph of faces (faces that share an edge)
   */
  private static buildAdjacency(faces: number[], mesh: TriangleMesh): Map<number, number[]> {
    const adjacency = new Map<number, number[]>();

    for (const faceIdx of faces) {
      adjacency.set(faceIdx, []);
    }

    for (let i = 0; i < faces.length; i++) {
      for (let j = i + 1; j < faces.length; j++) {
        const f1 = faces[i];
        const f2 = faces[j];

        if (mesh.facesShareEdge(f1, f2)) {
          adjacency.get(f1)!.push(f2);
          adjacency.get(f2)!.push(f1);
        }
      }
    }

    return adjacency;
  }

  /**
   * Check if a set of faces is edge-connected using BFS
   */
  private static isConnected(faces: number[], adjacency: Map<number, number[]>): boolean {
    if (faces.length === 0) {
      return false;
    }

    const visited = new Set<number>();
    const queue: number[] = [faces[0]];
    visited.add(faces[0]);

    while (queue.length > 0) {
      const current = queue.shift()!;
      const neighbors = adjacency.get(current) || [];

      for (const neighbor of neighbors) {
        if (faces.includes(neighbor) && !visited.has(neighbor)) {
          visited.add(neighbor);
          queue.push(neighbor);
        }
      }
    }

    return visited.size === faces.length;
  }

  /**
   * Create canonical form with smaller region first
   */
  private static canonicalForm(region1: number[], region2: number[]): Partition {
    if (region1.length <= region2.length) {
      return { region1, region2 };
    } else {
      return { region1: region2, region2: region1 };
    }
  }

  /**
   * Create unique key for partition
   */
  private static partitionKey(partition: Partition): string {
    const r1 = [...partition.region1].sort((a, b) => a - b).join(',');
    const r2 = [...partition.region2].sort((a, b) => a - b).join(',');
    return `${r1}|${r2}`;
  }
}
