import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './EnergyRegistry';

/**
 * Stochastic Bimodal Energy with Adaptive Random Edge Splits
 *
 * PRINCIPLE: Find the best edge-based bipartition by exploring random splits:
 * 1. Use ONE random split per vertex (cached)
 * 2. Every N iterations, try a NEW random split
 * 3. If the new split has lower energy, keep it; otherwise revert
 * 4. Over time, converges to locally optimal splits
 *
 * Alignment metric (more principled than variance):
 * - Perfect hinge: dot(normal, group_mean) = 1 for all normals in group
 * - Energy contribution: sum of (1 - dot) = 0 when perfectly aligned
 */
export class StochasticBimodalEnergy {
  static readonly name = 'Bimodal Alignment (Random Edge Split)';
  static readonly description = 'Custom: adaptive random split, alignment';
  static readonly supportsCompilation = false; // Uses randomness and iteration counters

  private static currentSplits: Map<number, [number, number]> = new Map();
  private static iterationCount = 0;
  private static resampleInterval = 10; // Try new split every N iterations
  private static rngSeed = 12345;

  /**
   * Simple seeded RNG for deterministic random splits
   */
  private static rng = (() => {
    let seed = StochasticBimodalEnergy.rngSeed;
    return () => {
      seed = (seed * 1664525 + 1013904223) | 0;
      return ((seed >>> 0) / 4294967296);
    };
  })();

  /**
   * Reset the energy state (call when starting new optimization)
   */
  static reset(): void {
    this.currentSplits.clear();
    this.iterationCount = 0;
    this.rng = (() => {
      let seed = this.rngSeed;
      return () => {
        seed = (seed * 1664525 + 1013904223) | 0;
        return ((seed >>> 0) / 4294967296);
      };
    })();
  }

  static compute(mesh: TriangleMesh): Value {
    const vertexEnergies: Value[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      vertexEnergies.push(this.computeVertexEnergy(i, mesh));
    }

    this.iterationCount++;
    return V.sum(vertexEnergies);
  }

  static computeResiduals(mesh: TriangleMesh): Value[] {
    const residuals: Value[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      residuals.push(this.computeVertexEnergy(i, mesh));
    }

    this.iterationCount++;
    return residuals;
  }

  /**
   * Compute bimodal alignment energy for a vertex.
   * Uses cached split, or explores new split every N iterations.
   */
  static computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);
    if (star.length === 3) return V.C(0); // Skip valence-3 (triple points per paper)

    // Get all normals
    const normals: Vec3[] = [];
    for (const faceIdx of star) {
      normals.push(mesh.getFaceNormal(faceIdx).normalized);
    }

    // Initialize split if not exists
    if (!this.currentSplits.has(vertexIdx)) {
      this.currentSplits.set(vertexIdx, this.randomEdgeSplit(star.length));
    }

    const currentSplit = this.currentSplits.get(vertexIdx)!;
    const currentEnergy = this.computePartitionAlignment(currentSplit, normals);

    // Every N iterations, try a new random split
    if (this.iterationCount % this.resampleInterval === 0) {
      const newSplit = this.randomEdgeSplit(star.length);
      const newEnergy = this.computePartitionAlignment(newSplit, normals);

      // Keep the better split
      if (newEnergy.data < currentEnergy.data) {
        this.currentSplits.set(vertexIdx, newSplit);
        return newEnergy;
      }
    }

    return currentEnergy;
  }

  /**
   * Generate random edge-based split.
   * Returns [start, end] indices creating two segments:
   * - Segment 1: [start, end)
   * - Segment 2: [end, start) (wrapping around)
   *
   * Ensures both segments have at least 1 face.
   */
  private static randomEdgeSplit(starSize: number): [number, number] {
    if (starSize < 2) return [0, 1];

    // Pick random start edge (0 to starSize-1)
    const start = Math.floor(this.rng() * starSize);

    // Pick random length for segment 1 (1 to starSize-1)
    const minLength = 1;
    const maxLength = starSize - 1;
    const length = minLength + Math.floor(this.rng() * (maxLength - minLength + 1));

    const end = (start + length) % starSize;

    return [start, end];
  }

  /**
   * Compute alignment energy for a partition.
   */
  private static computePartitionAlignment(
    partition: [number, number],
    normals: Vec3[]
  ): Value {
    const [start, end] = partition;
    const n = normals.length;

    // Build segment indices
    const segment1: number[] = [];
    const segment2: number[] = [];

    if (start < end) {
      for (let i = start; i < end; i++) segment1.push(i);
      for (let i = end; i < n; i++) segment2.push(i);
      for (let i = 0; i < start; i++) segment2.push(i);
    } else {
      for (let i = start; i < n; i++) segment1.push(i);
      for (let i = 0; i < end; i++) segment1.push(i);
      for (let i = end; i < start; i++) segment2.push(i);
    }

    const align1 = this.computeSegmentAlignment(segment1, normals);
    const align2 = this.computeSegmentAlignment(segment2, normals);

    return V.add(align1, align2);
  }

  /**
   * Compute alignment energy for a segment.
   */
  private static computeSegmentAlignment(indices: number[], normals: Vec3[]): Value {
    if (indices.length === 0) return V.C(0);

    const segmentNormals = indices.map(i => normals[i]);
    const n = segmentNormals.length;

    if (n < 1) return V.C(0);

    // Compute mean normal
    let meanNormal = Vec3.zero();
    for (const normal of segmentNormals) {
      meanNormal = meanNormal.add(normal);
    }
    meanNormal = meanNormal.div(n).normalized;

    // Compute alignment: sum of (1 - dot(n_i, mean))
    const misalignments: Value[] = [];
    for (const normal of segmentNormals) {
      const dotProd = Vec3.dot(normal, meanNormal);
      const clampedDot = V.clamp(dotProd, -1, 1);
      misalignments.push(V.sub(V.C(1), clampedDot));
    }

    return V.div(V.sum(misalignments), n);
  }

  static classifyVertices(
    mesh: TriangleMesh,
    hingeThreshold: number = 0.1
  ): { hingeVertices: number[]; seamVertices: number[] } {
    const hingeVertices: number[] = [];
    const seamVertices: number[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      const energy = this.computeVertexEnergy(i, mesh).data;
      if (energy < hingeThreshold) {
        hingeVertices.push(i);
      } else {
        seamVertices.push(i);
      }
    }

    return { hingeVertices, seamVertices };
  }
}

// Register with energy registry
EnergyRegistry.register(StochasticBimodalEnergy);
