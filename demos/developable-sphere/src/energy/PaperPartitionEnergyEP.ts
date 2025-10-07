import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './utils/EnergyRegistry';

/**
 * Combinatorial Energy (E^P) from "Developability of Triangle Meshes" (Stein et al., SIGGRAPH 2018).
 *
 * This energy directly tests the "two planes" hinge model:
 * - For each vertex, enumerate all edge-connected bipartitions of its star
 * - For each partition, compute variance within each region
 * - Pick the partition with minimum total variance
 *
 * A perfect hinge has E^P = 0 (normals split into two planar groups).
 *
 * Cost: O(k²) partitions × O(k²) computation per partition for valence k.
 * Optimization uses subgradient descent (piecewise differentiable).
 *
 * See PaperPartitionEnergyEPStochastic for 10x faster stochastic variant.
 */
export class PaperPartitionEnergyEP {
  static readonly name: string = 'PaperPartitionEnergyEP';
  static readonly description: string = 'Paper: E^P = min partition variance';
  static readonly supportsCompilation: boolean = false;
  /**
   * Compute total combinatorial energy for the mesh.
   */
  static compute(mesh: TriangleMesh): Value {
    const vertexEnergies: Value[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      vertexEnergies.push(this.computeVertexEnergy(i, mesh));
    }

    return V.sum(vertexEnergies);
  }

  /**
   * Compute per-vertex residuals for compiled optimization.
   */
  static computeResiduals(mesh: TriangleMesh): Value[] {
    const residuals: Value[] = [];

    for (let i = 0; i < mesh.vertices.length; i++) {
      const vertexEnergy = this.computeVertexEnergy(i, mesh);
      residuals.push(vertexEnergy);
    }

    return residuals;
  }

  /**
   * Compute combinatorial energy for a single vertex.
   *
   * Enumerates all edge-connected bipartitions and picks the one
   * with minimum total variance (sum of variances in both regions).
   */
  static computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);
    if (star.length === 3) return V.C(0); // Skip valence-3 (triple points per paper)

    // Get all normals (normalized)
    const normals: Vec3[] = [];
    for (const faceIdx of star) {
      normals.push(mesh.getFaceNormal(faceIdx).normalized);
    }

    // Find the partition with minimum variance
    const { minVariance } = this.findBestPartition(normals);

    return minVariance;
  }

  /**
   * Find the edge-connected bipartition with minimum total variance.
   * Returns the minimum variance and the best partition for gradient computation.
   *
   * For efficiency, we use prefix sums to compute variance in O(1) per partition:
   * Var(R) = Σ|n_i|² - |R| * |mean(R)|²
   *
   * Note: Assumes star faces are radially ordered (guaranteed by icosphere).
   *
   * Protected to allow subclass (PaperPartitionEnergyEPStochastic) to reuse this logic.
   */
  protected static findBestPartition(
    normals: Vec3[]
  ): { minVariance: Value; bestPartition: [number, number] } {
    const k = normals.length;

    // Precompute prefix sums for efficiency
    // S1[i] = sum of normals from 0 to i-1
    // S2[i] = sum of ||n||² from 0 to i-1 (kept as Value for gradient flow)
    const S1: Vec3[] = [Vec3.zero()];
    const S2: Value[] = [V.C(0)];

    for (let i = 0; i < k; i++) {
      S1.push(S1[i].add(normals[i]));
      S2.push(V.add(S2[i], Vec3.dot(normals[i], normals[i])));
    }

    // Try all edge-connected bipartitions
    // Partition is defined by cut indices [p, q): region1 = [p..q), region2 = [q..p)
    let bestVariance: Value | null = null;
    let bestP = 0;
    let bestQ = 1;

    for (let p = 0; p < k; p++) {
      for (let q = p + 1; q <= k; q++) {
        // Region 1: [p, q) (wrap around if needed)
        // Region 2: complement
        const n1 = q - p;
        const n2 = k - n1;

        if (n1 === 0 || n2 === 0) continue; // Must have both regions non-empty

        // Compute variance for region 1: [p, q)
        const sum1 = S1[q].sub(S1[p]);
        const sumSq1 = V.sub(S2[q], S2[p]);
        const mean1 = sum1.div(n1);
        const var1 = V.sub(sumSq1, V.mul(V.C(n1), Vec3.dot(mean1, mean1)));

        // Compute variance for region 2: [q, p) (wrapped)
        const sum2 = S1[k].sub(sum1);
        const sumSq2 = V.sub(S2[k], sumSq1);
        const mean2 = sum2.div(n2);
        const var2 = V.sub(sumSq2, V.mul(V.C(n2), Vec3.dot(mean2, mean2)));

        const totalVar = V.add(var1, var2);

        // Early exit if perfect partition found
        if (totalVar.data < 1e-12) {
          return { minVariance: totalVar, bestPartition: [p, q] };
        }

        if (bestVariance === null || totalVar.data < bestVariance.data) {
          bestVariance = totalVar;
          bestP = p;
          bestQ = q;
        }
      }
    }

    // If no valid partition found (shouldn't happen), return zero
    if (bestVariance === null) {
      return { minVariance: V.C(0), bestPartition: [0, 1] };
    }

    return { minVariance: bestVariance, bestPartition: [bestP, bestQ] };
  }

}
// Register with energy registry
EnergyRegistry.register(PaperPartitionEnergyEP);
