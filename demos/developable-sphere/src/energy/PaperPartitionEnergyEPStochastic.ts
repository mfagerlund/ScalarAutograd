import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from './EnergyRegistry';
import { PaperPartitionEnergyEP } from './PaperPartitionEnergyEP';

/**
 * Stochastic variant of E^P (PaperPartitionEnergyEP).
 *
 * **Optimization**: Uses probabilistic partition caching to reduce computational cost:
 * - Adaptive schedule: 100% search rate for first 30 iterations, then 10%
 * - Otherwise reuses cached partition (gradients still flow through variance)
 * - Reduces average cost from O(k³) to O(k³/10) per vertex after warmup
 *
 * **Rationale**: Discrete partition choice breaks differentiability anyway (argmin has zero gradient),
 * so using stale partitions doesn't hurt gradient quality—only affects which local optimum we converge to.
 * Early iterations need fresh searches as geometry changes rapidly; later iterations can cache safely.
 *
 * Inherits partition search logic from PaperPartitionEnergyEP.
 */
export class PaperPartitionEnergyEPStochastic extends PaperPartitionEnergyEP {
  static override readonly name = 'PaperPartitionEnergyEPStochastic';
  static override readonly description = 'Paper: E^P (stochastic, adaptive)';

  // Partition cache: vertex index → [p, q] partition
  private static partitionCache = new Map<number, [number, number]>();

  // Track computation count to estimate iteration number
  private static callCount = 0;
  private static lastVertexCount = 0;

  // Adaptive probability schedule
  private static readonly warmupIterations = 30;
  private static readonly warmupProbability = 1.0;
  private static readonly steadyProbability = 0.1;

  /**
   * Compute combinatorial energy for a single vertex with adaptive stochastic caching.
   */
  static override computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);
    if (star.length === 3) return V.C(0);

    const normals: Vec3[] = [];
    for (const faceIdx of star) {
      normals.push(mesh.getFaceNormal(faceIdx).normalized);
    }

    // Track mesh size changes (indicates new optimization run)
    if (this.lastVertexCount !== mesh.vertices.length) {
      this.partitionCache.clear();
      this.callCount = 0;
      this.lastVertexCount = mesh.vertices.length;
    }

    // Estimate iteration number from call pattern
    this.callCount++;
    const estimatedIteration = this.lastVertexCount > 0
      ? Math.floor(this.callCount / this.lastVertexCount)
      : 0;

    // Adaptive probability: 100% warmup, then 10%
    const updateProbability = estimatedIteration < this.warmupIterations
      ? this.warmupProbability
      : this.steadyProbability;

    // Probabilistic update: re-search or use cached partition
    const shouldUpdate = !this.partitionCache.has(vertexIdx) ||
                        Math.random() < updateProbability;

    let partition: [number, number];
    if (shouldUpdate) {
      // Expensive: search for best partition
      const result = this.findBestPartition(normals);
      partition = result.bestPartition;
      this.partitionCache.set(vertexIdx, partition);
      return result.minVariance;
    } else {
      // Cheap: reuse cached partition
      partition = this.partitionCache.get(vertexIdx)!;
      return this.computePartitionVariance(partition, normals);
    }
  }

  /**
   * Compute variance for a given partition (without searching).
   * Uses same prefix-sum optimization as findBestPartition.
   */
  private static computePartitionVariance(partition: [number, number], normals: Vec3[]): Value {
    const [p, q] = partition;
    const k = normals.length;

    // Build prefix sums
    const S1: Vec3[] = [Vec3.zero()];
    const S2: Value[] = [V.C(0)];

    for (let i = 0; i < k; i++) {
      S1.push(S1[i].add(normals[i]));
      S2.push(V.add(S2[i], Vec3.dot(normals[i], normals[i])));
    }

    // Region sizes
    const n1 = q - p;
    const n2 = k - n1;

    // Variance for region 1: [p, q)
    const sum1 = S1[q].sub(S1[p]);
    const sumSq1 = V.sub(S2[q], S2[p]);
    const mean1 = sum1.div(n1);
    const var1 = V.sub(sumSq1, V.mul(V.C(n1), Vec3.dot(mean1, mean1)));

    // Variance for region 2: [q, p) (wrapped)
    const sum2 = S1[k].sub(sum1);
    const sumSq2 = V.sub(S2[k], sumSq1);
    const mean2 = sum2.div(n2);
    const var2 = V.sub(sumSq2, V.mul(V.C(n2), Vec3.dot(mean2, mean2)));

    return V.add(var1, var2);
  }
}

EnergyRegistry.register(PaperPartitionEnergyEPStochastic);
