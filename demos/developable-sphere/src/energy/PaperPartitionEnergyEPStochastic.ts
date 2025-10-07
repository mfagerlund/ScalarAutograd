import { Value, V, Vec3 } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from '../../../ScalarAutograd/demos/developable-sphere/src/energy/utils/EnergyRegistry';
import { PaperPartitionEnergyEP } from './PaperPartitionEnergyEP';

/**
 * Sampled variant of E^P (PaperPartitionEnergyEP).
 *
 * **Optimization**: Uses deterministic partition sampling to reduce computational cost:
 * - Warmup phase: Full search (O(k²)) every iteration for first 30 iterations
 * - Steady phase: Sample 10% of partition space (O(k²/10)) EVERY iteration
 * - Always includes current cached partition in comparison
 * - Reduces cost while continuously exploring new partitions
 *
 * **Rationale**: Discrete partition choice breaks differentiability anyway (argmin has zero gradient),
 * so using sampled search doesn't hurt gradient quality—only affects which local optimum we converge to.
 * Sampling every iteration spreads cost evenly instead of periodic spikes.
 *
 * **L-BFGS Compatibility**: Uses DETERMINISTIC sampling (vertexIdx + iteration as seed, not Math.random())
 * to keep energy function consistent during line search. Non-deterministic functions break Armijo condition.
 *
 * **Auto-unstick**: Detects energy stagnation (< 0.01% change over 5 mesh evals) and
 * forces full partition search for 3 iterations to escape local minima. This gives
 * L-BFGS fresh descent directions without manual intervention.
 *
 * **Line Search Stability**: Iteration counting uses mesh evaluation completion (vertex wrap to 0)
 * instead of function call count. This prevents line search evaluations from triggering
 * partition updates mid-iteration, which would invalidate gradients and cause failures.
 *
 * **Idle Restart**: Clears cache after 1s idle for fresh warmup on new runs.
 *
 * Test: npx tsx demos/developable-sphere/debug-energy.ts PaperPartitionEnergyEPStochastic 300 3
 *
 * Inherits partition search logic from PaperPartitionEnergyEP.
 */
export class PaperPartitionEnergyEPStochastic extends PaperPartitionEnergyEP {
  static override readonly name = 'PaperPartitionEnergyEPStochastic';
  static override readonly description = 'Paper: E^P (cached, adaptive)';

  // Partition cache: vertex index → [p, q] partition
  private static partitionCache = new Map<number, [number, number]>();

  // Track mesh evaluations to estimate iteration number
  // meshEvaluationCount increments only when vertex idx wraps to 0
  // This prevents line search from triggering partition updates
  private static meshEvaluationCount = 0;
  private static lastVertexCount = 0;
  private static lastVertexIdx = -1;
  private static lastCallTime = 0;

  // Stagnation detection (forces full search when stuck)
  private static recentEnergies: number[] = [];
  private static forceFullSearchUntil = 0;
  private static readonly stagnationWindow = 5;
  private static readonly stagnationThreshold = 1e-4; // 0.01% change
  private static readonly fullSearchDuration = 3; // iterations

  // Deterministic sampling schedule
  private static readonly warmupIterations = 30;
  private static readonly sampleRate = 0.1; // Sample 10% of partition space in steady state
  private static readonly restartIdleTimeMs = 1000; // Restart warmup after 1s idle

  /**
   * Override compute to track total energy for stagnation detection.
   */
  static override compute(mesh: TriangleMesh): Value {
    const totalEnergy = super.compute(mesh);

    // Track energy history for stagnation detection
    this.recentEnergies.push(totalEnergy.data);
    if (this.recentEnergies.length > this.stagnationWindow) {
      this.recentEnergies.shift();
    }

    // Detect stagnation: energy barely changed over last N iterations
    if (this.recentEnergies.length >= this.stagnationWindow) {
      const oldest = this.recentEnergies[0];
      const newest = this.recentEnergies[this.recentEnergies.length - 1];
      const relativeChange = Math.abs((newest - oldest) / oldest);

      const currentIteration = this.meshEvaluationCount;

      // If stagnating and past warmup, force full search to escape local minimum
      // This automatically "unsticks" L-BFGS by providing fresh discrete choices
      if (relativeChange < this.stagnationThreshold &&
          currentIteration >= this.warmupIterations &&
          currentIteration >= this.forceFullSearchUntil) {
        this.forceFullSearchUntil = currentIteration + this.fullSearchDuration;
        this.partitionCache.clear(); // Clear cache to force fresh search
      }
    }

    return totalEnergy;
  }

  /**
   * Compute combinatorial energy for a single vertex with adaptive sampled search.
   */
  static override computeVertexEnergy(vertexIdx: number, mesh: TriangleMesh): Value {
    const star = mesh.getVertexStar(vertexIdx);
    if (star.length < 2) return V.C(0);
    if (star.length === 3) return V.C(0);

    const normals: Vec3[] = [];
    for (const faceIdx of star) {
      normals.push(mesh.getFaceNormal(faceIdx).normalized);
    }

    // Detect mesh size change OR idle restart (1s+ since last call)
    const now = Date.now();
    const meshChanged = this.lastVertexCount !== mesh.vertices.length;
    const idleRestart = now - this.lastCallTime > this.restartIdleTimeMs;

    if (meshChanged || idleRestart) {
      // New optimization run: clear cache and restart warmup
      this.partitionCache.clear();
      this.meshEvaluationCount = 0;
      this.lastVertexCount = mesh.vertices.length;
      this.recentEnergies = [];
      this.forceFullSearchUntil = 0;
    }

    this.lastCallTime = now;

    // Detect mesh evaluation completion (vertex idx wraps to 0)
    // This distinguishes L-BFGS iterations from line search evaluations
    if (vertexIdx === 0 && this.lastVertexIdx !== 0) {
      this.meshEvaluationCount++;
    }
    this.lastVertexIdx = vertexIdx;

    // Use mesh evaluation count as iteration estimate (more stable than call count)
    const estimatedIteration = this.meshEvaluationCount;

    // Warmup or forced search: do full search
    const doFullSearch = !this.partitionCache.has(vertexIdx) ||
      (estimatedIteration < this.warmupIterations) ||
      (estimatedIteration < this.forceFullSearchUntil);

    let partition: [number, number];
    if (doFullSearch) {
      // Expensive: search all partitions
      const result = this.findBestPartition(normals);
      partition = result.bestPartition;
      this.partitionCache.set(vertexIdx, partition);
      return result.minVariance;
    } else {
      // Steady state: sample subset of partition space every iteration
      // This spreads cost evenly (O(k²/10) every time) instead of spikes (O(k²) every 10th)
      const result = this.findBestPartitionSampled(normals, vertexIdx, this.meshEvaluationCount);
      partition = result.bestPartition;
      this.partitionCache.set(vertexIdx, partition);
      return result.minVariance;
    }
  }

  /**
   * Search a deterministic sample of partition space.
   * Samples sampleRate% of all possible partitions, always including current cached partition.
   * Uses (vertexIdx, iteration) as seed for deterministic sampling compatible with L-BFGS.
   */
  private static findBestPartitionSampled(
    normals: Vec3[],
    vertexIdx: number,
    iteration: number
  ): { bestPartition: [number, number]; minVariance: Value } {
    const k = normals.length;
    const totalPartitions = (k * (k - 1)) / 2;
    const sampleSize = Math.max(1, Math.floor(totalPartitions * this.sampleRate));

    // Always include current cached partition
    const currentPartition = this.partitionCache.get(vertexIdx);
    const candidatePartitions: [number, number][] = currentPartition ? [currentPartition] : [];

    // Deterministic sampling using vertex index and iteration as seed
    // Use simple hash to generate starting offset
    const seed = ((vertexIdx * 73856093) ^ (iteration * 19349663)) >>> 0;

    // Sample partitions deterministically
    let samplesNeeded = sampleSize - candidatePartitions.length;
    let attempts = 0;
    const maxAttempts = sampleSize * 3;

    while (samplesNeeded > 0 && attempts < maxAttempts) {
      // Generate partition index deterministically
      const partitionIdx = (seed + attempts * 2654435761) % totalPartitions;

      // Convert linear index to (p, q) partition
      let count = 0;
      let found = false;
      for (let p = 0; p < k && !found; p++) {
        for (let q = p + 2; q <= k && !found; q++) {
          if (count === partitionIdx) {
            // Check if this partition is already in candidates
            const isDuplicate = candidatePartitions.some(
              ([cp, cq]) => cp === p && cq === q
            );
            if (!isDuplicate) {
              candidatePartitions.push([p, q]);
              samplesNeeded--;
            }
            found = true;
          }
          count++;
        }
      }
      attempts++;
    }

    // Evaluate all candidate partitions
    let bestPartition: [number, number] = candidatePartitions[0];
    let minVariance = this.computePartitionVariance(bestPartition, normals);

    for (let i = 1; i < candidatePartitions.length; i++) {
      const partition = candidatePartitions[i];
      const variance = this.computePartitionVariance(partition, normals);
      if (variance.data < minVariance.data) {
        minVariance = variance;
        bestPartition = partition;
      }
    }

    return { bestPartition, minVariance };
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
