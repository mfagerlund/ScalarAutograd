import { TriangleMesh } from '../mesh/TriangleMesh';
import { DevelopableEnergy } from './DevelopableEnergy';
import { AlignmentBimodalEnergy } from './AlignmentBimodalEnergy';
import { EigenProxyEnergy } from './EigenProxyEnergy';

/**
 * Unified vertex classification using percentile-based thresholds.
 *
 * This solves two problems:
 * 1. Different energy functions have different scales (variance vs alignment vs eigenvalues)
 * 2. Fixed thresholds don't adapt to the mesh state
 *
 * Instead of arbitrary thresholds, we classify vertices by their energy percentile:
 * - Bottom 30% (lowest energy) = "developable" (flat regions, normals aligned)
 * - Top 70% (highest energy) = "non-developable" (curved regions, normals scattered)
 *
 * This makes all energy functions comparable and adapts to the mesh.
 */
export class EnergyClassifier {
  /**
   * Classify vertices using percentile-based threshold.
   *
   * @param mesh - The mesh to classify
   * @param energyType - Which energy function to use
   * @param developablePercentile - Percentile threshold (default 30 = bottom 30%)
   * @returns Classification with hinge (low energy) and seam (high energy) vertices
   */
  static classifyVertices(
    mesh: TriangleMesh,
    energyType: 'bimodal' | 'alignment' | 'eigenproxy',
    developablePercentile: number = 30
  ): { hingeVertices: number[]; seamVertices: number[] } {
    // Choose energy function
    const EnergyClass =
      energyType === 'eigenproxy' ? EigenProxyEnergy :
      energyType === 'alignment' ? AlignmentBimodalEnergy :
      DevelopableEnergy;

    // Compute energy for each vertex
    const vertexEnergies: { index: number; energy: number }[] = [];
    for (let i = 0; i < mesh.vertices.length; i++) {
      const energy = EnergyClass.computeVertexEnergy(i, mesh).data;
      vertexEnergies.push({ index: i, energy });
    }

    // Sort by energy (ascending)
    vertexEnergies.sort((a, b) => a.energy - b.energy);

    // Classify based on percentile
    const developableCount = Math.floor(mesh.vertices.length * (developablePercentile / 100));
    const hingeVertices: number[] = [];
    const seamVertices: number[] = [];

    for (let i = 0; i < vertexEnergies.length; i++) {
      if (i < developableCount) {
        // Bottom X% = hinges (low energy = developable)
        hingeVertices.push(vertexEnergies[i].index);
      } else {
        // Top (100-X)% = seams (high energy = curved)
        seamVertices.push(vertexEnergies[i].index);
      }
    }

    return { hingeVertices, seamVertices };
  }

  /**
   * Compute developability percentage (vertices below threshold).
   */
  static computeDevelopabilityPercentage(
    _mesh: TriangleMesh,
    _energyType: 'bimodal' | 'alignment' | 'eigenproxy',
    percentile: number = 30
  ): number {
    // This is trivial with percentile-based classification
    return percentile;
  }

  /**
   * Get energy statistics for analysis.
   */
  static getEnergyStats(
    mesh: TriangleMesh,
    energyType: 'bimodal' | 'alignment' | 'eigenproxy'
  ): {
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
    const EnergyClass =
      energyType === 'eigenproxy' ? EigenProxyEnergy :
      energyType === 'alignment' ? AlignmentBimodalEnergy :
      DevelopableEnergy;

    const energies: number[] = [];
    for (let i = 0; i < mesh.vertices.length; i++) {
      energies.push(EnergyClass.computeVertexEnergy(i, mesh).data);
    }

    energies.sort((a, b) => a - b);

    const getPercentile = (p: number) => {
      const index = Math.floor((energies.length - 1) * (p / 100));
      return energies[index];
    };

    const sum = energies.reduce((acc, e) => acc + e, 0);

    return {
      min: energies[0],
      max: energies[energies.length - 1],
      mean: sum / energies.length,
      median: getPercentile(50),
      p10: getPercentile(10),
      p30: getPercentile(30),
      p50: getPercentile(50),
      p70: getPercentile(70),
      p90: getPercentile(90),
    };
  }
}
