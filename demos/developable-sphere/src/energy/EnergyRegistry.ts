import { TriangleMesh } from '../mesh/TriangleMesh';
import { Value } from 'scalar-autograd';

/**
 * Interface for developable energy functions
 */
export interface DevelopableEnergyFunction {
  name: string;
  description: string;
  compute(mesh: TriangleMesh): Value;
  computeResiduals(mesh: TriangleMesh): Value[];
  classifyVertices(
    mesh: TriangleMesh,
    threshold?: number
  ): { hingeVertices: number[]; seamVertices: number[] };
}

/**
 * Global registry of available energy functions
 */
export class EnergyRegistry {
  private static energies: Map<string, DevelopableEnergyFunction> = new Map();

  static register(energy: DevelopableEnergyFunction): void {
    this.energies.set(energy.name, energy);
  }

  static get(name: string): DevelopableEnergyFunction | undefined {
    return this.energies.get(name);
  }

  static getAll(): DevelopableEnergyFunction[] {
    return Array.from(this.energies.values());
  }

  static getNames(): string[] {
    return Array.from(this.energies.keys());
  }
}
