import { Value } from './Value';

/**
 * Registry for tracking unique Value objects across residual graphs.
 * Handles deduplication of constants and optionally variables.
 * @internal
 */
export class ValueRegistry {
  private values: Value[] = [];
  private valueToId = new Map<Value, number>();

  /**
   * Register a Value and return its unique ID.
   * Deduplication rules:
   * - Constants (requiresGrad=false, no prev): dedupe by data value only
   * - Variables (requiresGrad=true, has paramName): dedupe by paramName
   * - Weights & computed values: always unique
   */
  register(value: Value): number {
    // Check if already registered
    if (this.valueToId.has(value)) {
      return this.valueToId.get(value)!;
    }

    // Constants: dedupe by value only (ignore labels/paramNames)
    if (!value.requiresGrad && (value as any).prev.length === 0) {
      const existing = this.values.find(v =>
        !v.requiresGrad &&
        (v as any).prev.length === 0 &&
        v.data === value.data
      );
      if (existing) {
        const existingId = this.valueToId.get(existing)!;
        this.valueToId.set(value, existingId);
        value._registryId = existingId;
        return existingId;
      }
    }

    // Variables: dedupe by paramName if present
    if (value.requiresGrad && value.paramName && (value as any).prev.length === 0) {
      const existing = this.values.find(v =>
        v.requiresGrad &&
        v.paramName === value.paramName &&
        (v as any).prev.length === 0
      );
      if (existing) {
        const existingId = this.valueToId.get(existing)!;
        this.valueToId.set(value, existingId);
        value._registryId = existingId;
        return existingId;
      }
    }

    // Weights & computed values: always unique
    const id = this.values.length;
    this.values.push(value);
    this.valueToId.set(value, id);
    value._registryId = id;
    return id;
  }

  /**
   * Get Value by ID
   */
  getValue(id: number): Value {
    return this.values[id];
  }

  /**
   * Get all registered values as data array
   */
  getDataArray(): number[] {
    return this.values.map(v => v.data);
  }

  /**
   * Update value array from current Value.data
   */
  updateDataArray(dataArray: number[]): void {
    for (let i = 0; i < this.values.length; i++) {
      dataArray[i] = this.values[i].data;
    }
  }

  /**
   * Get ID for a Value (must be registered)
   */
  getId(value: Value): number {
    const id = this.valueToId.get(value);
    if (id === undefined) {
      throw new Error('Value not registered');
    }
    return id;
  }

  /**
   * Total number of unique values
   */
  get size(): number {
    return this.values.length;
  }
}
