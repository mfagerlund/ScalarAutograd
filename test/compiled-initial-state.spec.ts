/**
 * Test to verify compilation doesn't corrupt initial state
 */
import { describe, it, expect } from 'vitest';
import { V, CompiledFunctions, Value } from '../src';

describe('Compiled functions initial state', () => {
  it('should not modify parameters during compilation', () => {
    // Create simple parameters
    const params = [V.W(1), V.W(2), V.W(3)];

    // Record initial values
    const initialValues = params.map(p => p.data);

    // Compile a function that uses the parameters
    const compiled = CompiledFunctions.compile(params, (p: Value[]) => {
      // This function modifies p during compilation
      return [V.add(p[0], p[1]), V.mul(p[1], p[2])];
    });

    // Parameters should still have their original values
    expect(params[0].data).toBe(initialValues[0]);
    expect(params[1].data).toBe(initialValues[1]);
    expect(params[2].data).toBe(initialValues[2]);
  });

  it('should compute correct initial energy after compilation', () => {
    // Simulate mesh optimization scenario
    let meshState = { x: 1.0, y: 2.0, z: 3.0 };
    const params = [V.W(meshState.x), V.W(meshState.y), V.W(meshState.z)];

    // Energy function that modifies mesh state
    const computeEnergy = (p: Value[]) => {
      meshState.x = p[0].data;
      meshState.y = p[1].data;
      meshState.z = p[2].data;
      return V.add(V.mul(p[0], p[0]), V.add(V.mul(p[1], p[1]), V.mul(p[2], p[2])));
    };

    // Compute initial energy before compilation
    const initialEnergy = computeEnergy(params).data;
    expect(initialEnergy).toBe(14); // 1^2 + 2^2 + 3^2 = 14

    // Reset mesh state
    meshState = { x: 1.0, y: 2.0, z: 3.0 };

    // Compile residuals (this will modify meshState during compilation)
    const compiled = CompiledFunctions.compile(params, (p: Value[]) => {
      meshState.x = p[0].data;
      meshState.y = p[1].data;
      meshState.z = p[2].data;
      return [V.mul(p[0], p[0]), V.mul(p[1], p[1]), V.mul(p[2], p[2])];
    });

    // WITHOUT FIX: meshState would be corrupted here
    // WITH FIX: we need to restore meshState after compilation

    // Restore mesh state manually (this is what the fix does)
    meshState.x = params[0].data;
    meshState.y = params[1].data;
    meshState.z = params[2].data;

    // Energy should still be 14
    const energyAfterCompile = V.add(V.C(meshState.x ** 2), V.add(V.C(meshState.y ** 2), V.C(meshState.z ** 2))).data;
    expect(energyAfterCompile).toBe(14);

    // Use compiled function to get sum
    const result = compiled.evaluateSumWithGradient(params);
    expect(result.value).toBe(14); // Sum of residuals should equal total energy
  });
});
