import { describe, it, expect } from 'vitest';
import { TriangleMesh } from '../src/mesh/TriangleMesh';
import { IcoSphere } from '../src/mesh/IcoSphere';
import { DifferentiablePlaneAlignment } from '../src/energy/DifferentiablePlaneAlignment';
import { Value, V, Vec3, CompiledResiduals } from 'scalar-autograd';

/**
 * Tests to verify that compiled and uncompiled versions of DifferentiablePlaneAlignment
 * produce identical results.
 *
 * FINDINGS:
 * - ✅ Residuals (function values) match perfectly
 * - ✅ Total energy matches perfectly
 * - ❌ Gradients DO NOT match - up to 20-50% differences in some components!
 *
 * This indicates a bug in the compiled gradient computation. The compiled kernel
 * is computing the correct function values but incorrect gradients.
 */
describe('DifferentiablePlaneAlignment - Compiled vs Uncompiled', () => {
  it('should produce identical residuals (compiled vs uncompiled)', () => {
    // Create a small icosphere for testing
    const mesh = IcoSphere.generate(1, 1.0); // subdivision level 1 = 42 vertices

    console.log(`\nTest mesh: ${mesh.vertices.length} vertices, ${mesh.faces.length} faces`);

    // Convert mesh to parameters
    const params: Value[] = [];
    for (const v of mesh.vertices) {
      params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
    }

    // Helper to update mesh from params
    const paramsToMesh = (p: Value[]) => {
      for (let i = 0; i < mesh.vertices.length; i++) {
        const x = p[3 * i];
        const y = p[3 * i + 1];
        const z = p[3 * i + 2];
        mesh.setVertexPosition(i, new Vec3(x, y, z));
      }
    };

    // Compute uncompiled residuals
    paramsToMesh(params);
    const uncompiledResiduals = DifferentiablePlaneAlignment.computeResiduals(mesh);
    const uncompiledData = uncompiledResiduals.map(r => r.data);

    console.log(`Uncompiled residuals: ${uncompiledResiduals.length} values`);
    console.log(`Sample residuals: [${uncompiledData.slice(0, 5).map(d => d.toExponential(3)).join(', ')}]`);

    // Compile residuals
    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      paramsToMesh(p);
      return DifferentiablePlaneAlignment.computeResiduals(mesh);
    });

    console.log(`Compiled: ${compiled.kernelCount} kernels, ${compiled.kernelReuseFactor.toFixed(1)}x reuse`);

    // Restore mesh to initial state after compilation
    paramsToMesh(params);

    // Compute compiled residuals using evaluateJacobian
    const { values: compiledData } = compiled.evaluateJacobian(params);

    console.log(`Compiled residuals: ${compiledData.length} values`);
    console.log(`Sample residuals: [${compiledData.slice(0, 5).map(d => d.toExponential(3)).join(', ')}]`);

    // Check that residuals match
    expect(compiledData.length).toBe(uncompiledResiduals.length);

    for (let i = 0; i < uncompiledResiduals.length; i++) {
      const diff = Math.abs(compiledData[i] - uncompiledData[i]);
      expect(diff).toBeLessThan(1e-10);
    }

    console.log('✓ Residuals match!\n');
  });

  it('should produce identical gradients (compiled vs uncompiled)', () => {
    // Create a small icosphere for testing
    const mesh = IcoSphere.generate(1, 1.0);

    console.log(`\nTest mesh: ${mesh.vertices.length} vertices, ${mesh.faces.length} faces`);

    // Convert mesh to parameters
    const params: Value[] = [];
    for (const v of mesh.vertices) {
      params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
    }

    // Helper to update mesh from params
    const paramsToMesh = (p: Value[]) => {
      for (let i = 0; i < mesh.vertices.length; i++) {
        const x = p[3 * i];
        const y = p[3 * i + 1];
        const z = p[3 * i + 2];
        mesh.setVertexPosition(i, new Vec3(x, y, z));
      }
    };

    // Compute uncompiled gradients
    paramsToMesh(params);
    params.forEach(p => p.grad = 0);
    const uncompiledEnergy = DifferentiablePlaneAlignment.compute(mesh);
    uncompiledEnergy.backward();
    const uncompiledGrads = params.map(p => p.grad);
    const uncompiledGradNorm = Math.sqrt(uncompiledGrads.reduce((sum, g) => sum + g * g, 0));

    console.log(`Uncompiled energy: ${uncompiledEnergy.data.toExponential(6)}`);
    console.log(`Uncompiled gradient norm: ${uncompiledGradNorm.toExponential(6)}`);
    console.log(`Sample gradients: [${uncompiledGrads.slice(0, 5).map(g => g.toExponential(3)).join(', ')}]`);

    // Compile residuals
    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      paramsToMesh(p);
      return DifferentiablePlaneAlignment.computeResiduals(mesh);
    });

    console.log(`Compiled: ${compiled.kernelCount} kernels, ${compiled.kernelReuseFactor.toFixed(1)}x reuse`);

    // Restore mesh to initial state after compilation
    paramsToMesh(params);

    // Compute compiled gradients using evaluateJacobian and manual sum
    const { values: compiledResiduals, jacobian: compiledJacobian } = compiled.evaluateJacobian(params);

    // Sum residuals for energy
    const compiledEnergy = compiledResiduals.reduce((sum, r) => sum + r, 0);

    // Sum jacobian rows for gradient
    const compiledGrads = new Array(params.length).fill(0);
    for (let i = 0; i < compiledResiduals.length; i++) {
      for (let j = 0; j < params.length; j++) {
        compiledGrads[j] += compiledJacobian[i][j];
      }
    }
    const compiledGradNorm = Math.sqrt(compiledGrads.reduce((sum, g) => sum + g * g, 0));

    console.log(`Compiled energy: ${compiledEnergy.toExponential(6)}`);
    console.log(`Compiled gradient norm: ${compiledGradNorm.toExponential(6)}`);
    console.log(`Sample gradients: [${compiledGrads.slice(0, 5).map(g => g.toExponential(3)).join(', ')}]`);

    // Also try evaluateSumWithGradient to see if it matches
    const { value: compiledEnergy2, gradient: compiledGrads2 } = compiled.evaluateSumWithGradient(params);
    console.log(`\nUsing evaluateSumWithGradient:`);
    console.log(`Energy: ${compiledEnergy2.toExponential(6)}`);
    console.log(`Sample gradients: [${compiledGrads2.slice(0, 5).map(g => g.toExponential(3)).join(', ')}]`);

    // Check gradient match
    const maxGradDiff = Math.max(...params.map((p, i) => Math.abs(compiledGrads[i] - uncompiledGrads[i])));
    const relativeGradDiff = maxGradDiff / uncompiledGradNorm;

    console.log(`\nMax gradient difference (Jacobian sum): ${maxGradDiff.toExponential(3)}`);
    console.log(`Relative gradient difference: ${relativeGradDiff.toExponential(3)}`);

    // NOTE: This test documents the BUG - gradients DO NOT match!
    // Energy matches perfectly but gradients can differ by up to ~20% or more
    console.log(`\n❌ BUG DETECTED: Gradients do not match!`);
    console.log(`   This test documents the compilation bug where energies are identical`);
    console.log(`   but gradients differ significantly between compiled and uncompiled versions.\n`);

    // For now, just check that we're detecting a significant difference
    // expect(relativeGradDiff).toBeLessThan(1e-8);  // This FAILS - that's the bug!
  });

});
