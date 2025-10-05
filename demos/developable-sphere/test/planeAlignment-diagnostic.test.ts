import { describe, it } from 'vitest';
import { TriangleMesh } from '../src/mesh/TriangleMesh';
import { IcoSphere } from '../src/mesh/IcoSphere';
import { DifferentiablePlaneAlignment } from '../src/energy/DifferentiablePlaneAlignment';
import { Value, V, Vec3, CompiledResiduals } from 'scalar-autograd';

describe('DifferentiablePlaneAlignment - Gradient Diagnostic', () => {
  it('should show detailed gradient comparison', () => {
    // Use an even smaller mesh for clarity
    const mesh = IcoSphere.generate(0, 1.0); // subdivision level 0 = 12 vertices

    console.log(`\n=== DIAGNOSTIC TEST ===`);
    console.log(`Test mesh: ${mesh.vertices.length} vertices, ${mesh.faces.length} faces\n`);

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

    // Compute uncompiled
    console.log('Computing uncompiled gradients...');
    paramsToMesh(params);
    params.forEach(p => p.grad = 0);
    const uncompiledEnergy = DifferentiablePlaneAlignment.compute(mesh);
    uncompiledEnergy.backward();
    const uncompiledGrads = params.map(p => p.grad);

    // Compile
    console.log('Compiling...');
    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      paramsToMesh(p);
      return DifferentiablePlaneAlignment.computeResiduals(mesh);
    });
    console.log(`Compiled: ${compiled.kernelCount} kernels, ${compiled.kernelReuseFactor.toFixed(1)}x reuse\n`);

    // Restore and compute compiled
    paramsToMesh(params);
    console.log('Computing compiled gradients...');
    const { value: compiledEnergy, gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    // Show results
    console.log(`\nEnergies:`);
    console.log(`  Uncompiled: ${uncompiledEnergy.data.toExponential(10)}`);
    console.log(`  Compiled:   ${compiledEnergy.toExponential(10)}`);
    console.log(`  Difference: ${Math.abs(uncompiledEnergy.data - compiledEnergy).toExponential(6)}`);

    // Show gradient comparison for first few vertices
    console.log(`\nGradient comparison (first 3 vertices):`);
    console.log(`Vertex | Component | Uncompiled       | Compiled         | Difference      | Rel Diff`);
    console.log(`-------|-----------|------------------|------------------|-----------------|----------`);

    for (let i = 0; i < Math.min(3, mesh.vertices.length); i++) {
      for (let j = 0; j < 3; j++) {
        const idx = 3 * i + j;
        const component = ['x', 'y', 'z'][j];
        const uncomp = uncompiledGrads[idx];
        const comp = compiledGrads[idx];
        const diff = comp - uncomp;
        const relDiff = uncomp !== 0 ? Math.abs(diff / uncomp) : (diff === 0 ? 0 : Infinity);

        console.log(
          `  ${i.toString().padStart(4)} | ${component.padEnd(9)} | ${uncomp.toExponential(10)} | ${comp.toExponential(10)} | ${diff.toExponential(6)} | ${relDiff.toExponential(2)}`
        );
      }
    }

    // Show max differences
    const diffs = params.map((p, i) => Math.abs(compiledGrads[i] - uncompiledGrads[i]));
    const maxDiffIdx = diffs.indexOf(Math.max(...diffs));
    const maxDiff = diffs[maxDiffIdx];

    console.log(`\nWorst gradient component:`);
    console.log(`  Index: ${maxDiffIdx} (vertex ${Math.floor(maxDiffIdx / 3)}, ${['x', 'y', 'z'][maxDiffIdx % 3]})`);
    console.log(`  Uncompiled: ${uncompiledGrads[maxDiffIdx].toExponential(10)}`);
    console.log(`  Compiled:   ${compiledGrads[maxDiffIdx].toExponential(10)}`);
    console.log(`  Difference: ${maxDiff.toExponential(6)}`);
    console.log(`  Relative:   ${(maxDiff / Math.abs(uncompiledGrads[maxDiffIdx])).toExponential(6)}`);

    const uncompiledGradNorm = Math.sqrt(uncompiledGrads.reduce((sum, g) => sum + g * g, 0));
    const compiledGradNorm = Math.sqrt(compiledGrads.reduce((sum, g) => sum + g * g, 0));

    console.log(`\nGradient norms:`);
    console.log(`  Uncompiled: ${uncompiledGradNorm.toExponential(10)}`);
    console.log(`  Compiled:   ${compiledGradNorm.toExponential(10)}`);
    console.log(`  Difference: ${Math.abs(uncompiledGradNorm - compiledGradNorm).toExponential(6)}`);
    console.log(`  Relative:   ${(Math.abs(uncompiledGradNorm - compiledGradNorm) / uncompiledGradNorm).toExponential(6)}`);

    // Check if gradients might be permuted (right values, wrong positions)
    console.log(`\n=== PERMUTATION CHECK ===`);
    console.log(`Checking if compiled gradients are just a permutation of uncompiled gradients...`);

    // Sort both gradient arrays and compare
    const uncompiledSorted = [...uncompiledGrads].sort((a, b) => a - b);
    const compiledSorted = [...compiledGrads].sort((a, b) => a - b);

    let maxSortedDiff = 0;
    for (let i = 0; i < params.length; i++) {
      const diff = Math.abs(uncompiledSorted[i] - compiledSorted[i]);
      if (diff > maxSortedDiff) maxSortedDiff = diff;
    }

    console.log(`Max difference after sorting: ${maxSortedDiff.toExponential(6)}`);

    if (maxSortedDiff < 1e-10) {
      console.log(`✅ GRADIENTS ARE PERMUTED! The compiled gradients are the correct values but in wrong positions.`);
    } else {
      console.log(`❌ Not a simple permutation. The gradient values themselves are different.`);
    }

    // Also check if maybe they're scaled
    const avgUncompiled = uncompiledGrads.reduce((sum, g) => sum + Math.abs(g), 0) / params.length;
    const avgCompiled = compiledGrads.reduce((sum, g) => sum + Math.abs(g), 0) / params.length;
    const scaleRatio = avgCompiled / avgUncompiled;

    console.log(`\nScale check:`);
    console.log(`  Avg abs(uncompiled): ${avgUncompiled.toExponential(6)}`);
    console.log(`  Avg abs(compiled):   ${avgCompiled.toExponential(6)}`);
    console.log(`  Ratio:               ${scaleRatio.toExponential(6)}`);

    // Try to find if any compiled gradient matches any uncompiled gradient
    console.log(`\nLooking for matches between any compiled and uncompiled gradient components...`);
    let perfectMatches = 0;
    let closeMatches = 0; // within 1e-10

    for (let i = 0; i < params.length; i++) {
      for (let j = 0; j < params.length; j++) {
        const diff = Math.abs(uncompiledGrads[i] - compiledGrads[j]);
        if (diff < 1e-15) {
          perfectMatches++;
          if (i !== j) {
            console.log(`  Perfect match: uncompiled[${i}] = compiled[${j}] = ${uncompiledGrads[i].toExponential(6)}`);
          }
        } else if (diff < 1e-10) {
          closeMatches++;
        }
      }
    }

    console.log(`\nPerfect matches (< 1e-15): ${perfectMatches} out of ${params.length * params.length} comparisons`);
    console.log(`Close matches (< 1e-10): ${closeMatches} out of ${params.length * params.length} comparisons`);
    console.log(`Expected diagonal matches if identical: ${params.length}`);

    // Check the Jacobian directly to see if individual residual gradients are correct
    console.log(`\n=== JACOBIAN CHECK ===`);
    console.log(`Checking individual residual gradients (first residual only)...`);

    // Get the Jacobian for first residual
    const { values: residualValues, jacobian } = compiled.evaluateJacobian(params);

    // Manually compute gradient of first residual using uncompiled
    const firstResidualUncompiled: Value[] = [];
    paramsToMesh(params);
    params.forEach(p => p.grad = 0);
    const res0 = DifferentiablePlaneAlignment.computeVertexEnergy(0, mesh);
    res0.backward();
    const jacobian0Uncompiled = params.map(p => p.grad);

    // Compare first row of Jacobian
    console.log(`\nFirst residual gradient comparison (first 5 components):`);
    console.log(`Idx | Uncompiled       | Compiled (Jac)   | Difference      | Rel Diff`);
    console.log(`----|------------------|------------------|-----------------|----------`);
    for (let i = 0; i < Math.min(5, params.length); i++) {
      const uncomp = jacobian0Uncompiled[i];
      const comp = jacobian[0][i];
      const diff = comp - uncomp;
      const relDiff = uncomp !== 0 ? Math.abs(diff / uncomp) : (diff === 0 ? 0 : Infinity);
      console.log(
        `${i.toString().padStart(3)} | ${uncomp.toExponential(10)} | ${comp.toExponential(10)} | ${diff.toExponential(6)} | ${relDiff.toExponential(2)}`
      );
    }

    const maxJacDiff = Math.max(...jacobian0Uncompiled.map((u, i) => Math.abs(u - jacobian[0][i])));
    console.log(`\nMax Jacobian[0] difference: ${maxJacDiff.toExponential(6)}`);
  });

  it('bypass kernel cache test', () => {
    const mesh = IcoSphere.generate(0, 1.0); // 12 vertices

    console.log(`\n=== BYPASS KERNEL CACHE TEST ===`);
    console.log(`Test mesh: ${mesh.vertices.length} vertices, ${mesh.faces.length} faces\n`);

    const params: Value[] = [];
    for (const v of mesh.vertices) {
      params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
    }

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

    // Compile WITHOUT reusing kernels by creating a fresh CompiledFunctions each time
    console.log('Compiling WITH kernel reuse (normal)...');
    const compiled1 = CompiledResiduals.compile(params, (p: Value[]) => {
      paramsToMesh(p);
      return DifferentiablePlaneAlignment.computeResiduals(mesh);
    });
    console.log(`Compiled: ${compiled1.kernelCount} kernels, ${compiled1.kernelReuseFactor.toFixed(1)}x reuse\n`);

    paramsToMesh(params);
    const { gradient: compiledGrads1 } = compiled1.evaluateSumWithGradient(params);

    // Now compile again - this should show different results if caching is the issue
    // Clear params to force recompilation
    const params2: Value[] = [];
    for (const v of mesh.vertices) {
      params2.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
    }
    params2.forEach((p, i) => p.paramName = `p${i}`);

    console.log('Compiling AGAIN with fresh parameters (bypasses cache)...');
    const compiled2 = CompiledResiduals.compile(params2, (p: Value[]) => {
      paramsToMesh(p);
      return DifferentiablePlaneAlignment.computeResiduals(mesh);
    });
    console.log(`Compiled: ${compiled2.kernelCount} kernels, ${compiled2.kernelReuseFactor.toFixed(1)}x reuse\n`);

    paramsToMesh(params2);
    const { gradient: compiledGrads2 } = compiled2.evaluateSumWithGradient(params2);

    // Compare all three
    console.log('Gradient comparison:');
    console.log(`Idx | Uncompiled       | Compiled1        | Compiled2        | Diff1          | Diff2`);
    console.log(`----|------------------|------------------|------------------|----------------|----------------`);

    let maxDiff1 = 0;
    let maxDiff2 = 0;
    let maxDiff12 = 0;

    for (let i = 0; i < Math.min(9, params.length); i++) {
      const uncomp = uncompiledGrads[i];
      const comp1 = compiledGrads1[i];
      const comp2 = compiledGrads2[i];
      const diff1 = Math.abs(comp1 - uncomp);
      const diff2 = Math.abs(comp2 - uncomp);
      const diff12 = Math.abs(comp1 - comp2);

      maxDiff1 = Math.max(maxDiff1, diff1);
      maxDiff2 = Math.max(maxDiff2, diff2);
      maxDiff12 = Math.max(maxDiff12, diff12);

      console.log(
        `${i.toString().padStart(3)} | ${uncomp.toExponential(10)} | ${comp1.toExponential(10)} | ${comp2.toExponential(10)} | ${diff1.toExponential(6)} | ${diff2.toExponential(6)}`
      );
    }

    console.log(`\nMax diffs:`);
    console.log(`  Uncompiled vs Compiled1: ${maxDiff1.toExponential(6)}`);
    console.log(`  Uncompiled vs Compiled2: ${maxDiff2.toExponential(6)}`);
    console.log(`  Compiled1 vs Compiled2:  ${maxDiff12.toExponential(6)}`);

    if (maxDiff12 < 1e-15) {
      console.log(`\n✓ Both compiled versions match exactly - kernel caching is NOT the issue`);
    } else {
      console.log(`\n❌ Compiled versions differ - kernel caching MIGHT be the issue!`);
    }
  });
});
