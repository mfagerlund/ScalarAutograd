/**
 * Test whether mesh state during compilation differs from uncompiled execution.
 * This is the smoking gun test!
 */

import { V, Value, Vec3, CompiledResiduals } from 'scalar-autograd';
import { IcoSphere } from '../src/mesh/IcoSphere';
import { DifferentiablePlaneAlignment } from '../src/energy/DifferentiablePlaneAlignment';

describe('Compilation State Test', () => {
  it.concurrent('mesh state during compilation vs uncompiled', () => {
    const mesh = IcoSphere.generate(0, 1.0); // 12 vertices

    console.log(`\n=== COMPILATION STATE TEST ===\n`);

    const params: Value[] = [];
    for (const v of mesh.vertices) {
      params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
    }

    const paramsToMesh = (p: Value[]) => {
      for (let i = 0; i < mesh.vertices.length; i++) {
        // UPDATE existing Vec3 components (don't replace the Vec3 object)
        mesh.vertices[i].x = p[3 * i];
        mesh.vertices[i].y = p[3 * i + 1];
        mesh.vertices[i].z = p[3 * i + 2];
      }
      // No need to invalidate cache - normals/areas are no longer cached
    };

    // Track what graph is built during compilation
    let compilationGraph: Value[] | null = null;

    console.log('Starting compilation...');
    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      console.log('\n>>> Inside compilation callback <<<');
      console.log(`  Received params: ${p.length} values`);
      console.log(`  First param: ${p[0].data} (requiresGrad: ${p[0].requiresGrad})`);
      console.log(`  Param object same as original? ${p[0] === params[0]}`);

      paramsToMesh(p);

      console.log(`  After paramsToMesh:`);
      console.log(`    mesh.vertices[0].x === p[0]? ${mesh.vertices[0].x === p[0]}`);
      console.log(`    mesh.vertices[0].x.data: ${mesh.vertices[0].x.data}`);

      const residuals = DifferentiablePlaneAlignment.computeResiduals(mesh);
      compilationGraph = residuals;

      console.log(`  Built ${residuals.length} residuals`);
      console.log(`  First residual value: ${residuals[0].data.toExponential(10)}`);
      console.log(`  First residual prev.length: ${(residuals[0] as any).prev?.length || 0}`);

      return residuals;
    });

    console.log('\n>>> After compilation <<<');
    console.log(`Compiled: ${compiled.kernelCount} kernels\n`);

    // Print first kernel indices
    console.log('=== FIRST KERNEL INDICES ===');
    const firstDesc = (compiled as any).functionDescriptors[0];
    console.log(`InputIndices (first 10): [${firstDesc.inputIndices.slice(0, 10).join(', ')}]`);
    console.log(`GradientIndices (first 10): [${firstDesc.gradientIndices.slice(0, 10).join(', ')}]`);
    console.log(`Total input indices: ${firstDesc.inputIndices.length}`);

    // NOW build the uncompiled graph
    console.log('Building uncompiled graph...');
    paramsToMesh(params);
    console.log(`  After paramsToMesh for uncompiled:`);
    console.log(`    mesh.vertices[0].x === params[0]? ${mesh.vertices[0].x === params[0]}`);
    console.log(`    mesh.vertices[0].x.data: ${mesh.vertices[0].x.data}`);

    const uncompiledResiduals = DifferentiablePlaneAlignment.computeResiduals(mesh);
    console.log(`  Built ${uncompiledResiduals.length} residuals`);
    console.log(`  First residual value: ${uncompiledResiduals[0].data.toExponential(10)}`);
    console.log(`  First residual prev.length: ${(uncompiledResiduals[0] as any).prev?.length || 0}`);

    // Compare the graphs
    console.log(`\n=== GRAPH COMPARISON ===`);
    console.log(`Compilation graph first residual: ${compilationGraph![0].data.toExponential(15)}`);
    console.log(`Uncompiled graph first residual:  ${uncompiledResiduals[0].data.toExponential(15)}`);
    console.log(`Same object? ${compilationGraph![0] === uncompiledResiduals[0]}`);
    console.log(`Value difference: ${Math.abs(compilationGraph![0].data - uncompiledResiduals[0].data).toExponential(6)}`);

    // Check leaf nodes
    console.log(`\n=== LEAF NODE ANALYSIS ===`);
    function collectLeaves(node: Value): Value[] {
      const visited = new Set<Value>();
      const leaves: Value[] = [];
      function traverse(n: Value) {
        if (visited.has(n)) return;
        visited.add(n);
        const prev = (n as any).prev as Value[];
        if (prev.length === 0) {
          if (!leaves.includes(n)) leaves.push(n);
        } else {
          for (const child of prev) traverse(child);
        }
      }
      traverse(node);
      return leaves;
    }

    const compilationLeaves = collectLeaves(compilationGraph![0]);
    const uncompiledLeaves = collectLeaves(uncompiledResiduals[0]);

    console.log(`Compilation graph leaf count: ${compilationLeaves.length}`);
    console.log(`Uncompiled graph leaf count: ${uncompiledLeaves.length}`);
    console.log(`First 5 compilation leaves:`);
    for (let i = 0; i < Math.min(5, compilationLeaves.length); i++) {
      const leaf = compilationLeaves[i];
      const isParam = params.includes(leaf);
      console.log(`  [${i}] data=${leaf.data.toExponential(6)}, isParam=${isParam}, requiresGrad=${leaf.requiresGrad}`);
    }
    console.log(`First 5 uncompiled leaves:`);
    for (let i = 0; i < Math.min(5, uncompiledLeaves.length); i++) {
      const leaf = uncompiledLeaves[i];
      const isParam = params.includes(leaf);
      console.log(`  [${i}] data=${leaf.data.toExponential(6)}, isParam=${isParam}, requiresGrad=${leaf.requiresGrad}`);
    }

    // Check if leaf objects are the same
    console.log(`\n=== LEAF IDENTITY CHECK ===`);
    let sameLeafObjects = 0;
    let diffLeafObjects = 0;
    for (let i = 0; i < compilationLeaves.length; i++) {
      const compLeaf = compilationLeaves[i];
      const uncompLeaf = uncompiledLeaves[i];
      if (compLeaf === uncompLeaf) {
        sameLeafObjects++;
      } else {
        diffLeafObjects++;
        if (diffLeafObjects <= 5) {
          console.log(`  Diff[${i}]: comp.data=${compLeaf.data.toExponential(6)}, uncomp.data=${uncompLeaf.data.toExponential(6)}, sameValue=${compLeaf.data === uncompLeaf.data}`);
        }
      }
    }
    console.log(`Same leaf objects: ${sameLeafObjects}/${compilationLeaves.length}`);
    console.log(`Different leaf objects: ${diffLeafObjects}/${compilationLeaves.length}`);

    // Check if the param leaves are actually the params
    console.log(`\n=== PARAM VERIFICATION ===`);
    console.log(`Checking if compilation leaves that are params are actually in params array:`);
    let paramsInCompilation = 0;
    for (const leaf of compilationLeaves) {
      if (params.includes(leaf)) paramsInCompilation++;
    }
    console.log(`Params found in compilation leaves: ${paramsInCompilation}/36 expected`);

    console.log(`\nChecking if mesh vertices point to params:`);
    for (let i = 0; i < Math.min(3, mesh.vertices.length); i++) {
      console.log(`  vertex[${i}].x === params[${3*i}]? ${mesh.vertices[i].x === params[3*i]}`);
      console.log(`  vertex[${i}].y === params[${3*i+1}]? ${mesh.vertices[i].y === params[3*i+1]}`);
      console.log(`  vertex[${i}].z === params[${3*i+2}]? ${mesh.vertices[i].z === params[3*i+2]}`);
    }

    // Now compute gradients from both
    console.log(`\n=== GRADIENT COMPARISON ===`);

    // EXPERIMENT: Use the SAME graph (compilation graph) for uncompiled backward
    params.forEach(p => p.grad = 0);
    const uncompiledEnergy = V.sum(compilationGraph!);
    console.log(`\nUncompiled (using COMPILATION graph): summing ${compilationGraph!.length} residuals`);
    console.log(`  Total energy: ${uncompiledEnergy.data.toExponential(15)}`);
    console.log(`  First 3 residuals: [${compilationGraph!.slice(0, 3).map(r => r.data.toExponential(6)).join(', ')}]`);
    uncompiledEnergy.backward();
    const uncompiledGrads = params.map(p => p.grad);

    // Compiled gradients
    paramsToMesh(params);
    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    console.log(`\nFirst 5 gradients:`);
    console.log(`Idx | Uncompiled       | Compiled         | Difference`);
    console.log(`----|------------------|------------------|------------------`);
    for (let i = 0; i < Math.min(5, params.length); i++) {
      const diff = Math.abs(uncompiledGrads[i] - compiledGrads[i]);
      console.log(`${i.toString().padStart(3)} | ${uncompiledGrads[i].toExponential(10)} | ${compiledGrads[i].toExponential(10)} | ${diff.toExponential(6)}`);
    }

    const maxDiff = Math.max(...params.map((_, i) => Math.abs(uncompiledGrads[i] - compiledGrads[i])));
    console.log(`\nMax gradient difference: ${maxDiff.toExponential(6)}`);

    if (maxDiff > 1e-10) {
      console.log(`\n❌ GRADIENTS DIFFER! This confirms the bug.\n`);
    } else {
      console.log(`\n✓ Gradients match.\n`);
    }
  });
});
