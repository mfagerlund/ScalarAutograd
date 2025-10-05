// Benchmark: Compare eager vs lazy vs no compilation
const { V } = require('../../dist/Value');
const { CompiledResiduals, LazyCompiledResiduals } = require('../../dist/index');
const { TriangleMesh } = require('../dist/mesh/TriangleMesh');
const { createIcosphere } = require('../dist/mesh/IcosphereFactory');

// Import ridge energy (has data-dependent branching)
const { RidgeBasedEnergy } = require('../dist/energy/RidgeBasedEnergy');
// Import differentiable plane alignment (static, no branching)
const { DifferentiablePlaneAlignment } = require('../dist/energy/DifferentiablePlaneAlignment');

console.log('='.repeat(70));
console.log('Compilation Mode Benchmark');
console.log('='.repeat(70));

// Create test mesh
const mesh = createIcosphere(1);
console.log(`\nMesh: ${mesh.vertices.length} vertices, ${mesh.faces.length} faces\n`);

// Convert mesh to params
function meshToParams(mesh) {
  const params = [];
  for (const v of mesh.vertices) {
    params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
  }
  return params;
}

function paramsToMesh(params, mesh) {
  for (let i = 0; i < mesh.vertices.length; i++) {
    mesh.setVertexPosition(i, {
      x: params[3 * i],
      y: params[3 * i + 1],
      z: params[3 * i + 2]
    });
  }
}

// Test 1: Static energy (DifferentiablePlaneAlignment)
console.log('TEST 1: Static Energy (Differentiable Plane Alignment)');
console.log('-'.repeat(70));

{
  const params = meshToParams(mesh);
  const iterations = 10;

  // Mode 1: No compilation (pure graph-based)
  {
    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
      paramsToMesh(params, mesh);
      const residuals = DifferentiablePlaneAlignment.computeResiduals(mesh);

      // Manual backward pass
      residuals.forEach(r => {
        r.grad = 1;
        r.backward();
      });
    }
    const time = performance.now() - t0;
    console.log(`  No compilation:  ${time.toFixed(1)}ms (${(time/iterations).toFixed(1)}ms/iter)`);
  }

  // Mode 2: Eager compilation
  {
    const params2 = meshToParams(mesh);
    paramsToMesh(params2, mesh);

    const compileStart = performance.now();
    const compiled = CompiledResiduals.compile(params2, (p) => {
      paramsToMesh(p, mesh);
      return DifferentiablePlaneAlignment.computeResiduals(mesh);
    });
    const compileTime = performance.now() - compileStart;

    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
      compiled.evaluateJacobian(params2);
    }
    const time = performance.now() - t0;

    console.log(`  Eager compile:   ${time.toFixed(1)}ms (${(time/iterations).toFixed(1)}ms/iter) [compile: ${compileTime.toFixed(1)}ms, ${compiled.kernelCount} kernels, ${compiled.kernelReuseFactor.toFixed(1)}x reuse]`);
  }

  // Mode 3: Lazy compilation
  {
    const params3 = meshToParams(mesh);
    const lazy = new LazyCompiledResiduals(params3, (p) => {
      paramsToMesh(p, mesh);
      return DifferentiablePlaneAlignment.computeResiduals(mesh);
    });

    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
      lazy.evaluateJacobian(params3);
    }
    const time = performance.now() - t0;

    console.log(`  Lazy compile:    ${time.toFixed(1)}ms (${(time/iterations).toFixed(1)}ms/iter) [${lazy.compilations} compilations, ${lazy.kernelCount} kernels]`);
  }
}

// Test 2: Dynamic energy (RidgeBasedEnergy)
console.log('\nTEST 2: Dynamic Energy (Ridge Detection - data-dependent branching)');
console.log('-'.repeat(70));

{
  const params = meshToParams(mesh);
  const iterations = 10;

  // Mode 1: No compilation
  {
    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
      paramsToMesh(params, mesh);
      const residuals = RidgeBasedEnergy.computeResiduals(mesh);

      residuals.forEach(r => {
        r.grad = 1;
        r.backward();
      });
    }
    const time = performance.now() - t0;
    console.log(`  No compilation:  ${time.toFixed(1)}ms (${(time/iterations).toFixed(1)}ms/iter)`);
  }

  // Mode 2: Eager compilation (WRONG - will use iteration 0's graph forever)
  {
    const params2 = meshToParams(mesh);
    paramsToMesh(params2, mesh);

    const compileStart = performance.now();
    const compiled = CompiledResiduals.compile(params2, (p) => {
      paramsToMesh(p, mesh);
      return RidgeBasedEnergy.computeResiduals(mesh);
    });
    const compileTime = performance.now() - compileStart;

    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
      compiled.evaluateJacobian(params2);
    }
    const time = performance.now() - t0;

    console.log(`  Eager compile:   ${time.toFixed(1)}ms (${(time/iterations).toFixed(1)}ms/iter) [compile: ${compileTime.toFixed(1)}ms, ${compiled.kernelCount} kernels] ⚠️  INCORRECT (uses iter 0 graph)`);
  }

  // Mode 3: Lazy compilation (CORRECT - recompiles as graph changes)
  {
    const params3 = meshToParams(mesh);
    const lazy = new LazyCompiledResiduals(params3, (p) => {
      paramsToMesh(p, mesh);
      return RidgeBasedEnergy.computeResiduals(mesh);
    });

    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
      lazy.evaluateJacobian(params3);
    }
    const time = performance.now() - t0;

    console.log(`  Lazy compile:    ${time.toFixed(1)}ms (${(time/iterations).toFixed(1)}ms/iter) [${lazy.compilations} compilations, ${lazy.kernelCount} kernels, ${lazy.kernelReuseFactor.toFixed(1)}x reuse]`);
  }
}

console.log('\n' + '='.repeat(70));
console.log('\nSummary:');
console.log('  • Static energies: Eager compilation is fastest (1 compile, infinite reuse)');
console.log('  • Dynamic energies: Lazy compilation is correct AND fast (compiles as needed)');
console.log('  • No compilation: Slowest but always works (good for debugging)');
console.log('='.repeat(70));
