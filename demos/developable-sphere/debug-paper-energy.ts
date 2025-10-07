/**
 * DEBUG SCRIPT: PaperCovarianceEnergyELambda Training
 *
 * PURPOSE:
 * This script trains PaperCovarianceEnergyELambda on a 1-subdivision sphere to debug why it's not learning.
 * The energy should decrease over iterations, pushing the sphere toward a developable surface.
 *
 * GROUND TRUTH REFERENCE:
 * C:\Dev\ScalarAutograd\demos\developable-sphere\src\energy\hinge_energy.cpp (lines 114-216)
 *
 * OUR IMPLEMENTATION:
 * C:\Dev\ScalarAutograd\demos\developable-sphere\src\energy\PaperCovarianceEnergyELambda.ts
 *
 * KEY ALGORITHM (from C++ reference):
 * For each vertex i:
 *   1. Compute area-weighted vertex normal: Nv = normalize(Σ area_f * Nf)
 *   2. For each adjacent face f:
 *      a. Get face normal Nf and interior angle theta
 *      b. Compute tangent projection: Nfw = normalize((Nv × Nf) × Nv) * acos(Nv·Nf)
 *      c. Add to covariance: mat += theta * Nfw * Nfw^T
 *   3. Eigendecompose mat and return smallest eigenvalue λ_min
 *   4. Energy = Σ λ_min over all vertices
 *
 * EXPECTED BEHAVIOR:
 * - Initial energy should be > 0 (sphere is not developable)
 * - Energy should decrease over iterations
 * - L-BFGS should not fail on first step
 * - Gradients should be non-zero and reasonable magnitude
 *
 * KNOWN ISSUES TO CHECK:
 * 1. Are gradients zero/NaN/infinite?
 * 2. Is energy constant across iterations?
 * 3. Does L-BFGS fail with line search errors?
 * 4. Are vertex energies all zero?
 *
 * HOW TO RUN:
 * npx tsx demos/developable-sphere/debug-paper-energy.ts
 */

import { IcoSphere } from './src/mesh/IcoSphere';
import { DevelopableOptimizer } from './src/optimization/DevelopableOptimizer';
import { PaperCovarianceEnergyELambda } from './src/energy/PaperCovarianceEnergyELambda';
import { V, Vec3, Value } from 'scalar-autograd';

console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('DEBUG: PaperCovarianceEnergyELambda Training');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('');

console.log('STEP 1: Create 1-subdivision sphere');
const sphere = IcoSphere.generate(1, 1.0);
console.log(`  Vertices: ${sphere.vertices.length}`);
console.log(`  Faces: ${sphere.faces.length}`);
console.log('');

console.log('STEP 2: Inspect initial vertex energies');
console.log('  Computing energy for first 5 vertices...');
for (let i = 0; i < Math.min(5, sphere.vertices.length); i++) {
  const star = sphere.getVertexStar(i);
  const vertexEnergy = PaperCovarianceEnergyELambda.computeVertexEnergy(i, sphere);
  console.log(`  Vertex ${i}: star size=${star.length}, energy=${vertexEnergy.data.toExponential(3)}`);
}
console.log('');

console.log('STEP 3: Convert vertices to trainable parameters (like optimizer does)');
const params: Value[] = [];
for (const v of sphere.vertices) {
  params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
}
console.log(`  Created ${params.length} trainable parameters`);

for (let i = 0; i < sphere.vertices.length; i++) {
  const x = params[3 * i];
  const y = params[3 * i + 1];
  const z = params[3 * i + 2];
  sphere.vertices[i] = new Vec3(x, y, z);
}
console.log('  ✓ Mesh vertices now connected to trainable params');
console.log('');

console.log('STEP 4: Compute total initial energy');
const initialEnergy = PaperCovarianceEnergyELambda.compute(sphere);
console.log(`  Total energy: ${initialEnergy.data.toExponential(6)}`);
console.log('');

console.log('STEP 5: Check gradient computation');
console.log('  Computing gradient for total energy...');
initialEnergy.backward();

let gradMagnitudeSum = 0;
let gradCount = 0;
let nonZeroGrads = 0;
let maxGrad = 0;
let minGrad = Infinity;

for (let i = 0; i < params.length; i += 3) {
  const gx = Math.abs(params[i].grad);
  const gy = Math.abs(params[i + 1].grad);
  const gz = Math.abs(params[i + 2].grad);
  const gradMag = Math.sqrt(gx * gx + gy * gy + gz * gz);

  if (gradMag > 0) nonZeroGrads++;
  gradMagnitudeSum += gradMag;
  gradCount++;
  maxGrad = Math.max(maxGrad, gradMag);
  minGrad = Math.min(minGrad, gradMag);
}

const avgGradMagnitude = gradMagnitudeSum / gradCount;
console.log(`  Average gradient magnitude: ${avgGradMagnitude.toExponential(3)}`);
console.log(`  Max gradient magnitude: ${maxGrad.toExponential(3)}`);
console.log(`  Min gradient magnitude: ${minGrad.toExponential(3)}`);
console.log(`  Non-zero gradients: ${nonZeroGrads} / ${gradCount * 3}`);

if (avgGradMagnitude === 0) {
  console.log('  ❌ ERROR: All gradients are zero! Energy is constant.');
  console.log('  This means the energy function is not differentiable or there is a bug.');
  process.exit(1);
}

if (!isFinite(avgGradMagnitude)) {
  console.log('  ❌ ERROR: Gradients are NaN or infinite!');
  process.exit(1);
}

console.log('  ✓ Gradients look reasonable');
console.log('');

console.log('STEP 6: Run optimization (non-compiled, L-BFGS)');
console.log('  Settings:');
console.log('    - Subdivisions: 1');
console.log('    - Max iterations: 10');
console.log('    - Energy: PaperCovarianceEnergyELambda');
console.log('    - Optimizer: lbfgs');
console.log('    - Compiled: false');
console.log('');

const optimizer = new DevelopableOptimizer(sphere);

let iterationCount = 0;
const energyHistory: number[] = [];

const result = await optimizer.optimizeAsync({
  maxIterations: 10,
  gradientTolerance: 1e-8,
  verbose: true,
  captureInterval: 1,
  chunkSize: 1,
  energyType: 'PaperCovarianceEnergyELambda',
  useCompiled: false,
  optimizer: 'lbfgs',
  onProgress: (iteration, energy, history) => {
    iterationCount = iteration;
    energyHistory.push(energy);
    console.log(`  Iteration ${iteration}: energy = ${energy.toExponential(6)}`);

    if (iteration === 0 && energy === initialEnergy.data) {
      console.log('    ✓ Initial energy matches');
    }

    if (iteration > 0) {
      const prevEnergy = energyHistory[energyHistory.length - 2];
      const energyChange = energy - prevEnergy;
      const energyChangePercent = (energyChange / prevEnergy) * 100;

      console.log(`    Energy change: ${energyChange.toExponential(3)} (${energyChangePercent.toFixed(2)}%)`);

      if (energyChange > 0) {
        console.log('    ⚠ Warning: Energy increased!');
      } else if (Math.abs(energyChange) < 1e-12) {
        console.log('    ⚠ Warning: Energy not changing!');
      }
    }
  },
});

console.log('');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('OPTIMIZATION RESULTS');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log(`Iterations: ${result.iterations}`);
console.log(`Convergence: ${result.convergenceReason}`);
console.log(`Function evaluations: ${result.functionEvaluations}`);
console.log('');

console.log('Energy progression:');
if (energyHistory.length > 0) {
  const initialEnergyVal = energyHistory[0];
  const finalEnergyVal = energyHistory[energyHistory.length - 1];
  const totalChange = finalEnergyVal - initialEnergyVal;
  const totalChangePercent = (totalChange / initialEnergyVal) * 100;

  console.log(`  Initial: ${initialEnergyVal.toExponential(6)}`);
  console.log(`  Final:   ${finalEnergyVal.toExponential(6)}`);
  console.log(`  Change:  ${totalChange.toExponential(6)} (${totalChangePercent.toFixed(2)}%)`);

  if (result.iterations < 2) {
    console.log('');
    console.log('❌ FAILURE: Optimization stopped after < 2 iterations');
    console.log('   L-BFGS likely failed on first step');
    console.log('   This suggests:');
    console.log('   1. Line search failed (energy not decreasing along search direction)');
    console.log('   2. Gradients are wrong (not matching true descent direction)');
    console.log('   3. Energy function is not smooth');
  } else if (Math.abs(totalChange) < 1e-10) {
    console.log('');
    console.log('⚠ WARNING: Energy barely changed');
    console.log('   Possible causes:');
    console.log('   1. Energy is already at minimum (check if sphere is already developable)');
    console.log('   2. Gradients are too small');
    console.log('   3. Energy landscape is flat');
  } else if (totalChange < 0) {
    console.log('');
    console.log('✓ SUCCESS: Energy decreased');
  } else {
    console.log('');
    console.log('❌ FAILURE: Energy increased overall');
  }
}
console.log('');

console.log('DIAGNOSIS CHECKLIST:');
console.log(`  [${avgGradMagnitude > 0 ? '✓' : '✗'}] Gradients are non-zero`);
console.log(`  [${isFinite(avgGradMagnitude) ? '✓' : '✗'}] Gradients are finite`);
console.log(`  [${result.iterations >= 2 ? '✓' : '✗'}] Optimizer ran for >= 2 iterations`);
console.log(`  [${energyHistory.length >= 2 && energyHistory[energyHistory.length - 1] < energyHistory[0] ? '✓' : '✗'}] Energy decreased`);
console.log('');

console.log('NEXT STEPS:');
if (result.iterations < 2) {
  console.log('  1. Check gradient correctness with finite differences');
  console.log('  2. Verify energy computation matches C++ reference');
  console.log('  3. Check if eigenvalue computation has correct gradients');
  console.log('  4. Inspect Matrix3x3.smallestEigenvalueCustomGrad implementation');
} else if (Math.abs(energyHistory[energyHistory.length - 1] - energyHistory[0]) < 1e-10) {
  console.log('  1. Verify initial sphere is NOT already developable');
  console.log('  2. Check if energy scale is appropriate');
  console.log('  3. Consider adding perturbation to initial geometry');
} else {
  console.log('  1. Run for more iterations to see convergence');
  console.log('  2. Compare vertex energies between iterations');
  console.log('  3. Visualize the optimized mesh');
}
console.log('');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
