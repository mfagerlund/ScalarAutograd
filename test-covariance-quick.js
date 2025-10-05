// Quick test of covariance energy optimization
import { IcoSphere } from './demos/developable-sphere/src/mesh/IcoSphere.js';
import { DevelopableOptimizer } from './demos/developable-sphere/src/optimization/DevelopableOptimizer.js';
import { CovarianceEnergy } from './demos/developable-sphere/src/energy/CovarianceEnergy.js';
import { CurvatureClassifier } from './demos/developable-sphere/src/energy/CurvatureClassifier.js';

console.log('Testing Covariance Energy Optimization...\n');

// Create sphere
const sphere = IcoSphere.generate(2, 1.0);
console.log(`Mesh: ${sphere.vertices.length} vertices, ${sphere.faces.length} faces`);

// Initial metrics
const initialEnergy = CovarianceEnergy.compute(sphere).data;
const initialClassification = CurvatureClassifier.classifyVertices(sphere);
const initialDev = (initialClassification.hingeVertices.length / sphere.vertices.length) * 100;
const initialRegions = CurvatureClassifier.countDevelopableRegions(sphere);

console.log(`\nInitial State:`);
console.log(`  Covariance Energy: ${initialEnergy.toExponential(3)}`);
console.log(`  Developability: ${initialDev.toFixed(1)}%`);
console.log(`  Regions: ${initialRegions}`);

// Optimize (non-compiled for debugging)
console.log(`\nOptimizing with Covariance Energy (non-compiled)...`);
const optimizer = new DevelopableOptimizer(sphere);

const startTime = Date.now();
const result = await optimizer.optimizeAsync({
  maxIterations: 30,
  gradientTolerance: 1e-6,
  verbose: true,
  energyType: 'covariance',
  useCompiled: false,
  chunkSize: 30,
});

const timeElapsed = (Date.now() - startTime) / 1000;

// Final metrics
const finalMesh = result.history[result.history.length - 1];
const finalEnergy = CovarianceEnergy.compute(finalMesh).data;
const finalClassification = CurvatureClassifier.classifyVertices(finalMesh);
const finalDev = (finalClassification.hingeVertices.length / finalMesh.vertices.length) * 100;
const finalRegions = CurvatureClassifier.countDevelopableRegions(finalMesh);

console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
console.log(`RESULTS:`);
console.log(`  Time: ${timeElapsed.toFixed(2)}s`);
console.log(`  Iterations: ${result.iterations}`);
console.log(`  Convergence: ${result.convergenceReason}`);
console.log(`\nEnergy:`);
console.log(`  Before: ${initialEnergy.toExponential(3)}`);
console.log(`  After:  ${finalEnergy.toExponential(3)}`);
console.log(`  Reduction: ${((1 - finalEnergy / initialEnergy) * 100).toFixed(1)}%`);
console.log(`\nDevelopability:`);
console.log(`  Before: ${initialDev.toFixed(1)}%`);
console.log(`  After:  ${finalDev.toFixed(1)}%`);
console.log(`  Improvement: +${(finalDev - initialDev).toFixed(1)}%`);
console.log(`\nRegions:`);
console.log(`  Before: ${initialRegions}`);
console.log(`  After:  ${finalRegions}`);
console.log(`  Quality: ${finalDev.toFixed(1)}% / ${finalRegions} = ${(finalDev / finalRegions).toFixed(1)}% per region`);
console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`);

// Expected: ~8-12 regions for a good sphere approximation
if (finalRegions < 20 && finalDev > 30) {
  console.log('✓ PASS: Good region count and developability!');
} else if (finalRegions > 50) {
  console.log('✗ WARNING: Too many regions - fragmentation issue!');
} else {
  console.log('? Uncertain result - may need more iterations');
}
