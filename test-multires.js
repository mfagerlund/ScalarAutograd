// Quick test of multi-resolution optimization
import { IcoSphere } from './demos/developable-sphere/src/mesh/IcoSphere.js';
import { SubdividedMesh } from './demos/developable-sphere/src/mesh/SubdividedMesh.js';
import { DevelopableOptimizer } from './demos/developable-sphere/src/optimization/DevelopableOptimizer.js';
import { CurvatureClassifier } from './demos/developable-sphere/src/energy/CurvatureClassifier.js';

console.log('=== Multi-Resolution Optimization Demo ===\n');

// Create base mesh at level 0 (12 vertices, 20 faces)
const baseMesh = IcoSphere.generate(0, 1.0);
const subdividedBase = SubdividedMesh.fromMesh(baseMesh);

console.log(`Base mesh: ${baseMesh.vertices.length} vertices, ${baseMesh.faces.length} faces`);
console.log(`Starting multi-resolution optimization: Level 0 → 1 → 2\n`);

const startTime = Date.now();

// Run multi-resolution optimization
const result = await DevelopableOptimizer.optimizeMultiResolution(subdividedBase, {
  startLevel: 0,
  targetLevel: 2,
  iterationsPerLevel: 50,
  gradientTolerance: 1e-6,
  verbose: true,
  coarseEnergyType: 'combinatorial',  // E^P for coarse
  fineEnergyType: 'covariance',       // E^λ for fine
  useCompiled: false,
});

const timeElapsed = (Date.now() - startTime) / 1000;

// Analyze final mesh
const finalMesh = result.history[result.history.length - 1];
const classification = CurvatureClassifier.classifyVertices(finalMesh);
const dev = (classification.hingeVertices.length / finalMesh.vertices.length) * 100;
const regions = CurvatureClassifier.countDevelopableRegions(finalMesh);

console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
console.log(`FINAL RESULTS:`);
console.log(`  Time: ${timeElapsed.toFixed(2)}s`);
console.log(`  Total iterations: ${result.iterations}`);
console.log(`  Subdivision levels: ${result.subdivisionLevels.join(' → ')}`);
console.log(`\nFinal Mesh:`);
console.log(`  Vertices: ${finalMesh.vertices.length}`);
console.log(`  Faces: ${finalMesh.faces.length}`);
console.log(`\nDevelopability:`);
console.log(`  Developability: ${dev.toFixed(1)}%`);
console.log(`  Regions: ${regions}`);
console.log(`  Quality: ${(dev / Math.max(1, regions)).toFixed(1)}% per region`);
console.log(`\nEnergy progression:`);
for (let i = 0; i < result.subdivisionLevels.length; i++) {
  console.log(`  Level ${result.subdivisionLevels[i]}: ${result.energiesPerLevel[i].toExponential(3)}`);
}
console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`);

// Quality assessment
if (regions <= 12) {
  console.log(`✓ EXCELLENT: ${regions} regions (target: 8-12 for sphere)`);
} else if (regions <= 20) {
  console.log(`✓ GOOD: ${regions} regions`);
} else {
  console.log(`⚠ WARNING: ${regions} regions (may have fragmentation)`);
}

if (dev > 60) {
  console.log(`✓ EXCELLENT developability: ${dev.toFixed(1)}%`);
} else if (dev > 30) {
  console.log(`✓ GOOD developability: ${dev.toFixed(1)}%`);
} else {
  console.log(`⚠ LOW developability: ${dev.toFixed(1)}%`);
}

console.log(`\nMulti-resolution optimization complete!`);
