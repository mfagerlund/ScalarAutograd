/**
 * Backend optimizer for developable sphere
 *
 * Runs parameter sweeps using Node.js to find optimal settings.
 * Can use worker_threads for parallelization in the future.
 *
 * Usage: node backend-optimizer.js
 */

import { V } from '../../dist/index.js';
import { IcoSphere } from './dist/mesh/IcoSphere.js';
import { DevelopableOptimizer } from './dist/optimization/DevelopableOptimizer.js';
import { CurvatureClassifier } from './dist/energy/utils/CurvatureClassifier.js';
import { performance } from 'perf_hooks';
import { writeFile } from 'fs/promises';

// Parameter sweep configuration
const CONFIG = {
  subdivisions: [2, 3, 4],
  maxIterations: [50, 100, 200],
  gradientTolerances: [1e-6, 1e-7, 1e-8],
  chunkSizes: [5, 10, 20],
};

async function runOptimization(params) {
  const { subdivisions, maxIterations, gradientTolerance, chunkSize } = params;

  console.log(`\n${'='.repeat(60)}`);
  console.log(`Testing: subdiv=${subdivisions}, maxIter=${maxIterations}, gradTol=${gradientTolerance}, chunk=${chunkSize}`);
  console.log('='.repeat(60));

  const startTime = performance.now();

  // Create sphere
  const sphere = IcoSphere.generate(subdivisions, 1.0);
  const optimizer = new DevelopableOptimizer(sphere);

  // Run optimization
  const result = await optimizer.optimizeAsync({
    maxIterations,
    gradientTolerance,
    verbose: false,
    captureInterval: Math.max(1, Math.floor(maxIterations / 20)),
    chunkSize,
  });

  const endTime = performance.now();
  const timeMs = endTime - startTime;

  // Compute final metrics
  const finalMesh = result.history[result.history.length - 1];
  const classification = CurvatureClassifier.classifyVertices(finalMesh);
  const developableRatio = classification.hingeVertices.length / finalMesh.vertices.length;

  const stats = {
    params,
    developableRatio,
    finalEnergy: result.finalEnergy,
    iterations: result.iterations,
    functionEvals: result.functionEvaluations,
    convergenceReason: result.convergenceReason,
    gradientNorm: result.gradientNorm,
    kernelCount: result.kernelCount,
    kernelReuseFactor: result.kernelReuseFactor,
    timeMs,
    vertices: finalMesh.vertices.length,
    hingeVertices: classification.hingeVertices.length,
    seamVertices: classification.seamVertices.length,
  };

  console.log(`✓ Developable: ${(developableRatio * 100).toFixed(2)}%`);
  console.log(`  Energy: ${result.finalEnergy.toExponential(3)}`);
  console.log(`  Iterations: ${result.iterations}`);
  console.log(`  Function Evals: ${result.functionEvaluations}`);
  console.log(`  Kernels: ${result.kernelCount} (${result.kernelReuseFactor.toFixed(1)}x reuse)`);
  console.log(`  Convergence: ${result.convergenceReason}`);
  console.log(`  Gradient Norm: ${result.gradientNorm?.toExponential(3) || 'N/A'}`);
  console.log(`  Time: ${timeMs.toFixed(0)}ms`);

  return stats;
}

async function sweepParameters() {
  const results = [];

  console.log('Starting parameter sweep...');
  console.log(`Total combinations: ${CONFIG.subdivisions.length * CONFIG.maxIterations.length * CONFIG.gradientTolerances.length * CONFIG.chunkSizes.length}`);

  for (const subdivisions of CONFIG.subdivisions) {
    for (const maxIterations of CONFIG.maxIterations) {
      for (const gradientTolerance of CONFIG.gradientTolerances) {
        for (const chunkSize of CONFIG.chunkSizes) {
          try {
            const result = await runOptimization({
              subdivisions,
              maxIterations,
              gradientTolerance,
              chunkSize,
            });
            results.push(result);
          } catch (error) {
            console.error(`✗ Failed:`, error.message);
          }
        }
      }
    }
  }

  return results;
}

function analyzeResults(results) {
  console.log('\n' + '='.repeat(60));
  console.log('RESULTS SUMMARY');
  console.log('='.repeat(60));

  // Sort by developable ratio
  const sorted = [...results].sort((a, b) => b.developableRatio - a.developableRatio);

  console.log('\nTop 5 configurations by developable ratio:');
  sorted.slice(0, 5).forEach((r, i) => {
    console.log(`\n${i + 1}. ${(r.developableRatio * 100).toFixed(2)}% developable`);
    console.log(`   subdiv=${r.params.subdivisions}, maxIter=${r.params.maxIterations}, gradTol=${r.params.gradientTolerance}, chunk=${r.params.chunkSize}`);
    console.log(`   Energy: ${r.finalEnergy.toExponential(3)}, Time: ${r.timeMs.toFixed(0)}ms`);
    console.log(`   Convergence: ${r.convergenceReason}`);
  });

  // Best for each subdivision level
  console.log('\n\nBest configuration per subdivision level:');
  for (const subdiv of CONFIG.subdivisions) {
    const best = sorted.find(r => r.params.subdivisions === subdiv);
    if (best) {
      console.log(`\nSubdivision ${subdiv} (${best.vertices} vertices):`);
      console.log(`  ${(best.developableRatio * 100).toFixed(2)}% developable`);
      console.log(`  maxIter=${best.params.maxIterations}, gradTol=${best.params.gradientTolerance}, chunk=${best.params.chunkSize}`);
      console.log(`  Time: ${best.timeMs.toFixed(0)}ms, Kernels: ${best.kernelCount}`);
    }
  }
}

// Run the sweep
sweepParameters()
  .then(async results => {
    analyzeResults(results);

    // Save results to JSON
    await writeFile(
      './optimization-results.json',
      JSON.stringify(results, null, 2)
    );
    console.log('\n✓ Results saved to optimization-results.json');
  })
  .catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
