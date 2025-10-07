/**
 * Compact energy function debugger
 * Usage: npx tsx demos/developable-sphere/debug-energy.ts <EnergyName>
 */

import { IcoSphere } from './src/mesh/IcoSphere';
import { DevelopableOptimizer } from './src/optimization/DevelopableOptimizer';
import { EnergyRegistry } from './src/energy/EnergyRegistry';
import { V, Vec3, Value } from 'scalar-autograd';

import './src/energy/DevelopableEnergy';
import './src/energy/PaperCovarianceEnergyELambda';
import './src/energy/RidgeBasedEnergy';
import './src/energy/AlignmentBimodalEnergy';
import './src/energy/PaperPartitionEnergyEP';
import './src/energy/PaperPartitionEnergyEPStochastic';
import './src/energy/EigenProxyEnergy';
import './src/energy/FastCovarianceEnergy';
import './src/energy/GreatCircleEnergyEx';
import './src/energy/DifferentiablePlaneAlignment';

const energyName = process.argv[2] || 'PaperCovarianceEnergyELambda';
const maxIters = parseInt(process.argv[3] || '10');
const subdivisions = parseInt(process.argv[4] || '1');
const energyFn = EnergyRegistry.get(energyName);
if (!energyFn) {
  console.error(`Unknown: ${energyName}. Available: ${EnergyRegistry.getNames().join(', ')}`);
  process.exit(1);
}

console.log(`Testing: ${energyName} (subdiv=${subdivisions})`);

const sphere = IcoSphere.generate(subdivisions, 1.0);
console.log(`Mesh: ${sphere.vertices.length}v, ${sphere.faces.length}f`);

const params: Value[] = [];
for (const v of sphere.vertices) {
  params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
}
for (let i = 0; i < sphere.vertices.length; i++) {
  sphere.vertices[i] = new Vec3(params[3*i], params[3*i+1], params[3*i+2]);
}

const e0 = energyFn.compute(sphere);
console.log(`E0: ${e0.data.toExponential(3)}`);

// Measure developability if available
let dev0 = null;
if ('computeQualityMetrics' in energyFn) {
  dev0 = (energyFn as any).computeQualityMetrics(sphere);
  console.log(`Dev0: ${dev0.developabilityPct.toFixed(1)}% (${dev0.numRegions.toFixed(0)} regions)`);
}

e0.backward();
let gradSum = 0;
for (const p of params) gradSum += Math.abs(p.grad);
const avgGrad = gradSum / params.length;

if (avgGrad === 0) {
  console.log('FAIL: Zero gradients');
  process.exit(1);
}
console.log(`Grad: ${avgGrad.toExponential(2)}`);

const opt = new DevelopableOptimizer(sphere);
const t0 = Date.now();
const result = await opt.optimizeAsync({
  maxIterations: maxIters,
  gradientTolerance: 1e-8,
  verbose: false,
  captureInterval: maxIters,
  chunkSize: maxIters,
  energyType: energyName,
  useCompiled: false,
  optimizer: 'lbfgs',
});
const time = ((Date.now() - t0) / 1000).toFixed(1);

const ef = result.finalEnergy;
const change = ((ef - e0.data) / e0.data) * 100;

console.log(`E${maxIters}: ${ef.toExponential(3)} (${change.toFixed(1)}%)`);

// Measure developability after optimization
if (dev0 && 'computeQualityMetrics' in energyFn) {
  const devF = (energyFn as any).computeQualityMetrics(sphere);
  const devChange = devF.developabilityPct - dev0.developabilityPct;
  console.log(`DevF: ${devF.developabilityPct.toFixed(1)}% (${devF.numRegions.toFixed(0)} regions) [+${devChange.toFixed(1)}%]`);
}

console.log(`Time: ${time}s, Iters: ${result.iterations}`);

if (result.iterations < 2) {
  console.log('FAIL: Stopped too early');
  process.exit(1);
}

if (Math.abs(change) < 0.1) {
  console.log('WARN: Energy barely changed');
}

if (change > 0) {
  console.log('FAIL: Energy increased');
  process.exit(1);
}

console.log('PASS');
