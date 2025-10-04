import { IcoSphere } from './dist/mesh/IcoSphere.js';
import { DevelopableEnergy } from './dist/energy/DevelopableEnergy.js';
import { ContiguousBimodalEnergy } from './dist/energy/ContiguousBimodalEnergy.js';

const sphere = IcoSphere.generate(3, 1.0);

console.log('Testing bimodal vs contiguous on icosphere (subdiv 3)');
console.log(`Vertices: ${sphere.vertices.length}, Faces: ${sphere.faces.length}\n`);

const bimodalEnergy = DevelopableEnergy.compute(sphere).data;
const contiguousEnergy = ContiguousBimodalEnergy.compute(sphere).data;

const bimodalClass = DevelopableEnergy.classifyVertices(sphere, 1e-3);
const contiguousClass = ContiguousBimodalEnergy.classifyVertices(sphere, 1e-3);

console.log('BIMODAL (quasi-random split):');
console.log(`  Energy: ${bimodalEnergy.toFixed(4)}`);
console.log(`  Hinges: ${bimodalClass.hingeVertices.length} (${(bimodalClass.hingeVertices.length/sphere.vertices.length*100).toFixed(1)}%)`);

console.log('\nCONTIGUOUS (spatial split):');
console.log(`  Energy: ${contiguousEnergy.toFixed(4)}`);
console.log(`  Hinges: ${contiguousClass.hingeVertices.length} (${(contiguousClass.hingeVertices.length/sphere.vertices.length*100).toFixed(1)}%)`);

const diff = ((contiguousEnergy - bimodalEnergy) / bimodalEnergy * 100);
console.log(`\nDifference: ${diff > 0 ? '+' : ''}${diff.toFixed(1)}%`);
