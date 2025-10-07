import { IcoSphere } from './dist/mesh/IcoSphere.js';
import { DevelopableEnergy } from './dist/energy/DevelopableEnergy.js';
import { CurvatureClassifier } from './dist/energy/utils/CurvatureClassifier.js';

const sphere = IcoSphere.generate(3, 1.0);

console.log('Testing energy on icosphere (subdiv 3)');
console.log(`Vertices: ${sphere.vertices.length}, Faces: ${sphere.faces.length}\n`);

const bimodalEnergy = DevelopableEnergy.compute(sphere).data;

const classification = CurvatureClassifier.classifyVertices(sphere, 0.1);

console.log('ENERGY:');
console.log(`  Energy: ${bimodalEnergy.toFixed(4)}`);
console.log(`  Hinges: ${classification.hingeVertices.length} (${(classification.hingeVertices.length/sphere.vertices.length*100).toFixed(1)}%)`);
