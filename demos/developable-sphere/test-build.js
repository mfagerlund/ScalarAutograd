// Quick test to verify imports work
import { Vec3, Geometry } from 'scalar-autograd';
import { IcoSphere } from './src/mesh/IcoSphere.ts';
import { CurvatureClassifier } from './src/energy/utils/CurvatureClassifier.ts';

console.log('Testing imports...');
console.log('Vec3:', typeof Vec3);
console.log('Geometry:', typeof Geometry);

try {
  const sphere = IcoSphere.generate(1, 1.0);
  console.log('Created sphere with', sphere.vertices.length, 'vertices');

  const classification = CurvatureClassifier.classifyVertices(sphere);
  console.log('Classification:', classification);

  console.log('✅ All imports working!');
} catch (e) {
  console.error('❌ Error:', e.message);
  console.error(e.stack);
}
