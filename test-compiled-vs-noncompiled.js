import { V, Vec3, lbfgs, CompiledFunctions } from './dist/index.js';

// Simple sphere test - just 12 vertices (icosphere base)
const t = (1.0 + Math.sqrt(5.0)) / 2.0;
let vertices = [
  Vec3.W(-1, t, 0), Vec3.W(1, t, 0), Vec3.W(-1, -t, 0), Vec3.W(1, -t, 0),
  Vec3.W(0, -1, t), Vec3.W(0, 1, t), Vec3.W(0, -1, -t), Vec3.W(0, 1, -t),
  Vec3.W(t, 0, -1), Vec3.W(t, 0, 1), Vec3.W(-t, 0, -1), Vec3.W(-t, 0, 1),
].map(v => {
  const len = Math.sqrt(v.x.data * v.x.data + v.y.data * v.y.data + v.z.data * v.z.data);
  return Vec3.W(v.x.data / len, v.y.data / len, v.z.data / len);
});

// Simple energy: sum of squared distances from origin minus 1
function computeResiduals(verts) {
  const residuals = [];
  for (let i = 0; i < verts.length; i++) {
    const v = verts[i];
    const distSq = V.add(V.add(V.mul(v.x, v.x), V.mul(v.y, v.y)), V.mul(v.z, v.z));
    const diff = V.sub(distSq, V.C(1));
    residuals.push(diff);
  }
  return residuals;
}

// Test 1: Non-compiled
const params1 = [];
for (const v of vertices) {
  params1.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
}

const result1 = lbfgs(params1, (p) => {
  const verts = [];
  for (let i = 0; i < 12; i++) {
    verts.push(new Vec3(p[3*i], p[3*i+1], p[3*i+2]));
  }
  const residuals = computeResiduals(verts);
  return V.sum(residuals);
}, { maxIterations: 50, verbose: false });

console.log('Non-compiled:');
console.log('  Energy:', result1.finalCost);
console.log('  Iterations:', result1.iterations);
console.log('  First 3 params:', params1.slice(0, 3).map(p => p.data));

// Test 2: Compiled
const params2 = [];
for (const v of vertices) {
  params2.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
}

const compiled = CompiledFunctions.compile(params2, (p) => {
  const verts = [];
  for (let i = 0; i < 12; i++) {
    verts.push(new Vec3(p[3*i], p[3*i+1], p[3*i+2]));
  }
  return computeResiduals(verts);
});

const result2 = lbfgs(params2, compiled, { maxIterations: 50, verbose: false });

console.log('\nCompiled:');
console.log('  Energy:', result2.finalCost);
console.log('  Iterations:', result2.iterations);
console.log('  First 3 params:', params2.slice(0, 3).map(p => p.data));

console.log('\nDifference:');
console.log('  Energy diff:', Math.abs(result1.finalCost - result2.finalCost));
console.log('  Param diff:', Math.max(...params1.map((p, i) => Math.abs(p.data - params2[i].data))));
