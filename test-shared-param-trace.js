// Test to trace how parameters are shared in the mesh
import { V, Value, Vec3, CompiledResiduals } from './dist/index.js';
import { IcoSphere } from './demos/developable-sphere/dist/mesh/IcoSphere.js';

const mesh = IcoSphere.generate(0, 1.0);
const params = [];
for (const v of mesh.vertices) {
  params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
}

console.log(`Total params: ${params.length}`);
console.log(`Total vertices: ${mesh.vertices.length}`);

const paramsToMesh = (p) => {
  for (let i = 0; i < mesh.vertices.length; i++) {
    mesh.vertices[i].x = p[3 * i];
    mesh.vertices[i].y = p[3 * i + 1];
    mesh.vertices[i].z = p[3 * i + 2];
  }
};

// Call once to set up mesh
paramsToMesh(params);

const star = mesh.getVertexStar(0);
console.log(`\nVertex 0 star (neighboring faces): ${star}`);

// Get face 0 and print its vertices
const face0 = mesh.faces[star[0]];
console.log(`\nFace ${star[0]} vertex indices: [${face0.a}, ${face0.b}, ${face0.c}]`);

// Get the Vec3 objects for face 0
const v0_face0 = mesh.vertices[face0.a];
const v1_face0 = mesh.vertices[face0.b];
const v2_face0 = mesh.vertices[face0.c];

console.log(`\nFace ${star[0]} vertex 0 Value object IDs:`);
console.log(`  v0.x === params[${face0.a * 3}]: ${v0_face0.x === params[face0.a * 3]}`);
console.log(`  v0.y === params[${face0.a * 3 + 1}]: ${v0_face0.y === params[face0.a * 3 + 1]}`);
console.log(`  v0.z === params[${face0.a * 3 + 2}]: ${v0_face0.z === params[face0.a * 3 + 2]}`);

// Get face 1
const face1 = mesh.faces[star[1]];
console.log(`\nFace ${star[1]} vertex indices: [${face1.a}, ${face1.b}, ${face1.c}]`);

// Check if vertex 0 appears in face 1
console.log(`\nVertex 0 appears in face ${star[1]}?`);
console.log(`  face1.a = ${face1.a}, face1.b = ${face1.b}, face1.c = ${face1.c}`);
console.log(`  Contains vertex 0: ${face1.a === 0 || face1.b === 0 || face1.c === 0}`);

// Get the value objects
const v0_face1_a = mesh.vertices[face1.a];
const v0_face1_b = mesh.vertices[face1.b];
const v0_face1_c = mesh.vertices[face1.c];

console.log(`\nSame Value objects used?`);
if (face1.a === 0) {
  console.log(`  v0_face1_a.x === v0_face0.x: ${v0_face1_a.x === v0_face0.x}`);
  console.log(`  v0_face1_a === v0_face0: ${v0_face1_a === v0_face0}`);
}
if (face1.b === 0) {
  console.log(`  v0_face1_b.x === v0_face0.x: ${v0_face1_b.x === v0_face0.x}`);
  console.log(`  v0_face1_b === v0_face0: ${v0_face1_b === v0_face0}`);
}
if (face1.c === 0) {
  console.log(`  v0_face1_c.x === v0_face0.x: ${v0_face1_c.x === v0_face0.x}`);
  console.log(`  v0_face1_c === v0_face0: ${v0_face1_c === v0_face0}`);
}

console.log(`\nDirect check: params[0] used in multiple faces?`);
console.log(`  v0_face0.x === params[0]: ${v0_face0.x === params[0]}`);
console.log(`  v0_face1_a.x === params[0]: ${v0_face1_a.x === params[0]}`);
