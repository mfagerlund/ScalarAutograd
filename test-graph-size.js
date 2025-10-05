// Estimate graph size for developable sphere
const subdivisions = 3;

// Icosphere vertex count after subdivisions
let vertices = 12;
for (let i = 0; i < subdivisions; i++) {
  vertices = vertices * 4; // Rough approximation
}

console.log(`Vertices: ${vertices}`);

// Each vertex has ~6 neighbors (faces in star)
const avgFacesPerVertex = 6;
console.log(`Total faces in stars: ${vertices * avgFacesPerVertex}`);

// Each vertex energy:
// - Split into 2 regions (3 faces each)
// - Each region: compute mean (3 normals × 3 components = 9 Values)
// - Each region: compute variance (3 normals × deviation calc = ~20 Values)
// - Total per vertex: ~40 Values
const valuesPerVertex = 40;
console.log(`Graph nodes for all residuals: ${vertices * valuesPerVertex}`);

// During compilation, ALL these nodes exist in memory simultaneously
console.log(`Estimated total graph size: ${vertices * valuesPerVertex} Value objects`);
