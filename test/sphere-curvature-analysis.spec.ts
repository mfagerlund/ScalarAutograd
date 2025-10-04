import { describe, it, expect } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { CurvatureClassifier } from '../demos/developable-sphere/src/energy/CurvatureClassifier';
import { TriangleMesh } from '../demos/developable-sphere/src/mesh/TriangleMesh';
import { Vec3 } from '../src/Vec3';

describe('Sphere Curvature Analysis', () => {
  it('measures angle defect at different subdivision levels', () => {
    console.log('\nAngle defect on unit spheres (adaptive threshold):\n');

    for (let sub = 2; sub <= 5; sub++) {
      const sphere = IcoSphere.generate(sub, 1.0);
      const stats = CurvatureClassifier.getCurvatureStats(sphere);

      const totalCurvature = stats.mean * sphere.vertices.length;
      const expectedTotal = 4 * Math.PI;

      // Use adaptive threshold (default multiplier = 0.1)
      const classification = CurvatureClassifier.classifyVertices(sphere, 0.1);
      const devRatio = classification.hingeVertices.length / sphere.vertices.length;

      const adaptiveThreshold = 0.1 * (4 * Math.PI / sphere.vertices.length);

      console.log(`Subdivision ${sub} (${sphere.vertices.length} verts):`);
      console.log(`  Mean angle defect: ${stats.mean.toExponential(3)} rad`);
      console.log(`  Adaptive threshold: ${adaptiveThreshold.toExponential(3)} rad`);
      console.log(`  Classification: ${(devRatio * 100).toFixed(1)}% developable`);
      console.log(`  Total curvature: ${totalCurvature.toFixed(4)} (expected ${expectedTotal.toFixed(4)})`);
      console.log('');

      // Verify total curvature is conserved (Gauss-Bonnet theorem)
      expect(totalCurvature).toBeCloseTo(expectedTotal, 1);

      // Sphere should NEVER be classified as mostly developable
      expect(devRatio).toBeLessThan(0.2); // Less than 20% should be "flat"
    }

    console.log('✓ Adaptive threshold correctly classifies spheres as curved at all subdivisions');
  });

  it('correctly identifies flat surfaces as developable', () => {
    // Create a flat quad mesh (two triangles)
    const v0 = Vec3.C(0, 0, 0);
    const v1 = Vec3.C(1, 0, 0);
    const v2 = Vec3.C(1, 1, 0);
    const v3 = Vec3.C(0, 1, 0);

    const vertices = [v0, v1, v2, v3];
    const faces = [
      { vertices: [0, 1, 2] as [number, number, number] },
      { vertices: [0, 2, 3] as [number, number, number] }
    ];

    const flatMesh = new TriangleMesh(vertices, faces);
    const classification = CurvatureClassifier.classifyVertices(flatMesh);
    const devRatio = classification.hingeVertices.length / flatMesh.vertices.length;

    console.log('\nFlat quad mesh:');
    console.log(`  Vertices: ${flatMesh.vertices.length}`);
    console.log(`  Developable: ${(devRatio * 100).toFixed(1)}%`);

    // Flat surface should be mostly developable (boundary vertices may have different angles)
    expect(devRatio).toBeGreaterThan(0.5); // At least 50% should be flat
  });
});
