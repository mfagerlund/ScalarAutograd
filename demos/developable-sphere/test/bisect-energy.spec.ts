/**
 * Bisect DifferentiablePlaneAlignment to find the minimal failing part.
 */

import { V, Value, Vec3, CompiledResiduals } from 'scalar-autograd';
import { IcoSphere } from '../src/mesh/IcoSphere';
import { TriangleMesh } from '../src/mesh/TriangleMesh';

describe('Bisect Energy Function', () => {

  function testEnergyFunction(name: string, computeFn: (vertexIdx: number, mesh: TriangleMesh) => Value) {
    console.log(`\n=== TESTING: ${name} ===`);

    const mesh = IcoSphere.generate(0, 1.0);
    const params: Value[] = [];
    for (const v of mesh.vertices) {
      params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
    }

    const paramsToMesh = (p: Value[]) => {
      for (let i = 0; i < mesh.vertices.length; i++) {
        mesh.vertices[i].x = p[3 * i];
        mesh.vertices[i].y = p[3 * i + 1];
        mesh.vertices[i].z = p[3 * i + 2];
      }
    };

    // Compile
    let compilationGraph: Value[] | null = null;
    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      paramsToMesh(p);
      const residuals: Value[] = [];
      for (let i = 0; i < mesh.vertices.length; i++) {
        residuals.push(computeFn(i, mesh));
      }
      compilationGraph = residuals;
      return residuals;
    });

    // Graph backward
    params.forEach(p => p.grad = 0);
    const graphEnergy = V.sum(compilationGraph!);
    graphEnergy.backward();
    const graphGrads = params.map(p => p.grad);

    // Compiled backward
    paramsToMesh(params);
    const { gradient: compiledGrads } = compiled.evaluateSumWithGradient(params);

    // Compare
    const maxDiff = Math.max(...params.map((_, i) => Math.abs(graphGrads[i] - compiledGrads[i])));
    console.log(`Max gradient diff: ${maxDiff.toExponential(6)}`);

    if (maxDiff < 1e-10) {
      console.log(`✅ PASS - Gradients match!`);
    } else {
      console.log(`❌ FAIL - Gradients differ!`);
      console.log(`First 5 gradients:`);
      for (let i = 0; i < 5; i++) {
        console.log(`  [${i}] graph=${graphGrads[i].toExponential(6)}, compiled=${compiledGrads[i].toExponential(6)}`);
      }
    }
  }

  it('Full energy function', () => {
    testEnergyFunction('Full DifferentiablePlaneAlignment', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 2) return V.C(0);
      if (star.length === 3) return V.C(0);

      const normals: Vec3[] = [];
      const angles: Value[] = [];
      for (const faceIdx of star) {
        normals.push(mesh.getFaceNormal(faceIdx).normalized);
        angles.push(mesh.getInteriorAngle(faceIdx, vertexIdx));
      }

      let planeNormal = Vec3.zero();
      for (let i = 0; i < normals.length; i++) {
        for (let j = i + 1; j < normals.length; j++) {
          const cross = Vec3.cross(normals[i], normals[j]);
          const dotProduct = Vec3.dot(normals[i], normals[j]);
          const separation = V.sub(V.C(1), dotProduct);
          planeNormal = planeNormal.add(cross.mul(separation));
        }
      }

      const planeNormalMag = planeNormal.magnitude;
      const epsilon = V.C(1e-12);
      const safeMag = V.max(planeNormalMag, epsilon);
      const planeNormalNormalized = planeNormal.div(safeMag);

      let energy = V.C(0);
      for (let i = 0; i < normals.length; i++) {
        const dist = V.abs(Vec3.dot(normals[i], planeNormalNormalized));
        const weightedDist = V.mul(angles[i], V.mul(dist, dist));
        energy = V.add(energy, weightedDist);
      }

      return V.div(energy, star.length);
    });
  });

  it('Just the normals part', () => {
    testEnergyFunction('Just normals computation', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 2) return V.C(0);
      if (star.length === 3) return V.C(0);

      const normals: Vec3[] = [];
      for (const faceIdx of star) {
        normals.push(mesh.getFaceNormal(faceIdx).normalized);
      }

      // Just return magnitude of first normal as a simple test
      return normals[0].magnitude;
    });
  });

  it('Just cross products', () => {
    testEnergyFunction('Just cross products', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 2) return V.C(0);
      if (star.length === 3) return V.C(0);

      const normals: Vec3[] = [];
      for (const faceIdx of star) {
        normals.push(mesh.getFaceNormal(faceIdx).normalized);
      }

      let planeNormal = Vec3.zero();
      for (let i = 0; i < normals.length; i++) {
        for (let j = i + 1; j < normals.length; j++) {
          const cross = Vec3.cross(normals[i], normals[j]);
          planeNormal = planeNormal.add(cross);
        }
      }

      return planeNormal.magnitude;
    });
  });

  it('Cross products WITH separation weighting', () => {
    testEnergyFunction('Cross products with separation', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 2) return V.C(0);
      if (star.length === 3) return V.C(0);

      const normals: Vec3[] = [];
      for (const faceIdx of star) {
        normals.push(mesh.getFaceNormal(faceIdx).normalized);
      }

      let planeNormal = Vec3.zero();
      for (let i = 0; i < normals.length; i++) {
        for (let j = i + 1; j < normals.length; j++) {
          const cross = Vec3.cross(normals[i], normals[j]);
          const dotProduct = Vec3.dot(normals[i], normals[j]);
          const separation = V.sub(V.C(1), dotProduct);
          planeNormal = planeNormal.add(cross.mul(separation));
        }
      }

      return planeNormal.magnitude;
    });
  });

  it('Up to normalized plane normal', () => {
    testEnergyFunction('Normalized plane normal', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 2) return V.C(0);
      if (star.length === 3) return V.C(0);

      const normals: Vec3[] = [];
      for (const faceIdx of star) {
        normals.push(mesh.getFaceNormal(faceIdx).normalized);
      }

      let planeNormal = Vec3.zero();
      for (let i = 0; i < normals.length; i++) {
        for (let j = i + 1; j < normals.length; j++) {
          const cross = Vec3.cross(normals[i], normals[j]);
          const dotProduct = Vec3.dot(normals[i], normals[j]);
          const separation = V.sub(V.C(1), dotProduct);
          planeNormal = planeNormal.add(cross.mul(separation));
        }
      }

      const planeNormalMag = planeNormal.magnitude;
      const epsilon = V.C(1e-12);
      const safeMag = V.max(planeNormalMag, epsilon);
      const planeNormalNormalized = planeNormal.div(safeMag);

      return planeNormalNormalized.magnitude;
    });
  });

  it('Energy accumulation WITHOUT angles', () => {
    testEnergyFunction('Energy accumulation (no angles)', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 2) return V.C(0);
      if (star.length === 3) return V.C(0);

      const normals: Vec3[] = [];
      for (const faceIdx of star) {
        normals.push(mesh.getFaceNormal(faceIdx).normalized);
      }

      let planeNormal = Vec3.zero();
      for (let i = 0; i < normals.length; i++) {
        for (let j = i + 1; j < normals.length; j++) {
          const cross = Vec3.cross(normals[i], normals[j]);
          const dotProduct = Vec3.dot(normals[i], normals[j]);
          const separation = V.sub(V.C(1), dotProduct);
          planeNormal = planeNormal.add(cross.mul(separation));
        }
      }

      const planeNormalMag = planeNormal.magnitude;
      const epsilon = V.C(1e-12);
      const safeMag = V.max(planeNormalMag, epsilon);
      const planeNormalNormalized = planeNormal.div(safeMag);

      let energy = V.C(0);
      for (let i = 0; i < normals.length; i++) {
        const dist = V.abs(Vec3.dot(normals[i], planeNormalNormalized));
        energy = V.add(energy, V.mul(dist, dist));  // No angle weighting
      }

      return energy;
    });
  });

  it('Energy accumulation WITH angles', () => {
    testEnergyFunction('Energy accumulation (with angles)', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 2) return V.C(0);
      if (star.length === 3) return V.C(0);

      const normals: Vec3[] = [];
      const angles: Value[] = [];
      for (const faceIdx of star) {
        normals.push(mesh.getFaceNormal(faceIdx).normalized);
        angles.push(mesh.getInteriorAngle(faceIdx, vertexIdx));
      }

      let planeNormal = Vec3.zero();
      for (let i = 0; i < normals.length; i++) {
        for (let j = i + 1; j < normals.length; j++) {
          const cross = Vec3.cross(normals[i], normals[j]);
          const dotProduct = Vec3.dot(normals[i], normals[j]);
          const separation = V.sub(V.C(1), dotProduct);
          planeNormal = planeNormal.add(cross.mul(separation));
        }
      }

      const planeNormalMag = planeNormal.magnitude;
      const epsilon = V.C(1e-12);
      const safeMag = V.max(planeNormalMag, epsilon);
      const planeNormalNormalized = planeNormal.div(safeMag);

      let energy = V.C(0);
      for (let i = 0; i < normals.length; i++) {
        const dist = V.abs(Vec3.dot(normals[i], planeNormalNormalized));
        const weightedDist = V.mul(angles[i], V.mul(dist, dist));
        energy = V.add(energy, weightedDist);
      }

      return energy;  // No division by star.length yet
    });
  });

  it('JUST the angles', () => {
    testEnergyFunction('Just angles', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 2) return V.C(0);
      if (star.length === 3) return V.C(0);

      const angles: Value[] = [];
      for (const faceIdx of star) {
        angles.push(mesh.getInteriorAngle(faceIdx, vertexIdx));
      }

      // Just sum the angles
      let sum = V.C(0);
      for (const angle of angles) {
        sum = V.add(sum, angle);
      }
      return sum;
    });
  });

  it('Angles AND one face normal (combined)', () => {
    testEnergyFunction('Angles + one normal', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 2) return V.C(0);
      if (star.length === 3) return V.C(0);

      const angles: Value[] = [];
      for (const faceIdx of star) {
        angles.push(mesh.getInteriorAngle(faceIdx, vertexIdx));
      }

      // Get ONE normal
      const normal = mesh.getFaceNormal(star[0]).normalized;

      // Return angle sum + normal magnitude
      let sum = V.C(0);
      for (const angle of angles) {
        sum = V.add(sum, angle);
      }
      return V.add(sum, normal.magnitude);
    });
  });

  it('Normals used in TWO different ways', () => {
    testEnergyFunction('Normals used twice', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 2) return V.C(0);
      if (star.length === 3) return V.C(0);

      const normals: Vec3[] = [];
      const angles: Value[] = [];
      for (const faceIdx of star) {
        normals.push(mesh.getFaceNormal(faceIdx).normalized);
        angles.push(mesh.getInteriorAngle(faceIdx, vertexIdx));
      }

      // Use normals to build planeNormal
      let planeNormal = Vec3.zero();
      for (let i = 0; i < normals.length; i++) {
        for (let j = i + 1; j < normals.length; j++) {
          const cross = Vec3.cross(normals[i], normals[j]);
          planeNormal = planeNormal.add(cross);  // No separation weighting
        }
      }
      const planeNormalNormalized = planeNormal.normalized;

      // Now use THE SAME normals array with planeNormalNormalized
      let energy = V.C(0);
      for (let i = 0; i < normals.length; i++) {
        const dist = Vec3.dot(normals[i], planeNormalNormalized);
        const weighted = V.mul(angles[i], V.mul(dist, dist));  // ADD ANGLE WEIGHTING
        energy = V.add(energy, weighted);
      }

      return energy;
    });
  });

  it('Normals WITHOUT .normalized property', () => {
    testEnergyFunction('Normals without .normalized', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 2) return V.C(0);
      if (star.length === 3) return V.C(0);

      const normals: Vec3[] = [];
      const angles: Value[] = [];
      for (const faceIdx of star) {
        // Get face normal but DON'T call .normalized
        const rawNormal = mesh.getFaceNormal(faceIdx);
        const mag = rawNormal.magnitude;
        normals.push(rawNormal.div(mag));  // Manual normalization
        angles.push(mesh.getInteriorAngle(faceIdx, vertexIdx));
      }

      // Use normals to build planeNormal
      let planeNormal = Vec3.zero();
      for (let i = 0; i < normals.length; i++) {
        for (let j = i + 1; j < normals.length; j++) {
          const cross = Vec3.cross(normals[i], normals[j]);
          planeNormal = planeNormal.add(cross);
        }
      }

      // Manual normalization instead of .normalized
      const mag = planeNormal.magnitude;
      const planeNormalNormalized = planeNormal.div(mag);

      let energy = V.C(0);
      for (let i = 0; i < normals.length; i++) {
        const dist = Vec3.dot(normals[i], planeNormalNormalized);
        const weighted = V.mul(angles[i], V.mul(dist, dist));
        energy = V.add(energy, weighted);
      }

      return energy;
    });
  });

  it('Simplified: remove planeNormal computation', () => {
    testEnergyFunction('No planeNormal', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 2) return V.C(0);
      if (star.length === 3) return V.C(0);

      const normals: Vec3[] = [];
      const angles: Value[] = [];
      for (const faceIdx of star) {
        normals.push(mesh.getFaceNormal(faceIdx).normalized);
        angles.push(mesh.getInteriorAngle(faceIdx, vertexIdx));
      }

      // Just multiply first normal magnitude by first angle
      let energy = V.C(0);
      for (let i = 0; i < normals.length; i++) {
        const val = V.mul(angles[i], normals[i].magnitude);
        energy = V.add(energy, val);
      }

      return energy;
    });
  });

  it('One normal, one angle, with planeNormal', () => {
    testEnergyFunction('Minimal with planeNormal', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 2) return V.C(0);

      // Get just TWO normals and TWO angles
      const n0 = mesh.getFaceNormal(star[0]).normalized;
      const n1 = mesh.getFaceNormal(star[1]).normalized;
      const a0 = mesh.getInteriorAngle(star[0], vertexIdx);

      // Build planeNormal from cross product
      const planeNormal = Vec3.cross(n0, n1).normalized;

      // Use n0 with planeNormal, weighted by a0
      const dist = Vec3.dot(n0, planeNormal);
      return V.mul(a0, V.mul(dist, dist));
    });
  });

  it('THREE normals - add to planeNormal', () => {
    testEnergyFunction('Three normals planeNormal', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 3) return V.C(0);

      const n0 = mesh.getFaceNormal(star[0]).normalized;
      const n1 = mesh.getFaceNormal(star[1]).normalized;
      const n2 = mesh.getFaceNormal(star[2]).normalized;
      const a0 = mesh.getInteriorAngle(star[0], vertexIdx);

      // Build planeNormal from THREE normals
      const cross01 = Vec3.cross(n0, n1);
      const cross02 = Vec3.cross(n0, n2);
      const planeNormal = cross01.add(cross02).normalized;

      // Use n0 with planeNormal, weighted by a0
      const dist = Vec3.dot(n0, planeNormal);
      return V.mul(a0, V.mul(dist, dist));
    });
  });

  it('Use BOTH n0 and n1 in final energy', () => {
    testEnergyFunction('Both normals in energy', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 3) return V.C(0);

      const n0 = mesh.getFaceNormal(star[0]).normalized;
      const n1 = mesh.getFaceNormal(star[1]).normalized;
      const n2 = mesh.getFaceNormal(star[2]).normalized;
      const a0 = mesh.getInteriorAngle(star[0], vertexIdx);
      const a1 = mesh.getInteriorAngle(star[1], vertexIdx);

      // Build planeNormal
      const cross01 = Vec3.cross(n0, n1);
      const cross02 = Vec3.cross(n0, n2);
      const planeNormal = cross01.add(cross02).normalized;

      // Use BOTH n0 and n1 with planeNormal
      const dist0 = Vec3.dot(n0, planeNormal);
      const dist1 = Vec3.dot(n1, planeNormal);

      const term0 = V.mul(a0, V.mul(dist0, dist0));
      const term1 = V.mul(a1, V.mul(dist1, dist1));

      return V.add(term0, term1);
    });
  });

  it('WITHOUT n2', () => {
    testEnergyFunction('No n2', (vertexIdx, mesh) => {
      const star = mesh.getVertexStar(vertexIdx);
      if (star.length < 2) return V.C(0);

      const n0 = mesh.getFaceNormal(star[0]).normalized;
      const n1 = mesh.getFaceNormal(star[1]).normalized;
      const a0 = mesh.getInteriorAngle(star[0], vertexIdx);
      const a1 = mesh.getInteriorAngle(star[1], vertexIdx);

      // planeNormal from just n0 and n1
      const planeNormal = Vec3.cross(n0, n1).normalized;

      const dist0 = Vec3.dot(n0, planeNormal);
      const dist1 = Vec3.dot(n1, planeNormal);

      const term0 = V.mul(a0, V.mul(dist0, dist0));
      const term1 = V.mul(a1, V.mul(dist1, dist1));

      return V.add(term0, term1);
    });
  });
});
