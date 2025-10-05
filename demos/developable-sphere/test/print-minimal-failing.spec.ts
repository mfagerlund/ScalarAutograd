/**
 * Print the generated kernel code for the MINIMAL FAILING CASE
 */

import { V, Value, Vec3, CompiledResiduals } from 'scalar-autograd';
import { IcoSphere } from '../src/mesh/IcoSphere';

describe('Print Minimal Failing Kernel', () => {
  it('should print kernel code', () => {
    console.log('\n=== MINIMAL FAILING CASE ===\n');

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

    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      paramsToMesh(p);

      const residuals: Value[] = [];
      for (let vertexIdx = 0; vertexIdx < mesh.vertices.length; vertexIdx++) {
        const star = mesh.getVertexStar(vertexIdx);
        if (star.length < 3) {
          residuals.push(V.C(0));
          continue;
        }

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

        residuals.push(V.add(term0, term1));
      }

      return residuals;
    });

    console.log(`Compiled ${compiled.kernelCount} kernels`);
    console.log(`Num functions: ${compiled.numFunctions}\n`);

    // Get first kernel
    const firstDesc = (compiled as any).functionDescriptors[0];
    const kernelPool = (compiled as any).kernelPool;
    const firstKernel = kernelPool.kernels.get(firstDesc.kernelHash);

    console.log('=== KERNEL CODE FOR VERTEX 0 ===\n');
    console.log(firstKernel.kernel.toString());
    console.log('\n');

    const fs = require('fs');
    fs.writeFileSync('C:/Dev/ScalarAutograd/demos/developable-sphere/test/minimal-failing-kernel.js', firstKernel.kernel.toString());
    console.log('Saved to minimal-failing-kernel.js');
  });
});
