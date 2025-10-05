/**
 * Print the generated compiled code to inspect it.
 */

import { V } from "../src/V";
import { Value } from "../src/Value";
import { Vec3 } from "../src/Vec3";
import { ValueRegistry } from "../src/ValueRegistry";
import { compileIndirectKernel } from "../src/compileIndirectKernel";
import { testLog } from './testUtils';

/**
 * Helper to print the compiled code for an expression
 */
function printCompiledCode(
  buildExpr: (params: Value[]) => Value,
  paramValues: number[],
  testName: string
) {
  testLog(`\n${'='.repeat(80)}`);
  testLog(`${testName}`);
  testLog('='.repeat(80));

  const params = paramValues.map((val, i) => {
    const p = V.W(val, `p${i}`);
    p.paramName = `p${i}`;
    return p;
  });

  const expr = buildExpr(params);

  const registry = new ValueRegistry();
  params.forEach(p => registry.register(p));

  // Get the compiled kernel
  const kernel = compileIndirectKernel(expr, params, registry);

  testLog('\nCompiled function:');
  testLog(kernel.toString());
  testLog('');
}

describe('Print Compiled Code', () => {
  it('simple add', () => {
    printCompiledCode(
      ([a, b]) => V.add(a, b),
      [3.0, 5.0],
      'SIMPLE: a + b'
    );
  });

  it('cross product', () => {
    printCompiledCode(
      ([ax, ay, az, bx, by, bz]) => {
        const a = new Vec3(V.W(ax.data), V.W(ay.data), V.W(az.data));
        const b = new Vec3(V.W(bx.data), V.W(by.data), V.W(bz.data));
        const cross = Vec3.cross(a, b);
        return Vec3.dot(cross, cross);
      },
      [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      'CROSS PRODUCT: |a × b|²'
    );
  });

  it('plane alignment vertex energy', () => {
    printCompiledCode(
      ([n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z]) => {
        const normals = [
          new Vec3(V.W(n1x.data), V.W(n1y.data), V.W(n1z.data)),
          new Vec3(V.W(n2x.data), V.W(n2y.data), V.W(n2z.data)),
          new Vec3(V.W(n3x.data), V.W(n3y.data), V.W(n3z.data)),
        ];

        // Differentiable plane selection
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

        // Compute energy
        let energy = V.C(0);
        for (let i = 0; i < normals.length; i++) {
          const dist = V.abs(Vec3.dot(normals[i], planeNormalNormalized));
          energy = V.add(energy, V.mul(dist, dist));
        }

        return V.div(energy, normals.length);
      },
      [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
      'PLANE ALIGNMENT: Vertex Energy'
    );
  });
});
