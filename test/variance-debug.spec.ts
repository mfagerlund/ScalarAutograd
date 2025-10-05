import { describe, it } from 'vitest';
import { IcoSphere } from '../demos/developable-sphere/src/mesh/IcoSphere';
import { DevelopableEnergy } from '../demos/developable-sphere/src/energy/DevelopableEnergy';
import { V } from '../dist/V';
import { testLog } from './testUtils';

describe('Variance Debug Test', () => {
  it('should debug energy values', () => {
    testLog('\n=== VARIANCE DEBUG ===');

    const sphere = IcoSphere.generate(2, 1.0);
    testLog(`Sphere: ${sphere.vertices.length} vertices`);

    // Check a single vertex
    const vertexIdx = 0;
    const star = sphere.getVertexStar(vertexIdx);
    testLog(`\nVertex ${vertexIdx}: ${star.length} faces in star`);

    // Build covariance matrix manually
    let c00 = V.C(0), c01 = V.C(0), c02 = V.C(0);
    let c11 = V.C(0), c12 = V.C(0), c22 = V.C(0);

    for (const faceIdx of star) {
      const normal = sphere.getFaceNormal(faceIdx);
      const angle = sphere.getInteriorAngle(faceIdx, vertexIdx);

      testLog(`  Face ${faceIdx}: N=(${normal.x.data.toFixed(3)}, ${normal.y.data.toFixed(3)}, ${normal.z.data.toFixed(3)}), θ=${angle.data.toFixed(3)}`);

      c00 = V.add(c00, V.mul(angle, V.mul(normal.x, normal.x)));
      c01 = V.add(c01, V.mul(angle, V.mul(normal.x, normal.y)));
      c02 = V.add(c02, V.mul(angle, V.mul(normal.x, normal.z)));
      c11 = V.add(c11, V.mul(angle, V.mul(normal.y, normal.y)));
      c12 = V.add(c12, V.mul(angle, V.mul(normal.y, normal.z)));
      c22 = V.add(c22, V.mul(angle, V.mul(normal.z, normal.z)));
    }

    testLog(`\nCovariance matrix:`);
    testLog(`  [${c00.data.toFixed(4)}, ${c01.data.toFixed(4)}, ${c02.data.toFixed(4)}]`);
    testLog(`  [${c01.data.toFixed(4)}, ${c11.data.toFixed(4)}, ${c12.data.toFixed(4)}]`);
    testLog(`  [${c02.data.toFixed(4)}, ${c12.data.toFixed(4)}, ${c22.data.toFixed(4)}]`);

    const trace = V.add(V.add(c00, c11), c22);
    testLog(`\nTrace: ${trace.data.toFixed(4)}`);

    const frobSq = V.add(
      V.add(V.add(V.mul(2, V.square(c01)), V.mul(2, V.square(c02))), V.mul(2, V.square(c12))),
      V.add(V.add(V.square(c00), V.square(c11)), V.square(c22))
    );
    const sqrtFrob = V.sqrt(V.add(frobSq, V.C(1e-16)));
    testLog(`Frobenius norm: ${sqrtFrob.data.toFixed(4)}`);

    const lambdaMinProxy = V.sub(trace, sqrtFrob);
    testLog(`λ_min proxy (trace - ||C||_F): ${lambdaMinProxy.data.toFixed(4)}`);

    const energy = V.square(lambdaMinProxy);
    testLog(`Energy (squared): ${energy.data.toExponential(4)}`);

    const actualEnergy = DevelopableEnergy.computeVertexEnergy(vertexIdx, sphere);
    testLog(`Actual energy from function: ${actualEnergy.data.toExponential(4)}`);
  });
});
