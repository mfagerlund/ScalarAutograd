/**
 * Debug canonical strings for distance constraints
 */

import { V } from "../src/V";
import { CompiledFunctions } from "../src/CompiledFunctions";
import { testLog } from "./testUtils";

describe('Debug Distance Canonical Strings', () => {
  it('should show canonical strings for distance constraints', () => {
    const x1 = V.W(0.0, 'x1');
    const y1 = V.W(0.0, 'y1');
    const x2 = V.W(3.0, 'x2');
    const y2 = V.W(4.0, 'y2');
    const x3 = V.W(6.0, 'x3');
    const y3 = V.W(0.0, 'y3');

    const params = [x1, y1, x2, y2, x3, y3];

    const compiled = CompiledFunctions.compile(params, (p) => {
      const distanceSquared = (i1: number, i2: number, target: number) => {
        const dx = V.sub(p[i1], p[i2]);
        const dy = V.sub(p[i1 + 1], p[i2 + 1]);
        const distSq = V.add(V.square(dx), V.square(dy));
        return V.sub(distSq, V.C(target * target));
      };

      return [
        distanceSquared(0, 2, 5), // d(p1,p2) - 5
        distanceSquared(2, 4, 6), // d(p2,p3) - 6
        distanceSquared(4, 0, 6)  // d(p3,p1) - 6
      ];
    });

    testLog('\nKernel count:', compiled.kernelCount);
    testLog('Function count:', compiled.numFunctions);
    testLog('\nCanonical strings:');

    // @ts-ignore - accessing internal kernelPool
    const canonicals = compiled.kernelPool.getCanonicalStrings();
    canonicals.forEach((canon, i) => {
      testLog(`  Kernel ${i}: ${canon}`);
    });
  });
});
