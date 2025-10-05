import { describe, it, expect } from 'vitest';
import { V } from '../src/Value';
import { canonicalizeGraphHash } from '../src/GraphHashCanonicalizer';
import { testLog } from './testUtils';

describe('GraphHashCanonicalizer', () => {
  it('should produce different hashes for different operand orders (position-dependent)', () => {
    const x = V.W(1);
    const y = V.W(2);
    const z = V.W(3);

    // Position-dependent: different orders give different hashes
    const sum1 = V.add(V.add(x, y), z);
    const sum2 = V.add(V.add(z, y), x);
    const sum3 = V.add(z, V.add(x, y));

    const canon1 = canonicalizeGraphHash(sum1, [x, y, z]);
    const canon2 = canonicalizeGraphHash(sum2, [x, y, z]);
    const canon3 = canonicalizeGraphHash(sum3, [x, y, z]);

    testLog('Add variants:', {
      canon1: canon1.canon,
      canon2: canon2.canon,
      canon3: canon3.canon,
    });

    // Different orders produce different hashes (no sorting)
    expect(canon1.canon).not.toBe(canon2.canon);
    expect(canon1.canon).not.toBe(canon3.canon);
  });

  it('should produce different hashes for mul with different orders (position-dependent)', () => {
    const x = V.W(1);
    const y = V.W(2);
    const z = V.W(3);

    const prod1 = V.mul(V.mul(x, y), z);
    const prod2 = V.mul(z, V.mul(y, x));
    const prod3 = V.mul(V.mul(z, x), y);

    const canon1 = canonicalizeGraphHash(prod1, [x, y, z]);
    const canon2 = canonicalizeGraphHash(prod2, [x, y, z]);
    const canon3 = canonicalizeGraphHash(prod3, [x, y, z]);

    testLog('Mul variants:', {
      canon1: canon1.canon,
      canon2: canon2.canon,
      canon3: canon3.canon,
    });

    // Different orders produce different hashes (no sorting)
    expect(canon1.canon).not.toBe(canon2.canon);
    expect(canon1.canon).not.toBe(canon3.canon);
  });

  it('should produce different hashes for non-commutative operations', () => {
    const x = V.W(10);
    const y = V.W(5);

    const div1 = V.div(x, y); // 10 / 5
    const div2 = V.div(y, x); // 5 / 10

    const canon1 = canonicalizeGraphHash(div1, [x, y]);
    const canon2 = canonicalizeGraphHash(div2, [x, y]);

    testLog('Div variants:', {
      canon1: canon1.canon,
      canon2: canon2.canon,
    });

    expect(canon1.canon).not.toBe(canon2.canon);
  });

  it('should produce different hashes for sub operations', () => {
    const x = V.W(10);
    const y = V.W(5);

    const sub1 = V.sub(x, y); // 10 - 5
    const sub2 = V.sub(y, x); // 5 - 10

    const canon1 = canonicalizeGraphHash(sub1, [x, y]);
    const canon2 = canonicalizeGraphHash(sub2, [x, y]);

    testLog('Sub variants:', {
      canon1: canon1.canon,
      canon2: canon2.canon,
    });

    expect(canon1.canon).not.toBe(canon2.canon);
  });

  it('should handle mixed operations with position-dependent hashing', () => {
    const a = V.W(1);
    const b = V.W(2);
    const c = V.W(3);

    // (a*b) + (c/a)
    const expr1 = V.add(V.mul(a, b), V.div(c, a));
    // (b*a) + (c/a) - different (mul order matters now)
    const expr2 = V.add(V.mul(b, a), V.div(c, a));
    // (c/a) + (a*b) - different (add order matters now)
    const expr3 = V.add(V.div(c, a), V.mul(a, b));
    // (a*b) + (a/c) - different (div order matters)
    const expr4 = V.add(V.mul(a, b), V.div(a, c));

    const canon1 = canonicalizeGraphHash(expr1, [a, b, c]);
    const canon2 = canonicalizeGraphHash(expr2, [a, b, c]);
    const canon3 = canonicalizeGraphHash(expr3, [a, b, c]);
    const canon4 = canonicalizeGraphHash(expr4, [a, b, c]);

    testLog('Mixed ops:', {
      canon1: canon1.canon,
      canon2: canon2.canon,
      canon3: canon3.canon,
      canon4: canon4.canon,
    });

    // All different due to position-dependent hashing
    expect(canon1.canon).not.toBe(canon2.canon);
    expect(canon1.canon).not.toBe(canon3.canon);
    expect(canon1.canon).not.toBe(canon4.canon);
  });

  it('should handle gradient flags correctly', () => {
    const x = V.W(1); // requiresGrad = true
    const y = V.C(2); // requiresGrad = false

    const expr1 = V.add(x, y);
    const canon1 = canonicalizeGraphHash(expr1, [x, y]);

    // If we swap which one has grad...
    const a = V.C(1); // requiresGrad = false
    const b = V.W(2); // requiresGrad = true

    const expr2 = V.add(a, b);
    const canon2 = canonicalizeGraphHash(expr2, [a, b]);

    testLog('Gradient flags:', {
      canon1: canon1.canon,
      canon2: canon2.canon,
    });

    // Should be different because gradient requirements differ
    expect(canon1.canon).not.toBe(canon2.canon);
  });

  it('should produce same hash as operations are reused', () => {
    const x = V.W(1);
    const y = V.W(2);

    // Shared subexpression: x*y appears twice
    const xy = V.mul(x, y);
    const expr = V.add(xy, xy);

    const canon = canonicalizeGraphHash(expr, [x, y]);
    testLog('Shared subexpression canon:', canon.canon);

    // Should still work correctly
    expect(canon.canon).toBeTruthy();
    expect(canon.canon).toMatch(/^0g,1g\|[0-9a-f]{16}$/);
  });

  it('should handle debug mode', () => {
    const x = V.W(1);
    const y = V.W(2);

    const expr = V.add(V.mul(x, y), V.square(x));
    const canon = canonicalizeGraphHash(expr, [x, y], true);

    testLog('Debug mode:', {
      canon: canon.canon,
      debugExpr: canon.debugExpr,
    });

    expect(canon.debugExpr).toBeTruthy();
    expect(canon.debugExpr).toContain('+');
    expect(canon.debugExpr).toContain('*');
  });
});
