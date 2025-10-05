/**
 * Test graph canonicalization and signature matching
 * New canonical string implementation with flattening and commutative reordering
 */

import { V } from "../../src/V";
import { canonicalizeGraphNoSort } from "../../src/GraphCanonicalizerNoSort";

describe('Graph Canonicalization', () => {
  it('should match identical graph structures', () => {
    const a0 = V.W(1, 'a0');
    const b0 = V.W(2, 'b0');
    const a1 = V.W(3, 'a1');
    const b1 = V.W(4, 'b1');

    const graph1 = V.add(a0, b0);
    const graph2 = V.add(a1, b1);

    const { canon: canon1 } = canonicalizeGraphNoSort(graph1, [a0, b0]);
    const { canon: canon2 } = canonicalizeGraphNoSort(graph2, [a1, b1]);

    expect(canon1).toBe(canon2);
  });

  it('should distinguish different operations', () => {
    const a = V.W(1);
    const b = V.W(2);

    const add = V.add(a, b);
    const mul = V.mul(a, b);

    const { canon: canon1 } = canonicalizeGraphNoSort(add, [a, b]);
    const { canon: canon2 } = canonicalizeGraphNoSort(mul, [a, b]);

    expect(canon1).not.toBe(canon2);
    expect(canon1).toContain('(+');
    expect(canon2).toContain('(*');
  });

  it('should distinguish different topologies', () => {
    const a = V.W(1);
    const b = V.W(2);
    const c = V.W(3);

    const graph1 = V.mul(V.add(a, b), c);  // (a+b)*c
    const graph2 = V.mul(a, V.add(b, c));  // a*(b+c)

    const { canon: canon1 } = canonicalizeGraphNoSort(graph1, [a, b, c]);
    const { canon: canon2 } = canonicalizeGraphNoSort(graph2, [a, b, c]);

    expect(canon1).not.toBe(canon2);
  });

  it('should match same topology with different inputs', () => {
    const a0 = V.W(1);
    const b0 = V.W(2);
    const c0 = V.W(3);
    const a1 = V.W(10);
    const b1 = V.W(20);
    const c1 = V.W(30);

    const graph1 = V.mul(V.add(a0, b0), c0);  // (a0+b0)*c0
    const graph2 = V.mul(V.add(a1, b1), c1);  // (a1+b1)*c1

    const { canon: canon1 } = canonicalizeGraphNoSort(graph1, [a0, b0, c0]);
    const { canon: canon2 } = canonicalizeGraphNoSort(graph2, [a1, b1, c1]);

    expect(canon1).toBe(canon2);
  });

  it('should match distance constraint graphs', () => {
    // Distance 1
    const x1_1 = V.W(0);
    const y1_1 = V.W(0);
    const x2_1 = V.W(3);
    const y2_1 = V.W(4);

    const dx1 = V.sub(x2_1, x1_1);
    const dy1 = V.sub(y2_1, y1_1);
    const distSq1 = V.add(V.mul(dx1, dx1), V.mul(dy1, dy1));
    const dist1 = V.sqrt(distSq1);
    const r1 = V.sub(dist1, V.C(5.0));

    // Distance 2 (different values, same structure)
    const x1_2 = V.W(1);
    const y1_2 = V.W(2);
    const x2_2 = V.W(5);
    const y2_2 = V.W(6);

    const dx2 = V.sub(x2_2, x1_2);
    const dy2 = V.sub(y2_2, y1_2);
    const distSq2 = V.add(V.mul(dx2, dx2), V.mul(dy2, dy2));
    const dist2 = V.sqrt(distSq2);
    const r2 = V.sub(dist2, V.C(10.0)); // Different target, but same structure

    const { canon: canon1 } = canonicalizeGraphNoSort(r1, [x1_1, y1_1, x2_1, y2_1]);
    const { canon: canon2 } = canonicalizeGraphNoSort(r2, [x1_2, y1_2, x2_2, y2_2]);

    expect(canon1).toBe(canon2);
    console.log('Distance constraint signature:', canon1);
  });

  it('should match commutative reorderings: a+b == b+a', () => {
    const a = V.W(1);
    const b = V.W(2);

    const graph1 = V.add(a, b);
    const graph2 = V.add(b, a);  // Commutative reordering

    const { canon: canon1 } = canonicalizeGraphNoSort(graph1, [a, b]);
    const { canon: canon2 } = canonicalizeGraphNoSort(graph2, [a, b]);

    expect(canon1).toBe(canon2);
  });

  it('should flatten nested additions: (a+b)+c == a+b+c', () => {
    const a = V.W(1);
    const b = V.W(2);
    const c = V.W(3);

    const graph1 = V.add(V.add(a, b), c);  // (a+b)+c
    const graph2 = V.add(a, V.add(b, c));  // a+(b+c)

    const { canon: canon1 } = canonicalizeGraphNoSort(graph1, [a, b, c]);
    const { canon: canon2 } = canonicalizeGraphNoSort(graph2, [a, b, c]);

    expect(canon1).toBe(canon2);
    expect(canon1).toContain('(+,0g,1g,2g)');  // Flattened
  });

  it('should match cos(a)+sin(b) == sin(b)+cos(a)', () => {
    const a = V.W(1);
    const b = V.W(2);

    const graph1 = V.add(V.cos(a), V.sin(b));
    const graph2 = V.add(V.sin(b), V.cos(a));

    const { canon: canon1 } = canonicalizeGraphNoSort(graph1, [a, b]);
    const { canon: canon2 } = canonicalizeGraphNoSort(graph2, [a, b]);

    expect(canon1).toBe(canon2);
    expect(canon1).toContain('(cos');
    expect(canon1).toContain('(sin');
  });

  it('should handle single node graph', () => {
    const a = V.W(5);

    const { canon } = canonicalizeGraphNoSort(a, [a]);

    expect(canon).toBe('0g|0g');
  });

  it('should distinguish gradient requirements', () => {
    const x = V.W(5);
    const y = V.W(3);
    const c = V.C(3);

    const graph1 = V.add(V.square(x), V.square(y));  // Both need grads
    const graph2 = V.add(V.square(x), V.square(c));  // Only x needs grad

    const { canon: canon1 } = canonicalizeGraphNoSort(graph1, [x, y]);
    const { canon: canon2 } = canonicalizeGraphNoSort(graph2, [x, c]);

    expect(canon1).not.toBe(canon2);
    expect(canon1).toContain('0g,1g');  // Both grad
    expect(canon2).toContain('0g,1|');  // Only first grad
  });

  it('should normalize square: pow(x,2) -> square(x)', () => {
    const x = V.W(4);

    const squareGraph = V.square(x);  // Implemented as pow(x, 2)

    const { canon } = canonicalizeGraphNoSort(squareGraph, [x]);

    expect(canon).toContain('(square,0g)');  // Normalized
  });

  it('should NOT normalize pow with non-constant exponent', () => {
    const x = V.W(2);
    const n = V.W(3);  // Variable exponent

    const powGraph = V.powValue(x, n);

    const { canon } = canonicalizeGraphNoSort(powGraph, [x, n]);

    expect(canon).toContain('powValue');  // Not normalized
  });

  it('should match graphs with different constant values (same structure)', () => {
    const x1 = V.W(1);
    const y1 = V.W(2);
    const x2 = V.W(10);
    const y2 = V.W(20);

    // Same structure: sqrt((x2-x1)^2 + (y2-y1)^2) - target
    const dx1 = V.sub(x2, x1);
    const dy1 = V.sub(y2, y1);
    const dist1 = V.sqrt(V.add(V.square(dx1), V.square(dy1)));
    const r1 = V.sub(dist1, V.C(5.0));  // Target = 5.0

    const x3 = V.W(100);
    const y3 = V.W(200);
    const x4 = V.W(300);
    const y4 = V.W(400);

    const dx2 = V.sub(x4, x3);
    const dy2 = V.sub(y4, y3);
    const dist2 = V.sqrt(V.add(V.square(dx2), V.square(dy2)));
    const r2 = V.sub(dist2, V.C(10.0));  // Target = 10.0 (different!)

    const { canon: canon1 } = canonicalizeGraphNoSort(r1, [x1, y1, x2, y2]);
    const { canon: canon2 } = canonicalizeGraphNoSort(r2, [x3, y3, x4, y4]);

    expect(canon1).toBe(canon2);  // Should match despite different constant values
    // Constants are just numbered leaves like params, no special treatment needed
  });

  it('should distinguish genuinely different structures', () => {
    const x1 = V.W(1);
    const y1 = V.W(2);

    // Structure 1: x^2 + y^2 - constant
    const graph1 = V.sub(V.add(V.square(x1), V.square(y1)), V.C(5.0));

    const x2 = V.W(10);
    const y2 = V.W(20);

    // Structure 2: x^2 * y^2 - constant (different operation!)
    const graph2 = V.sub(V.mul(V.square(x2), V.square(y2)), V.C(10.0));

    const { canon: canon1 } = canonicalizeGraphNoSort(graph1, [x1, y1]);
    const { canon: canon2 } = canonicalizeGraphNoSort(graph2, [x2, y2]);

    expect(canon1).not.toBe(canon2);  // Different structures
    expect(canon1).toContain('(+');   // Addition
    expect(canon2).toContain('(*');   // Multiplication
  });

  it('should distinguish different topologies even with same operations', () => {
    const a1 = V.W(1);
    const b1 = V.W(2);

    // Topology 1: (a - const) + (b - const)
    const graph1 = V.add(V.sub(a1, V.C(5.0)), V.sub(b1, V.C(10.0)));

    const a2 = V.W(10);
    const b2 = V.W(20);

    // Topology 2: (a + b) - const
    const graph2 = V.sub(V.add(a2, b2), V.C(15.0));

    const { canon: canon1 } = canonicalizeGraphNoSort(graph1, [a1, b1]);
    const { canon: canon2 } = canonicalizeGraphNoSort(graph2, [a2, b2]);

    expect(canon1).not.toBe(canon2);  // Different topologies
  });

  it('should match multiple constraints with different constants', () => {
    // Create 3 distance constraints with different target values
    const constraints = [
      { x1: V.W(0), y1: V.W(0), x2: V.W(1), y2: V.W(1), target: 5.0 },
      { x1: V.W(2), y1: V.W(3), x2: V.W(4), y2: V.W(5), target: 10.0 },
      { x1: V.W(6), y1: V.W(7), x2: V.W(8), y2: V.W(9), target: 15.0 }
    ];

    const canons = constraints.map(c => {
      const dx = V.sub(c.x2, c.x1);
      const dy = V.sub(c.y2, c.y1);
      const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
      const residual = V.sub(dist, V.C(c.target));
      const { canon } = canonicalizeGraphNoSort(residual, [c.x1, c.y1, c.x2, c.y2]);
      return canon;
    });

    // All should have identical canonical strings
    expect(canons[0]).toBe(canons[1]);
    expect(canons[1]).toBe(canons[2]);
    console.log('Distance constraint signature:', canons[0]);
  });
});
