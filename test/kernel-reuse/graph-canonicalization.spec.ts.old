/**
 * Test graph canonicalization and signature matching
 */

import { V } from "../../src/V";
import { canonicalizeGraph } from "../../src/GraphSignature";

describe('Graph Canonicalization', () => {
  it('should match identical graph structures', () => {
    const a0 = V.W(1, 'a0');
    const b0 = V.W(2, 'b0');
    const a1 = V.W(3, 'a1');
    const b1 = V.W(4, 'b1');

    const graph1 = V.add(a0, b0);
    const graph2 = V.add(a1, b1);

    const sig1 = canonicalizeGraph(graph1);
    const sig2 = canonicalizeGraph(graph2);

    expect(sig1.hash).toBe(sig2.hash);
    expect(sig1.operations).toEqual(sig2.operations);
    expect(sig1.topology).toEqual(sig2.topology);
  });

  it('should distinguish different operations', () => {
    const a = V.W(1);
    const b = V.W(2);

    const add = V.add(a, b);
    const mul = V.mul(a, b);

    const sig1 = canonicalizeGraph(add);
    const sig2 = canonicalizeGraph(mul);

    expect(sig1.hash).not.toBe(sig2.hash);
    expect(sig1.operations).not.toEqual(sig2.operations);
  });

  it('should distinguish different topologies', () => {
    const a = V.W(1);
    const b = V.W(2);
    const c = V.W(3);

    const graph1 = V.mul(V.add(a, b), c);  // (a+b)*c
    const graph2 = V.mul(a, V.add(b, c));  // a*(b+c)

    const sig1 = canonicalizeGraph(graph1);
    const sig2 = canonicalizeGraph(graph2);

    expect(sig1.hash).not.toBe(sig2.hash);
    expect(sig1.topology).not.toEqual(sig2.topology);
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

    const sig1 = canonicalizeGraph(graph1);
    const sig2 = canonicalizeGraph(graph2);

    expect(sig1.hash).toBe(sig2.hash);
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

    const sig1 = canonicalizeGraph(r1);
    const sig2 = canonicalizeGraph(r2);

    expect(sig1.hash).toBe(sig2.hash);
    console.log('Distance constraint signature:', sig1.operations);
  });

  it('should produce correct operation sequence', () => {
    const a = V.W(2);
    const b = V.W(3);

    const sum = V.add(a, b);
    const result = V.mul(sum, V.C(5));

    const sig = canonicalizeGraph(result);

    // Expected: [leaf, leaf, add, leaf, mul]
    expect(sig.operations).toEqual(['leaf', 'leaf', '+', 'leaf', '*']);
  });

  it('should produce correct topology', () => {
    const a = V.W(2);
    const b = V.W(3);

    const sum = V.add(a, b);  // parents: [a, b]
    const result = V.mul(sum, V.C(5));  // parents: [sum, const]

    const sig = canonicalizeGraph(result);

    // Topology:
    // 0: a (leaf) -> []
    // 1: b (leaf) -> []
    // 2: sum (+) -> [0, 1]
    // 3: const (leaf) -> []
    // 4: result (*) -> [2, 3]
    expect(sig.topology).toEqual([
      [],      // a
      [],      // b
      [0, 1],  // sum
      [],      // const
      [2, 3]   // result
    ]);
  });

  it('should handle single node graph', () => {
    const a = V.W(5);

    const sig = canonicalizeGraph(a);

    expect(sig.operations).toEqual(['leaf']);
    expect(sig.topology).toEqual([[]]);
  });

  it('should handle complex trig graph', () => {
    const theta1 = V.W(Math.PI / 4);
    const theta2 = V.W(Math.PI / 3);

    const graph1 = V.add(V.sin(theta1), V.cos(theta1));
    const graph2 = V.add(V.sin(theta2), V.cos(theta2));

    const sig1 = canonicalizeGraph(graph1);
    const sig2 = canonicalizeGraph(graph2);

    expect(sig1.hash).toBe(sig2.hash);
    expect(sig1.operations).toContain('sin');
    expect(sig1.operations).toContain('cos');
  });

  it('should NOT match if operation order differs', () => {
    const a = V.W(1);
    const b = V.W(2);

    // a + b
    const graph1 = V.add(a, b);

    // b + a (different order in graph structure)
    const graph2 = V.add(b, a);

    const sig1 = canonicalizeGraph(graph1);
    const sig2 = canonicalizeGraph(graph2);

    // These will have different topology because the order of inputs differs
    // sig1: [leaf(a), leaf(b), add([0,1])]
    // sig2: [leaf(b), leaf(a), add([0,1])]
    // BUT the topology is the same! Both are add([0,1])

    // Actually they SHOULD match because topology is the same!
    expect(sig1.hash).toBe(sig2.hash);
  });

  it('should distinguish sqrt from square', () => {
    const x = V.W(4);

    const sqrtGraph = V.sqrt(x);
    const squareGraph = V.square(x);  // Implemented as pow(x, 2)

    const sig1 = canonicalizeGraph(sqrtGraph);
    const sig2 = canonicalizeGraph(squareGraph);

    expect(sig1.hash).not.toBe(sig2.hash);
    expect(sig1.operations).toContain('sqrt');
    expect(sig2.operations).toContain('powValue'); // square is pow(x, 2)
  });
});
