import { Value } from "../src/Value";

function numericalGrad(f: (x: number) => number, x0: number, eps = 1e-6): number {
  return (f(x0 + eps) - f(x0 - eps)) / (2 * eps);
}

function testUnaryGrad(opName: string, op: (x: Value) => Value, dOp: (x: number) => number, xval: number) {
  const x = new Value(xval, "x", true);
  const y = op(x);
  y.backward();
  const analytic = x.grad;
  const numeric = numericalGrad(z => op(new Value(z, "x", false)).data, xval);
  expect(analytic).toBeCloseTo(numeric, 4);
}

function testBinaryGrad(opName: string, op: (a: Value, b: Value) => Value, dOpA: (a: number, b: number) => number, dOpB: (a: number, b: number) => number, aval: number, bval: number) {
  const a = new Value(aval, "a", true);
  const b = new Value(bval, "b", true);
  const c = op(a, b);
  c.backward();
  const analyticA = a.grad;
  const analyticB = b.grad;
  const numericA = numericalGrad(x => op(new Value(x, "a", false), new Value(bval, "b", false)).data, aval);
  const numericB = numericalGrad(x => op(new Value(aval, "a", false), new Value(x, "b", false)).data, bval);
  expect(analyticA).toBeCloseTo(numericA, 4);
  expect(analyticB).toBeCloseTo(numericB, 4);
}
describe('Value autograd system', () => {
  it('runs the forward and backward pass example', () => {
    const a = new Value(2, 'a', true);
    const b = new Value(-3, 'b', true);
    const c = new Value(10, 'c', true);
    const e = a.mul(b);       // e = a * b
    const d = e.add(c);       // d = e + c
    const f = d.tanh();       // f = tanh(d)

    f.backward();

    expect(Number(a.data)).toBe(2);
    expect(Number(b.data)).toBe(-3);
    expect(Number(c.data)).toBe(10);
    expect(f.toString()).toMatch(/Value\(data=.*?, grad=.*?, label=tanh\(\(.+\)\)\)/);
    expect(Number.isFinite(a.grad)).toBe(true);
    expect(Number.isFinite(b.grad)).toBe(true);
    expect(Number.isFinite(c.grad)).toBe(true);
});



describe('Value new operators: powValue, mod, cmp, softplus, floor/ceil/round, square/cube, reciprocal, clamp, sum, mean', () => {
  it('powValue matches number math and gradients', () => {
    const a = new Value(2, 'a', true);
    const b = new Value(3, 'b', true);
    const c = a.powValue(b);
    c.backward();
    // da = b * a^(b-1); db = log(a) * a^b
    expect(c.data).toBeCloseTo(8);
    expect(a.grad).toBeCloseTo(3 * (2 ** 2));
    expect(b.grad).toBeCloseTo(Math.log(2) * 8);
  });

  it('mod computes values modulo', () => {
    const a = new Value(7);
    const b = new Value(3);
    expect(a.mod(b).data).toBeCloseTo(1);
  });

  it('cmp functions eq/neq/gt/lt/gte/lte match JS', () => {
    const a = new Value(5);
    const b = new Value(7);
    expect(a.eq(b).data).toBe(0);
    expect(b.eq(b).data).toBe(1);
    expect(a.neq(b).data).toBe(1);
    expect(b.neq(b).data).toBe(0);
    expect(a.gt(b).data).toBe(0);
    expect(b.gt(a).data).toBe(1);
    expect(a.lt(b).data).toBe(1);
    expect(b.lt(a).data).toBe(0);
    expect(a.gte(b).data).toBe(0);
    expect(b.gte(a).data).toBe(1);
    expect(a.lte(b).data).toBe(1);
    expect(b.lte(a).data).toBe(0);
  });

  it('softplus and its gradient', () => {
    const x = new Value(0.5, 'x', true);
    const y = x.softplus();
    y.backward();
    expect(y.data).toBeCloseTo(Math.log(1 + Math.exp(0.5)), 5);
    expect(x.grad).toBeCloseTo(1 / (1 + Math.exp(-0.5)), 5);
  });

  it('floor, ceil and round logic', () => {
    const x = new Value(-2.7);
    expect(x.floor().data).toBe(-3);
    expect(x.ceil().data).toBe(-2);
    expect(new Value(1.4).round().data).toBe(1);
    expect(new Value(1.6).round().data).toBe(2);
  });

  it('square, cube, reciprocal logic', () => {
    const x = new Value(3, 'x', true);
    const sq = x.square();
    const cu = x.cube();
    const rec = x.reciprocal();
    sq.backward();
    expect(sq.data).toBe(9);
    expect(x.grad).toBe(6);
    x.grad = 0;
    cu.backward();
    expect(cu.data).toBe(27);
    expect(x.grad).toBe(27);
    x.grad = 0;
    rec.backward();
    expect(rec.data).toBeCloseTo(1/3);
    expect(x.grad).toBeCloseTo(-1/9);
  });

  it('clamp clamps value and only has gradient when in interior', () => {
    const x = new Value(5, 'x', true);
    const c1 = x.clamp(0, 3);
    expect(c1.data).toBe(3);
    c1.backward();
    expect(x.grad).toBe(0);
    x.grad = 0;
    const c2 = x.clamp(0, 10);
    expect(c2.data).toBe(5);
    c2.backward();
    expect(x.grad).toBe(1);
    x.grad = 0;
    const c3 = x.clamp(7, 9);
    expect(c3.data).toBe(7);
    c3.backward();
    expect(x.grad).toBe(0);
  });

  it('sum and mean logic for array inputs', () => {
    const vals = [1, 3, 5].map((n, i) => new Value(n, 'v'+i, true));
    const s = Value.sum(vals);
    const m = Value.mean(vals);
    expect(s.data).toBe(9);
    expect(m.data).toBe(3);
    m.backward();
    for (const v of vals) expect(v.grad).toBeCloseTo(1/3);
  });
});

  it('computes gradients only for required nodes (example from user)', () => {
    const x = new Value(2.0, "x", true);
    const y = new Value(3.0, "y", false); // y doesn't require gradients
    const z = x.mul(y).add(x.pow(2));
    z.backward();
    expect(Number(x.grad)).toBeCloseTo(7.0);
    expect(Number(y.grad)).toBeCloseTo(0.0);
    expect(x.toString()).toMatch(/Value\(data=2.0000, grad=7.0000, label=x\)/);
    expect(y.toString()).toMatch(/Value\(data=3.0000, grad=0.0000, label=y\)/);
  });

  it('computes gradients for add operation', () => {
    const a = new Value(1.5, 'a', true);
    const b = new Value(-0.7, 'b', true);
    const c = a.add(b);
    c.backward();
    // dc/da = 1, dc/db = 1
    expect(a.grad).toBeCloseTo(1.0);
    expect(b.grad).toBeCloseTo(1.0);
  });

  it('computes gradients for mul operation', () => {
    const a = new Value(2, 'a', true);
    const b = new Value(3, 'b', true);
    const c = a.mul(b);
    c.backward();
    // dc/da = b, dc/db = a
    expect(a.grad).toBeCloseTo(3.0);
    expect(b.grad).toBeCloseTo(2.0);
  });

  it('computes gradients for sub operation', () => {
    const a = new Value(2.5, 'a', true);
    const b = new Value(1.2, 'b', true);
    const c = a.sub(b);
    c.backward();
    // dc/da = 1, dc/db = -1
    expect(a.grad).toBeCloseTo(1.0);
    expect(b.grad).toBeCloseTo(-1.0);
  });

  it('computes gradients for div operation', () => {
    const a = new Value(6, 'a', true);
    const b = new Value(2, 'b', true);
    const c = a.div(b);
    c.backward();
    // dc/da = 1/b, dc/db = -a/b^2
    expect(a.grad).toBeCloseTo(0.5);
    expect(b.grad).toBeCloseTo(-1.5);
  });

  it('computes gradients for pow operation', () => {
    const a = new Value(4, 'a', true);
    const c = a.pow(3);
    c.backward();
    // dc/da = 3*a^2 = 48
    expect(a.grad).toBeCloseTo(48.0);
  });

  it('computes gradients for tanh operation', () => {
    const a = new Value(1, 'a', true);
    const c = a.tanh();
    c.backward();
    // dc/da = 1-tanh(a)^2
    const t = Math.tanh(1);
    expect(a.grad).toBeCloseTo(1 - t*t);
  });

  it('computes gradients for sigmoid operation', () => {
    const a = new Value(0.7, 'a', true);
    const c = a.sigmoid();
    c.backward();
    // dc/da = sigmoid(a)*(1-sigmoid(a))
    const s = 1/(1+Math.exp(-0.7));
    expect(a.grad).toBeCloseTo(s*(1-s));
  });

  it('does not track graph when using Value.withNoGrad', () => {
    const a = new Value(5, 'a', true);
    const b = new Value(7, 'b', true);
    let c: Value | undefined = undefined;
    Value.withNoGrad(() => {
      c = a.add(b);
    });
    expect(c).toBeDefined();
    expect(c!.requiresGrad).toBe(false);
    expect(c!['prev'].length).toBe(0);
    c!.backward();
    expect(a.grad).toBe(0);
    expect(b.grad).toBe(0);
  });
});

describe('Value unary and binary operators: trigs, relu, abs, exp/log, min/max', () => {
  // Numerical vs analytic gradient checks for unary operators
  it('numerical gradient: sin',    () => testUnaryGrad('sin',    x=>x.sin(),       x=>Math.cos(x),                      1.1));
  it('numerical gradient: cos',    () => testUnaryGrad('cos',    x=>x.cos(),       x=>-Math.sin(x),                     0.5));
  it('numerical gradient: tan',    () => testUnaryGrad('tan',    x=>x.tan(),       x=>1/(Math.cos(x)**2),               0.8));
  it('numerical gradient: asin',   () => testUnaryGrad('asin',   x=>x.asin(),      x=>1/Math.sqrt(1-x*x),               0.25));
  it('numerical gradient: acos',   () => testUnaryGrad('acos',   x=>x.acos(),      x=>-1/Math.sqrt(1-x*x),              0.25));
  it('numerical gradient: atan',   () => testUnaryGrad('atan',   x=>x.atan(),      x=>1/(1+x*x),                        1.3));
  it('numerical gradient: relu',   () => testUnaryGrad('relu',   x=>x.relu(),      x=>(x>0?1:0),                        3.0));
  it('numerical gradient: abs',    () => testUnaryGrad('abs',    x=>x.abs(),       x=>(x >= 0 ? 1 : -1),                -3));
  it('numerical gradient: exp',    () => testUnaryGrad('exp',    x=>x.exp(),       x=>Math.exp(x),                      1.2));
  it('numerical gradient: log',    () => testUnaryGrad('log',    x=>x.log(),       x=>1/x,                              1.5));
  it('numerical gradient: tanh',   () => testUnaryGrad('tanh',   x=>x.tanh(),      x=>1-Math.tanh(x)**2,                0.9));
  it('numerical gradient: sigmoid',() => testUnaryGrad('sigmoid',x=>x.sigmoid(),   x=>{const s=1/(1+Math.exp(-x));return s*(1-s);},0.7));
  it('numerical gradient: sign',    () => testUnaryGrad('sign',    x=>x.sign(),      x=>0,  2.0));
  it('numerical gradient: sign negative', () => testUnaryGrad('sign', x=>x.sign(), x=>0, -2.0));
  it('gradient of sign(0) is 0', () => {
    const x = new Value(0.0, "x", true);
    const y = x.sign();
    expect(y.data).toBe(0); // sign(0) should be 0
    y.backward();
    expect(x.grad).toBe(0); // Analytical gradient for sign(0) is implemented as 0
  });
  
  // Numerical vs analytic gradient checks for binary operators
  it('numerical gradient: add', () => testBinaryGrad('add', (a,b)=>a.add(b), (a,b)=>1, (a,b)=>1, 1.3, -2.1));
  it('numerical gradient: sub', () => testBinaryGrad('sub', (a,b)=>a.sub(b), (a,b)=>1, (a,b)=>-1, 5.2, -1.2));
  it('numerical gradient: mul', () => testBinaryGrad('mul', (a,b)=>a.mul(b), (a,b)=>b, (a,b)=>a, 1.7, 2.5));
  it('numerical gradient: div', () => testBinaryGrad('div', (a,b)=>a.div(b), (a,b)=>1/b, (a,b)=>-a/(b*b), 4.0, -2.2));
  it('numerical gradient: pow', () => {
    const exp = 3.3;
    const grad = (a:number) => exp*Math.pow(a, exp-1);
    testUnaryGrad('pow', x=>x.pow(exp), grad, 2.0);
  });
  it('numerical gradient: min', () => testBinaryGrad('min', (a,b)=>a.min(b), (a,b)=>a<b?1:0, (a,b)=>b<a?1:0, -1.0, 0.8));
  it('numerical gradient: max', () => testBinaryGrad('max', (a,b)=>a.max(b), (a,b)=>a>b?1:0, (a,b)=>b>a?1:0, 2.3, -4.5));
});

describe('Gradient flow control', () => {
  it('stops gradient at non-requiresGrad nodes', () => {
    const x = new Value(2, 'x', true);
    const y = new Value(3, 'y', false);
    const z = new Value(4, 'z', true);
    const out = x.mul(y).add(z);
    out.backward();
    expect(x.grad).toBe(3);
    expect(y.grad).toBe(0);
    expect(z.grad).toBe(1);
  });

  it('handles detached computation graphs', () => {
    const x = new Value(2, 'x', true);
    const y = x.mul(3);
    const z = new Value(y.data, 'z', true);
    const out = z.mul(4);
    out.backward();
    expect(z.grad).toBe(4);
    expect(x.grad).toBe(0);
  });
});

describe('Memory management', () => {
  it('handles large computation graphs', () => {
    let x = new Value(1, 'x', true);
    for (let i = 0; i < 100; i++) {
      x = x.add(1).mul(1.01);
    }
    expect(() => x.backward()).not.toThrow();
  });

  it('zeroGradAll handles multiple disconnected graphs', () => {
    const x1 = new Value(1, 'x1', true);
    const y1 = x1.mul(2);
    const x2 = new Value(2, 'x2', true);
    const y2 = x2.mul(3);

    y1.backward();
    y2.backward();

    Value.zeroGradAll([y1, y2]);
    expect(x1.grad).toBe(0);
    expect(x2.grad).toBe(0);
  });
});