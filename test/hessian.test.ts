import { Value } from './Value';

// Extended Value that tracks second derivatives
class HessianValue extends Value {
  // For each parameter, store ∂²f/∂xi∂xj
  hessian: Map<string, number> = new Map();
  paramName?: string;

  constructor(data: number, paramName?: string) {
    super(data, '', true);
    this.paramName = paramName;
  }

  // Helper to get/set Hessian entries
  getH(xi: string, xj: string): number {
    const key = xi <= xj ? `${xi},${xj}` : `${xj},${xi}`;
    return this.hessian.get(key) || 0;
  }

  setH(xi: string, xj: string, value: number) {
    const key = xi <= xj ? `${xi},${xj}` : `${xj},${xi}`;
    this.hessian.set(key, value);
  }
}

// Type for a function that rebuilds the computation graph
type GraphBuilder = (values: number[]) => HessianValue;

function computeHessian(
  buildGraph: GraphBuilder,
  params: HessianValue[]
): number[][] {
  const n = params.length;
  const H: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));
  const eps = 1e-5;

  const values = params.map(p => p.data);

  // For each pair of parameters, compute ∂²f/∂xi∂xj
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {  // Only upper triangle (symmetric)
      if (i === j) {
        // Diagonal: ∂²f/∂xi² using central difference
        const v = [...values];

        v[i] = values[i] + eps;
        const fPlus = buildGraph(v);

        v[i] = values[i];
        const fCenter = buildGraph(v);

        v[i] = values[i] - eps;
        const fMinus = buildGraph(v);

        // f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
        H[i][i] = (fPlus.data - 2 * fCenter.data + fMinus.data) / (eps * eps);
      } else {
        // Off-diagonal: ∂²f/∂xi∂xj using mixed partial formula
        const v = [...values];

        // f(xi+eps, xj+eps)
        v[i] = values[i] + eps;
        v[j] = values[j] + eps;
        const f1 = buildGraph(v);

        // f(xi+eps, xj-eps)
        v[j] = values[j] - eps;
        const f2 = buildGraph(v);

        // f(xi-eps, xj+eps)
        v[i] = values[i] - eps;
        v[j] = values[j] + eps;
        const f3 = buildGraph(v);

        // f(xi-eps, xj-eps)
        v[j] = values[j] - eps;
        const f4 = buildGraph(v);

        // ∂²f/∂xi∂xj ≈ [f(xi+h,xj+h) - f(xi+h,xj-h) - f(xi-h,xj+h) + f(xi-h,xj-h)] / 4h²
        H[i][j] = (f1.data - f2.data - f3.data + f4.data) / (4 * eps * eps);
        H[j][i] = H[i][j];  // Symmetric
      }
    }
  }

  return H;
}

// Analytical Hessian computation (more accurate!)
function analyticHessian(output: Value, params: Value[]): number[][] {
  const n = params.length;
  const H: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));

  // This requires implementing forward-over-backward mode
  // For now, let's show the concept with specific operations

  return H;
}

describe('Hessian Computation', () => {
  it('computes Hessian of x² + y²', () => {
    const x = new HessianValue(2.0, 'x');
    const y = new HessianValue(3.0, 'y');

    const buildGraph = (v: number[]) => {
      const px = new HessianValue(v[0], 'x');
      const py = new HessianValue(v[1], 'y');
      return px.mul(px).add(py.mul(py)) as HessianValue;
    };

    const H = computeHessian(buildGraph, [x, y]);

    // Expected Hessian:
    // ∂²f/∂x² = 2,  ∂²f/∂y² = 2,  ∂²f/∂x∂y = 0
    console.log('Hessian of x² + y² at (2, 3):');
    console.log(H);

    expect(Math.abs(H[0][0] - 2)).toBeLessThan(0.01);  // ∂²f/∂x² = 2
    expect(Math.abs(H[1][1] - 2)).toBeLessThan(0.01);  // ∂²f/∂y² = 2
    expect(Math.abs(H[0][1] - 0)).toBeLessThan(0.01);  // ∂²f/∂x∂y = 0
    expect(Math.abs(H[1][0] - 0)).toBeLessThan(0.01);  // Symmetric
  });

  it('computes Hessian of x*y', () => {
    const x = new HessianValue(2.0, 'x');
    const y = new HessianValue(3.0, 'y');

    const buildGraph = (v: number[]) => {
      const px = new HessianValue(v[0], 'x');
      const py = new HessianValue(v[1], 'y');
      return px.mul(py) as HessianValue;
    };

    const H = computeHessian(buildGraph, [x, y]);

    // Expected Hessian:
    // ∂²f/∂x² = 0,  ∂²f/∂y² = 0,  ∂²f/∂x∂y = 1
    console.log('\nHessian of x*y at (2, 3):');
    console.log(H);

    expect(Math.abs(H[0][0] - 0)).toBeLessThan(0.01);  // ∂²f/∂x² = 0
    expect(Math.abs(H[1][1] - 0)).toBeLessThan(0.01);  // ∂²f/∂y² = 0
    expect(Math.abs(H[0][1] - 1)).toBeLessThan(0.01);  // ∂²f/∂x∂y = 1
    expect(Math.abs(H[1][0] - 1)).toBeLessThan(0.01);  // Symmetric
  });

  it('computes Hessian of a*b²', () => {
    const a = new HessianValue(2.0, 'a');
    const b = new HessianValue(3.0, 'b');

    const buildGraph = (v: number[]) => {
      const pa = new HessianValue(v[0], 'a');
      const pb = new HessianValue(v[1], 'b');
      return pa.mul(pb.mul(pb)) as HessianValue;
    };

    const H = computeHessian(buildGraph, [a, b]);

    // Expected Hessian at (a=2, b=3):
    // ∂²f/∂a² = 0
    // ∂²f/∂b² = 2a = 4
    // ∂²f/∂a∂b = 2b = 6
    console.log('\nHessian of a*b² at (2, 3):');
    console.log(H);

    expect(Math.abs(H[0][0] - 0)).toBeLessThan(0.01);  // ∂²f/∂a² = 0
    expect(Math.abs(H[1][1] - 4)).toBeLessThan(0.1);   // ∂²f/∂b² = 2a = 4
    expect(Math.abs(H[0][1] - 6)).toBeLessThan(0.1);   // ∂²f/∂a∂b = 2b = 6
    expect(Math.abs(H[1][0] - 6)).toBeLessThan(0.1);   // Symmetric
  });

  it('computes Hessian of complex expression', () => {
    const x = new HessianValue(1.0, 'x');
    const y = new HessianValue(2.0, 'y');
    const z = new HessianValue(3.0, 'z');

    const buildGraph = (v: number[]) => {
      const px = new HessianValue(v[0], 'x');
      const py = new HessianValue(v[1], 'y');
      const pz = new HessianValue(v[2], 'z');
      return px.mul(px)
        .add(px.mul(py))
        .add(py.mul(pz))
        .add(pz.mul(pz)) as HessianValue;
    };

    const H = computeHessian(buildGraph, [x, y, z]);

    console.log('\nHessian of x² + xy + yz + z² at (1, 2, 3):');
    console.log(H);

    // Expected Hessian:
    // ∂²f/∂x² = 2
    // ∂²f/∂y² = 0
    // ∂²f/∂z² = 2
    // ∂²f/∂x∂y = 1
    // ∂²f/∂x∂z = 0
    // ∂²f/∂y∂z = 1

    expect(Math.abs(H[0][0] - 2)).toBeLessThan(0.01);  // ∂²f/∂x²
    expect(Math.abs(H[1][1] - 0)).toBeLessThan(0.01);  // ∂²f/∂y²
    expect(Math.abs(H[2][2] - 2)).toBeLessThan(0.01);  // ∂²f/∂z²
    expect(Math.abs(H[0][1] - 1)).toBeLessThan(0.01);  // ∂²f/∂x∂y
    expect(Math.abs(H[0][2] - 0)).toBeLessThan(0.01);  // ∂²f/∂x∂z
    expect(Math.abs(H[1][2] - 1)).toBeLessThan(0.01);  // ∂²f/∂y∂z
  });

  it('shows analytical second derivative rules', () => {
    console.log('\n=== Second Derivative Rules ===\n');

    console.log('Addition: f = a + b');
    console.log('  ∂f/∂a = 1, ∂f/∂b = 1');
    console.log('  ∂²f/∂a² = 0, ∂²f/∂b² = 0, ∂²f/∂a∂b = 0\n');

    console.log('Multiplication: f = a * b');
    console.log('  ∂f/∂a = b, ∂f/∂b = a');
    console.log('  ∂²f/∂a² = 0, ∂²f/∂b² = 0, ∂²f/∂a∂b = 1\n');

    console.log('Square: f = a²');
    console.log('  ∂f/∂a = 2a');
    console.log('  ∂²f/∂a² = 2\n');

    console.log('Power: f = aⁿ');
    console.log('  ∂f/∂a = n·aⁿ⁻¹');
    console.log('  ∂²f/∂a² = n(n-1)·aⁿ⁻²\n');

    console.log('Product with square: f = a·b²');
    console.log('  ∂f/∂a = b², ∂f/∂b = 2ab');
    console.log('  ∂²f/∂a² = 0, ∂²f/∂b² = 2a, ∂²f/∂a∂b = 2b\n');

    console.log('Sin: f = sin(a)');
    console.log('  ∂f/∂a = cos(a)');
    console.log('  ∂²f/∂a² = -sin(a)\n');

    console.log('Exp: f = exp(a)');
    console.log('  ∂f/∂a = exp(a)');
    console.log('  ∂²f/∂a² = exp(a)\n');

    console.log('Division: f = a/b');
    console.log('  ∂f/∂a = 1/b, ∂f/∂b = -a/b²');
    console.log('  ∂²f/∂a² = 0, ∂²f/∂b² = 2a/b³, ∂²f/∂a∂b = -1/b²\n');
  });
});
