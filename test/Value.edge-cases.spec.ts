import { Value } from "../src/Value";

// Edge cases and error handling
describe('Value edge cases and error handling', () => {
    it('throws on invalid numeric inputs', () => {
      expect(() => new Value(NaN)).toThrow();
      expect(() => new Value(Infinity)).toThrow();
      expect(() => new Value(-Infinity)).toThrow();
    });
  
    it('handles gradient accumulation correctly', () => {
      const x = new Value(2, 'x', true);
      const y = x.mul(3);
      const z = x.mul(4);
      const out = y.add(z);
      out.backward();
      expect(x.grad).toBe(7); // 3 + 4
    });
  
    it('handles repeated use of same value in expression', () => {
      const x = new Value(3, 'x', true);
      const y = x.mul(x).mul(x); // x^3
      y.backward();
      expect(x.grad).toBeCloseTo(27); // 3*x^2 = 27
    });
  
    it('throws on division by zero', () => {
      const a = new Value(1);
      const b = new Value(0);
      expect(() => a.div(b)).toThrow();
    });
  
    it('throws on log of negative number', () => {
      const x = new Value(-1);
      expect(() => x.log()).toThrow();
    });
  
    it('throws on negative base with fractional exponent', () => {
      const x = new Value(-2);
      expect(() => x.pow(0.5)).toThrow();
    });
});

// Complex expressions
describe('Complex mathematical expressions', () => {
    it('computes gradient of complex expression', () => {
      const x = new Value(0.5, 'x', true);
      const y = x.sin().mul(x.cos()).add(x.exp());
      y.backward();
      const expected = Math.cos(0.5)**2 - Math.sin(0.5)**2 + Math.exp(0.5);
      expect(x.grad).toBeCloseTo(expected, 4);
    });
  
    it('handles nested activation functions', () => {
      const x = new Value(0.5, 'x', true);
      const y = x.tanh().sigmoid().relu();
      y.backward();
      expect(x.grad).toBeGreaterThan(0);
    });
});
