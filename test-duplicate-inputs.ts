import { Value } from './src/Value';

console.log('Testing ScalarAutograd duplicate inputs');
console.log('========================================\n');

// Test 1: Normal case - different inputs
console.log('Test 1: x * y (different inputs)');
{
  const x = new Value(2, 'x', true);
  const y = new Value(3, 'y', true);
  const result = x.mul(y);

  result.backward();

  console.log(`Result: ${result.data}`);
  console.log(`x gradient: ${x.grad} (expected: 3.0)`);
  console.log(`y gradient: ${y.grad} (expected: 2.0)`);
  console.log();
}

// Test 2: Same input used twice - this tests the bug
console.log('Test 2: t * t (same input twice)');
{
  const t = new Value(2, 't', true);
  const result = t.mul(t);

  result.backward();

  console.log(`Result: ${result.data}`);
  console.log(`t gradient: ${t.grad} (expected: 4.0)`);

  const expected = 4.0;
  const ratio = t.grad / expected;
  console.log(`Ratio: ${ratio.toFixed(2)}`);

  if (Math.abs(ratio - 1.0) < 0.01) {
    console.log('PASS');
  } else {
    console.log('FAIL - gradient is wrong!');
  }
  console.log();
}

// Test 3: More complex case - t used in log(1 - t^2)
console.log('Test 3: log(1 - t^2) - like in SAC entropy');
{
  const t = new Value(0.5, 't', true);
  const tSquared = t.mul(t);
  const oneMinusTSquared = new Value(1, '1').sub(tSquared);
  const result = oneMinusTSquared.log();

  result.backward();

  console.log(`Result: ${result.data.toFixed(6)}`);
  console.log(`t gradient: ${t.grad.toFixed(6)}`);

  // Analytical derivative of log(1-t^2) is -2t/(1-t^2)
  const expected = -2 * 0.5 / (1 - 0.25);
  console.log(`Expected gradient: ${expected.toFixed(6)}`);
  console.log(`Match: ${Math.abs(t.grad - expected) < 0.0001 ? 'YES' : 'NO'}`);
}
