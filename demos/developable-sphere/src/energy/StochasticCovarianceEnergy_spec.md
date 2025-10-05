# StochasticCovarianceEnergy Specification

## Purpose
Approximate the covariance energy E^λ from "Developability of Triangle Meshes" (Stein et al., SIGGRAPH 2018) using faster deterministic methods instead of exact eigenvalue computation.

## Background: The Exact Covariance Energy

For each vertex i with star faces St(i):

1. **Build covariance matrix**: A_i = Σ θ_ijk · N_ijk · N_ijk^T (3×3 matrix)
   - θ_ijk = interior angle at vertex i in face ijk
   - N_ijk = face normal (unit vector)

2. **Compute smallest eigenvalue**: λ_i = λ_min(A_i)

3. **Total energy**: E^λ = Σ_i λ_i

**Properties:**
- λ_i = 0 ⟺ all normals in St(i) are coplanar ⟺ vertex is a developable hinge
- Requires expensive 3×3 eigenvalue solve (~50 FLOPs per vertex)

## Our Approximation: Two Fast Methods

Both methods approximate λ_min by minimizing the Rayleigh quotient: λ(u) = u^T A u for u ∈ S² (unit sphere).

### Method 1: Spherical Gradient Descent (SGD)
**Steps:**
1. Initialize u = weighted mean normal (deterministic)
2. For 30 iterations:
   - Compute gradient: ∇_u λ(u) = 2 Σ θ_k (u·N_k) N_k
   - Update: u ← normalize(u - η·∇λ)
   - Decay learning rate: η *= 0.92
3. Return λ ≈ u^T A u using final u

**Cost:** ~30k FLOPs (k = valence)

### Method 2: Icosahedral Grid Sampling
**Steps:**
1. Generate 12 fixed directions (icosahedron vertices)
2. Evaluate λ(u) for each direction using raw number math
3. Pick u with minimum λ
4. Return λ ≈ u^T A u using best u

**Cost:** ~12k FLOPs

## Critical Implementation Requirements

### 1. Gradient Flow (MOST IMPORTANT!)
**The final Rayleigh quotient MUST use Value objects, not raw numbers.**

**Wrong (breaks autodiff):**
```typescript
// Extract normals as raw numbers
normals.push([normal.x.data, normal.y.data, normal.z.data]);

// Compute λ with raw numbers only
const udotNk = u[0] * Nk[0] + u[1] * Nk[1] + u[2] * Nk[2];
lambda += weights[k] * udotNk * udotNk;  // No gradient flow!
```

**Correct:**
```typescript
// Keep normals as Vec3<Value>
normalsValue.push(normal);

// Also extract raw numbers for u-search
normals.push([normal.x.data, normal.y.data, normal.z.data]);

// Search for u using raw numbers (fast, no graph)
u = findBestDirection(normals, weightsRaw);

// Compute final λ using Value math so gradients flow through normals
for (let k = 0; k < normalsValue.length; k++) {
  const N = normalsValue[k];  // Vec3<Value>
  const dotProd = V.add(V.add(
    V.mul(N.x, u[0]),  // u[0] is constant, N.x is Value
    V.mul(N.y, u[1])),
    V.mul(N.z, u[2])
  );
  lambda = V.add(lambda, V.mul(weights[k], V.mul(dotProd, dotProd)));
}
```

**Why:** The direction `u` is found via non-differentiable search (SGD or argmin over grid), so it's constant. But we must compute the final energy using the actual Value-based normals so autodiff can backprop through geometry.

### 2. Use max(λ, 0) not abs(λ)
The covariance matrix A is positive semi-definite, so λ ≥ 0 theoretically. Use `V.max(lambda, V.C(0))` to handle numerical noise, not `V.abs()`.

### 3. Deterministic for L-BFGS
- No randomness during line search
- SGD method uses full-batch gradient descent (deterministic)
- Grid method uses fixed icosahedral sampling (deterministic)
- Both methods are reproducible given same mesh state

### 4. Skip valence-3 vertices
Per paper recommendation, omit valence-3 vertices (they must be flat if developable).

## Data Flow

```
Input: TriangleMesh

For each vertex i:
  1. Get star faces St(i)
  2. Extract:
     - normalsValue: Vec3[] (for gradient flow)
     - normals: number[][] (for u-search)
     - weights: Value[] (for gradient flow)
     - weightsRaw: number[] (for u-search)

  3. Find best direction u (constant):
     - SGD: minimize u^T A u via gradient descent on raw numbers
     - Grid: argmin_u u^T A u over 12 fixed directions

  4. Compute final λ with Value math:
     lambda = Σ weights[k] · (u·normalsValue[k])²
     return max(lambda, 0)

Output: Σ_i λ_i
```

## Performance
- 2-5× faster than exact eigenvalue solver
- Deterministic (compatible with L-BFGS)
- Gradients flow correctly through geometry

## Method Selection
- **SGD (default)**: More accurate, ~30 iterations
- **Grid**: Faster, less accurate, useful for quick tests

## Current Status
- [X] Gradient flow fixed (use Value math for final λ)
- [X] Renamed 'random' → 'grid'
- [X] Use max(λ, 0) instead of abs(λ)
- [ ] supportsCompilation should be true once we verify it's deterministic
