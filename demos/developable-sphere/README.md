# Developable Sphere Demo

This demo implements the **Developability of Triangle Meshes** algorithm from [Stein, Grinspun, and Crane (SIGGRAPH 2018)](https://www.cs.cmu.edu/~kmcrane/Projects/DevelopabilityOfTriangleMeshes/).

Starting with a sphere (which cannot be flattened without distortion), we use gradient descent with L-BFGS to evolve it toward a **piecewise developable surface** - one that can be cut into flat pieces and manufactured from sheet material.

## Key Insight

By minimizing a local "developability energy" at each vertex, the mesh naturally evolves to concentrate all curvature onto a sparse collection of seam curves, leaving the rest of the surface perfectly flattenable.

## Running the Demo

```bash
# From the demo directory
npm run dev

# Or from the root directory
cd demos/developable-sphere
npm run dev
```

Then open http://localhost:5173 in your browser.

## How It Works

1. **Initial Mesh**: Start with an icosphere (subdivided icosahedron)
2. **Energy Computation**: For each vertex, find the best bipartition of its triangle star and compute how "hinge-like" it is
3. **Optimization**: Use L-BFGS to minimize the total developability energy by adjusting vertex positions
4. **Visualization**: Color vertices by their developability (blue = hinge, orange = seam)

## Controls

- **Drag**: Rotate the mesh
- **Scroll**: Zoom in/out
- **Subdivisions**: Control mesh resolution (1-3)
- **Max Iterations**: Control optimization length (20-200)
- **Run Optimization**: Start the gradient descent
- **Play/Pause**: Animate through the optimization history
- **Frame Slider**: Manually scrub through frames

## Expected Results

- **Initial Sphere**: High energy (~0.05-0.1), 0% developable
- **After Optimization**: Low energy (~0.001-0.01), 80-95% developable
- **Visual**: Distinct flat patches separated by sharp creases (seam curves)

## Implementation

The demo showcases several features of ScalarAutograd:

- **Differentiable Geometry**: All mesh operations (normals, areas, angles) are differentiable
- **L-BFGS Optimization**: Efficient quasi-Newton method for non-convex optimization
- **Vec3 Operations**: Differentiable 3D vector arithmetic
- **Geometry Utilities**: Triangle mesh operations with automatic gradients

## Architecture

```
src/
├── mesh/
│   ├── TriangleMesh.ts      # Core mesh data structure
│   └── IcoSphere.ts          # Sphere generation
├── energy/
│   ├── PartitionEnumerator.ts  # Enumerate bipartitions
│   └── DevelopableEnergy.ts    # Energy computation
├── optimization/
│   └── DevelopableOptimizer.ts # L-BFGS wrapper
├── visualization/
│   └── MeshRenderer.ts       # Three.js rendering
├── App.tsx                   # React UI
└── main.tsx                  # Entry point
```

## References

- **Paper**: Stein, O., Grinspun, E., & Crane, K. (2018). "Developability of Triangle Meshes". ACM SIGGRAPH
- **ScalarAutograd**: https://github.com/mfagerlund/ScalarAutograd
