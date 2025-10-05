# Developable Sphere Demo - Complete! ✅

## 🎯 What This Demo Does

This demo implements the **Developability of Triangle Meshes** algorithm (SIGGRAPH 2018) using ScalarAutograd's automatic differentiation and L-BFGS optimizer.

Starting with a sphere (which cannot be flattened), the optimization evolves it into a **piecewise developable surface** - one that can be cut into flat pieces, like paper or sheet metal.

## 🚀 Running the Demo

**The demo is currently running at: http://localhost:5178**

To start it again later:
```bash
cd C:\Dev\ScalarAutograd\demos\developable-sphere
npm run dev
```

## 🎮 How to Use

1. **Set Subdivisions** (1-3): Controls mesh resolution
   - Level 1: 42 vertices (fast)
   - Level 2: 162 vertices (recommended)
   - Level 3: 642 vertices (slow but detailed)

2. **Set Max Iterations** (20-200): Controls optimization length

3. **Click "Run Optimization"**: Starts the gradient descent
   - Watch the progress indicator
   - See energy decrease in real-time
   - Wait for completion (may take 1-10 minutes depending on settings)

4. **Explore Results**:
   - **Drag** to rotate the mesh
   - **Scroll** to zoom
   - **Play** button to animate the evolution
   - **Frame slider** to manually scrub through optimization history

## 📊 Understanding the Results

### Color Coding
- **Blue vertices** = Hinges (developable, can be flattened)
- **Orange vertices** = Seams (curved, concentrate curvature)

### Metrics
- **Hinge Vertices**: Number of developable vertices
- **Seam Vertices**: Number of curved vertices
- **Developable Ratio**: Percentage of mesh that is flattenable
- **Avg Energy**: Lower is better (target: <0.01)

### Expected Evolution
- **Start**: Smooth sphere, 0% developable, high energy
- **End**: Faceted surface with sharp creases, 80-95% developable, low energy

## 🏗️ Implementation Highlights

### Library Extensions Added
1. **Vec3 Enhancements**: New differentiable vector operations
   - `angleBetween()`, `project()`, `lerp()`, `distance()`
   - Component-wise `min()`/`max()`
   - Array conversion utilities

2. **Geometry Utility Class**: Triangle mesh operations
   - `triangleNormal()`, `triangleArea()`, `interiorAngle()`
   - `angleDefect()`, `dihedralAngle()`
   - All fully differentiable with automatic gradients

### Demo Components
- **TriangleMesh**: Efficient mesh data structure with caching
- **IcoSphere**: Generates subdivided spheres
- **PartitionEnumerator**: Finds all valid bipartitions of vertex stars
- **DevelopableEnergy**: Implements the combinatorial energy function
- **DevelopableOptimizer**: Wraps L-BFGS for mesh optimization
- **MeshRenderer**: Three.js visualization with interactive controls

## 📈 Performance Tips

### For Faster Results
- Use **Subdivision Level 1** (42 vertices)
- Set **Max Iterations to 50-100**
- Energy will converge quickly but result may be less smooth

### For Best Quality
- Use **Subdivision Level 2-3** (162-642 vertices)
- Set **Max Iterations to 100-200**
- Takes longer but produces smoother, more developable surfaces

## 🔬 Technical Details

### Energy Function
For each vertex, we:
1. Enumerate all bipartitions of its triangle star
2. For each partition, measure how "flat" each region is
3. Take the minimum energy across all partitions
4. Sum across all vertices

The optimizer adjusts vertex positions to minimize this total energy.

### Optimization
- **Method**: L-BFGS (quasi-Newton)
- **Parameters**: ~126-1926 (3 coordinates per vertex)
- **Gradient**: Computed automatically via reverse-mode autodiff
- **Convergence**: Typically 50-200 iterations

## 📁 Project Structure

```
demos/developable-sphere/
├── src/
│   ├── mesh/
│   │   ├── TriangleMesh.ts      # Core mesh data structure
│   │   └── IcoSphere.ts          # Sphere generation
│   ├── energy/
│   │   ├── PartitionEnumerator.ts  # Bipartition enumeration
│   │   └── DevelopableEnergy.ts    # Energy computation
│   ├── optimization/
│   │   └── DevelopableOptimizer.ts # L-BFGS wrapper
│   ├── visualization/
│   │   └── MeshRenderer.ts       # Three.js rendering
│   ├── App.tsx                   # React UI
│   ├── App.css                   # Styling
│   └── main.tsx                  # Entry point
├── package.json
├── tsconfig.json
├── vite.config.ts
├── index.html
├── README.md
├── PLAN.md
└── DEMO_SUMMARY.md (this file)
```

## 🎓 What You'll Learn

This demo showcases:
- **Geometric optimization** using automatic differentiation
- **L-BFGS** for high-dimensional non-convex problems
- **Differentiable geometry** operations on triangle meshes
- **Interactive visualization** of optimization progress
- **Real-world application** of autodiff to computer graphics

## 🐛 Troubleshooting

### Build Errors
If you see import errors, rebuild the main library:
```bash
cd C:\Dev\ScalarAutograd
npm run build
```

### Performance Issues
- Reduce subdivision level
- Lower max iterations
- Close other browser tabs
- The optimization runs in the main thread and will block the UI

### Visualization Issues
- Make sure WebGL is enabled in your browser
- Try a different browser (Chrome/Edge recommended)
- Refresh the page

## 📚 References

- **Paper**: Stein, O., Grinspun, E., & Crane, K. (2018). "Developability of Triangle Meshes". ACM SIGGRAPH
- **Project**: https://www.cs.cmu.edu/~kmcrane/Projects/DevelopabilityOfTriangleMeshes/
- **ScalarAutograd**: https://github.com/mfagerlund/ScalarAutograd

---

Enjoy exploring geometric optimization with ScalarAutograd! 🎉
