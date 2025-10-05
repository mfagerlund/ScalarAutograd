# Developable Meshes — Summary & Reproduction Protocol

## Summary (condensed, complete)

### Problem & core idea
Developable surfaces are made by bending flat sheets without stretch/shear. Many discrete notions (e.g., zero angle defect) allow “crumpled” meshes that are not useful for fabrication. The paper defines **discrete developability** for triangle meshes that captures both (i) local flattenability and (ii) straight rulings, then proposes a **variational method** that drives any mesh toward **piecewise developable** geometry by minimizing a **per-vertex energy** supported only on the vertex star. Curvature concentrates along sparse **seam curves**; away from seams the mesh becomes near-developable.

### Discrete developability
- **Flattenable** (discrete): angle defect Ωᵢ = 0 at every vertex; insufficient (permits crumpling).
- **Discrete developable (hinge)**: a vertex star St(i) splits into two edge-connected planar regions whose face normals are constant on each region. Special case: flat (all normals parallel). Consequence: non-flat discrete developable meshes are **discrete ruled** (paths of parallel edges through each vertex).
- **Equivalent characterization (key for optimization)**: a star is a hinge **iff** all its triangle normals lie in a **common plane** (and the star is embedded). This turns developability into a coplanarity test in normal space (Gauss image has zero minimum width).

Corner cases (spikes, needles, fins, double-covers) are excluded by requiring a **simplicial immersion** (locally injective star), mirroring smooth regularity.

### Energies that vanish on hinges
The method minimizes one of two energies; both are per-vertex and sum over vertices.

1) **Covariance energy (recommended)**  
For vertex i, build a 3×3 **normal covariance**
\[ A_i = \sum_{ijk\in St(i)} \theta_i^{jk}\, (N_{ijk} N_{ijk}^\top) \]
where \(\theta_i^{jk}\) are corner angles as tessellation-invariant weights. Let \(\lambda_i\) be the **smallest eigenvalue** of \(A_i\). Define \(E_\lambda = \sum_i \lambda_i\). Minimizing \(E_\lambda\) shrinks the **minimum width** of the Gauss polygon; \(\lambda_i = 0\) iff normals are coplanar (hinge). An intrinsic spherical variant reduces “spikes”. Use a robust 3×3 eigensolver.

**Branching avoidance (optional):** replace the sum with a **max** objective per vertex
\[ \lambda_i^{\max} = \min_{\|u\|=1} \max_{\text{faces in } St(i)} \langle u, N \rangle^2, \quad E_\lambda^{\max} = \sum_i \lambda_i^{\max}, \]
to discourage “V-shaped” branching of rulings along seams.

2) **Combinatorial width (alternative)**  
Partition \(St(i)\) into two **edge-connected** face sets \((F_1,F_2)\). Penalize within-cluster normal variance; pick the best partition:
\[ E_P = \sum_i \min_P \sum_{p\in\{1,2\}} \frac{1}{|F_p|} \sum_{\sigma\in F_p} \|N_\sigma - \bar N_p\|^2. \]
A max-variant similarly avoids branching. This energy is **costlier** (quadratic in valence) than the covariance energy.

**Why piecewise developable emerges:** seams contribute energy but their **measure shrinks** under refinement, so minimizing \(E\) drives curvature to a **sparse curve set** while the rest becomes developable; an L¹ sparsity penalty would instead yield piecewise-flat solutions.

### Gradients (what you need to implement)
Per triangle \(ijk\): areas \(A_{ijk}\), normals \(N_{ijk}\), corner angles \(\theta_i^{jk}\), and their gradients w.r.t. vertex positions. For the covariance energy,
\(\lambda_i = u_i^\top A_i u_i\) (\(u_i\): smallest eigenvector). Away from repeated eigenvalues,
\[ \nabla_{f_p} \lambda_i = 2 \sum \theta_i^{jk} \langle u_i,N_{ijk}\rangle \langle u_i, \nabla_{f_p}N_{ijk} \rangle + \sum (\nabla_{f_p} \theta_i^{jk}) \langle u_i, N_{ijk} \rangle^2, \]
and \(\nabla E\) sums over vertices. For \(E_P\), differentiate the **active partition**; ties → subgradient. For max-variants, differentiate the **maximal term** (subgradient). Primitive formulas for \(A, N, \theta\) and their gradients are listed in the protocol below.

### Optimization loop
- Variables: all vertex positions.  
- Optimizer: **L-BFGS** with strong Wolfe line search; (sub)gradient descent also works. Use **double precision**. A stable **3×3 eigensolver** is critical.
- **Stability remeshing only**: if a triangle has two tiny angles → **edge flip**; one tiny angle → **edge collapse**. No other remeshing to enforce developability.
- **Valence-3 vertices**: must be flat if hinge; **omit** from the energy to serve as **triple points** for seams.
- **Coarse-to-fine**: when \(\|\nabla E\|\) is small or the design is acceptable, apply **regular 4–1 subdivision** and continue. Seam curves smooth naturally under refinement.

### Cutting & flattening (for fabrication/UV)
- Mark vertices as **seams** where developability energy exceeds a tolerance \(\varepsilon>0\), reflecting **material stiffness** (stiffer → cut more). Build a **cut graph** through all marked vertices; on spheres, a **minimum spanning tree** suffices. Add edge weights favoring shortness and alignment with the **small principal direction** (smallest-eigenvector of \(A_i\)).
- Flatten with **Boundary-First Flattening** (BFF-style) to control boundary isometry; LSCM/ABF can cause area blow-ups in skinny connections. Glue tiny “cracks” in parameter space if needed; split charts to avoid global overlaps. Example: lowering \(\varepsilon\) from 0.01 → 0.001 reduces mean scale distortion from ≈2.4 → 4×10⁻⁴ (trend), at the cost of longer cuts.

### Validation & behavior
- Minimizing \(\sum\Omega^2\) gives flattenable but **crumpled** surfaces; minimizing developability energies yields **smooth ruled** regions with clear rulings.
- **Curvature concentrates** on seams; away from seams the mesh becomes highly flattenable.
- Ruling lines may **branch** with sum-energies; **max-variants** mitigate this at extra cost.
- No need to pre-partition into disks; seams appear and smooth **without explicit detection** during optimization.

### Results, applications, limitations
- Approximates complex shapes with **piecewise developable** geometry suited for sheet forming, papercraft, or as seeds for **PQ-strip** extraction and **flank-milling** toolpaths.
- Limitations: non-uniqueness (depends on mesh/initialization/energy/optimizer), potential ruling non-straightness without max-variants or additional remeshing; full manufacturing pipelines (global spline networks, collision-aware toolpaths) require extra steps.

---

## Reproduction Protocol — *Developability of Triangle Meshes* (Stein, Grinspun, Crane 2018)

> Goal: Implement the variational approach that drives a triangle mesh toward **piecewise developable** surfaces by minimizing a **per-vertex developability energy** supported on the vertex star. This protocol is sufficient to reproduce the paper’s core results and pipeline.

---

### 0. Notation and Data Layout
- Mesh: manifold triangle mesh **M = (V, E, F)** with vertex positions **f_i ∈ ℝ³**.
- For triangle **ijk**:
  - Oriented unit normal **N_ijk ∈ S²**.
  - Area **A_ijk**.
  - Interior angle at vertex **i**: **θ_i^{jk}**.
- Vertex star **St(i)** = set of faces incident to **i** in radial order.
- Angle defect **Ω_i = 2π − Σ_{ijk∈F} θ_i^{jk}** (diagnostic only).

**Per vertex i:**
- Normal covariance **A_i = Σ_{ijk∈St(i)} θ_i^{jk} · (N_{ijk} N_{ijk}^\top)** (3×3).
- Smallest eigenpair **(λ_i, u_i)** with **‖u_i‖=1**, **A_i u_i = λ_i u_i**.

### 1. Discrete Developability (Definitions)
- **Discrete flattenable**: all vertices have zero angle defect (**Ω_i = 0**); *not sufficient*.
- **Discrete developable (hinge)**: **St(i)** can be partitioned into two **edge-connected** planar regions whose face normals are constant on each region; equivalently, **all face normals in St(i) lie in a common plane**.
- **Discrete ruled**: every interior vertex lies on a path of parallel edges with endpoints on the boundary (follows for non-flat discrete developables).

### 2. Energies to Encourage Developability
Two equivalent targets that vanish on hinges.

#### 2.1 Covariance Energy (recommended)
For each vertex: **λ_i** = smallest eigenvalue of **A_i = Σ θ_i^{jk} N_{ijk} N_{ijk}^\top**. Define **E_λ = Σ_i λ_i**.

Notes:
- Angle weights **θ_i^{jk}** make the energy tessellation- and scale-invariant.
- Minimizing **E_λ** shrinks the minimum width of the Gauss image polygon per vertex.
- Intrinsic (spherical) variant optional; reduces spikes.

**Branching-avoid variant (optional):**
\(\lambda_i^{\max} = \min_{\|u\|=1} \max_{ijk∈St(i)} \langle u, N_{ijk} \rangle^2\),
**E_λ^{max} = Σ_i λ_i^{max}**.

#### 2.2 Combinatorial Width (alternative)
- Partition **St(i)** into two edge-connected sets **F₁,F₂**.
- Let **\(\bar N_p\)** be the average normal over **F_p**.
- Partition score: \(\pi(P) = \sum_{p=1,2} ( 1/|F_p| \sum_{\sigma∈F_p} \|N_\sigma − \bar N_p\|^2 )\).
- **E_P = Σ_i \min_{P \text{ edge-connected}} \pi(P)**.

**Branching-avoid variant:** use **\(\pi_{max}(P) = \max_{F_k∈P} \max_{N_1,N_2∈F_k} \|N_2 − N_1\|^2\)**.

### 3. Quantities and Gradients
Per triangle **ijk**:
- **Area**: \(A_{ijk} = \tfrac{1}{2} \| (f_j − f_i) × (f_k − f_i) \|\).
- **Unit normal**: \(N_{ijk} = ((f_j − f_i) × (f_k − f_i)) / (2 A_{ijk})\).
- **Corner angle**: \(\theta_i^{jk} = \angle(f_j − f_i, f_k − f_i)\) (use `atan2` with clamping).

**Gradients** (w.r.t. vertex positions):
- \(\nabla_{f_i} A_{ijk} = \tfrac{1}{2} · N_{ijk} × (f_k − f_j)\).
- \(\nabla_{f_i} N_{ijk} = (1/A_{ijk}) · ( (f_k − f_j) × N_{ijk} ) N_{ijk}^\top\) (and cyclic for j,k).
- \(\nabla_{f_j} \theta_i^{jk} = N_{ijk} × (f_i − f_j) / \|f_i − f_j\|\);
  \(\nabla_{f_k} \theta_i^{jk} = N_{ijk} × (f_k − f_i) / \|f_k − f_i\|\);
  \(\nabla_{f_i} \theta_i^{jk} = −(\nabla_{f_j} \theta_i^{jk} + \nabla_{f_k} \theta_i^{jk})\).

**Gradient of covariance energy**:
- For vertex **i**, with smallest eigenvector **u_i** (simple eigenvalue),
  - **\(\lambda_i = u_i^\top A_i u_i\)**,
  - **\(\nabla_{f_p} \lambda_i\)** as in the summary above; accumulate over all vertices.
- Handle non-smooth events (eigenvalue multiplicity) via subgradient or tiny Tikhonov jitter to **A_i**.

**Gradient of combinatorial energy**:
- Once the best partition **P\*** is picked (minimizer), differentiate **\(\pi(P\*)\)**; ties → subgradient.
- For **\(\pi_{max}\)**, differentiate only the maximal term.

### 4. Optimization Strategy
- **Variables**: all vertex positions **f ∈ ℝ^{3|V|}** (optionally pin a few boundary vertices or center the mesh).
- **Method**: L-BFGS with strong Wolfe line search; projected (sub)gradient descent is acceptable.
- **Precision**: double throughout.
- **Eigen-solver**: robust 3×3 symmetric eigensolver for smallest eigenpair (Rayleigh quotient iteration or closed-form; stabilize near degeneracy).

**Remeshing for stability only**:
- If a triangle has **two very small angles**, perform an **edge flip**.
- If a triangle has **one very small angle**, **edge-collapse** it.
- “Very small” heuristic: e.g., **< 5°**. Maintain manifoldness and orientation.

**Triple points**:
- Omit **valence-3** vertices from the energy (they must be flat), allowing clean seam junctions.

**Regularization via refinement**:
- Work coarse-to-fine. When **\(\|\nabla E\|\)** stalls or the design is acceptable, apply **regular 4–1 subdivision**; continue optimizing; repeat ≤ 2–3×.

**Stopping criteria**:
- Relative gradient norm < **1e−6**, or no significant energy drop over k iterations, or desired quality reached.

### 5. Cutting and Flattening (for fabrication/UVs)
**Seam detection**:
- Mark vertices with **\(\lambda_i > \varepsilon\)**; choose \(\varepsilon\) per material stiffness (e.g., 1e−2, 5e−3, 2e−3, 1e−3).

**Cut graph**:
- Compute a cut passing through all marked vertices.
  - General topology: cut to disk while passing marked vertices (e.g., shortest cut that hits all).
  - Spherical topology: a **minimum spanning tree** suffices.

**Edge weights (aesthetics)**:
- Weight = **α · length + β · alignment**, where alignment encourages following the principal direction from the **smallest-eigenvector** at endpoints (e.g., α=1, β∈[0,1]).

**Flattening**:
- Use a **boundary-first** method (BFF-style) to enforce near-isometry at the boundary and avoid extreme area distortions in skinny regions.
- Optionally glue tiny cracks in parameter space and re-flatten. If global overlaps occur, split into a few charts.

**Diagnostic**:
- Visualize **log conformal factor** over faces to inspect scale distortion.

### 6. Pipeline (Top–Down Reproduction)
1. **Input** a coarse target mesh (or coarsen an existing model).
2. **Compute** per-face areas, normals; per-corner angles; build per-vertex **A_i**.
3. **Evaluate** **E_λ** and **∇E_λ**.
4. **Optimize** vertex positions (L-BFGS). Apply stability remeshing as needed. Omit valence-3 vertices from energy.
5. **Refine** with 4–1 subdivision; repeat 2–4.
6. **Extract seams** via threshold **\(\varepsilon\)**; build **cut graph**.
7. **Flatten** with boundary-first approach; fix overlaps; optionally merge tiny cracks.
8. **(Optional)** Extract **PQ strips** and plan flank-milling toolpaths downstream.

### 7. Parameters / Heuristics (suggested)
- Small-angle threshold for flips/collapses: **5°** (tune 3–10°).
- Gradient norm tolerance: **1e−6**.
- Eigen-jitter: add **δ I** with **δ ~ 1e−12** if needed.
- Seam threshold **\(\varepsilon\)**: try **1e−2 → 1e−3**; lower \(\varepsilon\) yields longer but more isometric cuts and less scale distortion. Example trend for average scale distortion when \(\varepsilon\) decreases: **2.4, 0.09, 0.006, 0.0004**.
- Max-variant (branching avoidance): apply at fine stages if you observe “V” branches; it is costlier.

### 8. Validation and Diagnostics
- **Curvature concentration**: plot per-vertex **|Ω_i|** before/after; expect concentration onto sparse seams; away from seams, flattenability improves.
- **Crumpling control**: compare minimizing **Σ Ω_i²** vs **E_λ**; the latter yields smoother ruled regions.
- **Seam smoothing**: seam curves regularize as you refine (energetically small but finite on coarse meshes).
- **Ruling visualization**: shade edges by dihedral angle; look for approximate **PQ strip** structure.

### 9. Pseudocode (Covariance Energy Variant)
```pseudo
function optimize_to_developable(mesh M):
    for it in 1..MaxOuter:
        E, grad = 0, zeros_like(vertices(M))
        for i in vertices(M):
            Ai = zeros(3,3)
            for (ijk) in star_faces(i) in radial order:
                N = face_normal(ijk)
                theta = corner_angle(i, ijk)
                Ai += theta * (N * N^T)
            (lambda, u) = smallest_eigenpair(Ai)
            E += lambda
            for (ijk) in star_faces(i):
                N = face_normal(ijk)
                theta = corner_angle(i, ijk)
                for p in vertices(ijk):
                    dtheta_dp = grad_corner_angle(p; i, ijk)
                    dN_dp     = grad_face_normal(p; ijk)
                    grad[p] += 2 * theta * dot(u, N) * dot(u, dN_dp) + dtheta_dp * (dot(u, N))^2

        f_new = LBFGS_step(M.vertices, E, grad)
        M.update_vertices(f_new)

        if small_angles_detected(M):
            M = stabilize_by_edge_flips_and_collapses(M, threshold_deg=5)

        if converged(E, grad): break

    return M
```

### 10. Implementation Tips
- Ensure consistent face orientation; fix flipped normals early.
- Cache triangle primitives and gradients; update only affected stars per local change if optimizing performance.
- Use **double precision**; avoid ε/δ that dwarf real curvature signals.
- Handle eigenvalue multiplicity via subgradient or slight diagonal regularization.

### 11. Expected Outcomes
- Coarse meshes evolve toward **piecewise developable** shapes.
- **Gaussian curvature** concentrates along **seam curves**; elsewhere meshes become nearly flattenable.
- Seams become smoother on refinement; ruling directions emerge; the max-variant helps avoid “V” branching.

### 12. Optional Downstream
- **Paper craft**: segment along cuts, print/plot, assemble.
- **Flank milling**: extract PQ strips, fit NURBS, generate toolpaths in CAD/CAM.

### 13. Minimal Dependencies
- Linear algebra (3×3 eigensolver).
- L-BFGS + line search.
- Mesh data structure with edge flip/collapse operations.
- Optional UV flattener with boundary-length control (BFF-style).

---

**Checklist**
- [ ] Per-triangle primitives and gradients
- [ ] Build per-vertex covariance **A_i**
- [ ] Smallest eigenpair per vertex (robust)
- [ ] Assemble **E_λ** and **∇E_λ**
- [ ] L-BFGS loop + stabilization flips/collapses
- [ ] Omit valence-3 vertices from energy
- [ ] Coarse-to-fine 4–1 subdivision schedule
- [ ] Seam thresholding, cut graph, boundary-first flattening
- [ ] Diagnostics: angle defect, dihedral shading, conformal factor



---


## Implementation notes from the official code review

**Source**: Official C++ repo with CMake build and vendored libigl (macOS 10.13/Xcode target). Run `./applications/main/main path_to_mesh.obj`. `../runs/*.txt` drives preset examples.

**Energies implemented**
- Covariance energy (L2 and L∞). Two variants: a **spike-reduced** version and the **raw** version.
- Combinatorial energy (L2 and L∞).
- Keyboard `e` cycles: 0/1 = cov(spike-reduced) L2/L∞; 2/3 = cov(raw) L2/L∞; 4/5 = combinatorial L2/L∞.

**Optimization options**
- `m` toggles: **L-BFGS + Overton line search**, **gradient descent + backtracking**, **gradient descent (no LS)**.
- `TAB` = single step; `Space` = continuous; arrows adjust step (overridden by line search).

**Mesh operations & diagnostics**
- `s` = in-place **1–4 subdivision** (use between optimization stages).
- `n` = toggle **remeshing of degenerate triangles**.
- `l` = one pass **Laplacian smoothing**.
- `p` = energy plot; `c` = color by energy; `d` = edge color by dihedral.
- `w` = write current mesh (`out.obj`); `v` = record frames (`frame*.obj`).

**Cutting & flattening**
- `f` = **cut + flatten**; writes `flattened_out.obj` and `mesh_with_flattened_texcoord.obj`.
- Sensitive to parameters defined near the top of `applications/main/measure_once_cut_twice.cpp`.

**Dependencies**
- libigl submodule is pinned in the repo. MATLAB is optional (can be disabled in CMake and by `#undef REQUIRE_MATLAB`).

**Preprocessing (authors’ guide)**
- Use *Instant Field-Aligned Meshes*; set **Triangles**, enable **extrinsic** and **sharp creases**.
- Start around **5k vertices**; denser meshes deform less.
- Align the orientation field with the **smaller principal curvature** direction before remeshing.
- Export the remeshed triangle mesh; fix holes/non-manifold issues in an editor.

---

## Code & Repro (official)
- **Repository:** [github.com/odedstein/DevelopabilityOfTriangleMeshes](https://github.com/odedstein/DevelopabilityOfTriangleMeshes)
- **Build (CMake):**
  ```bash
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j
  ./applications/main/main path/to/mesh.obj
  ```
- **Notes:** Repo pins a compatible **libigl** as a submodule; targets macOS 10.13/Xcode. MATLAB dependency can be disabled in CMake and `applications/main/main.cpp`.

---

## Other implementations and related tools (reviewed)

> The items below are **not** part of Stein–Grinspun–Crane (2018) but are useful companions/references. Each includes code pointers and actionable notes for reproduction/integration.

### A) Developability‑Driven Piecewise Approximations for Triangular Meshes (SIGGRAPH 2022)
**Repo:** USTC – `DevelopApp` (C++/VS2019, OpenMesh+Eigen, OpenMP). Output: `final.obj` and `final_seg.txt` (per‑triangle patch IDs).
**Use‑cases:** Post‑segment a near‑developable mesh into few developable patches; good as a second stage after covariance/combinatorial energy minimization.
**Operational notes:** CLI binary; tune patch count indirectly via their energy; stable on typical 50k–150k‑tri meshes; uses surface‑mesh framework. 
**Integration tip:** Run after our “Cutting & flattening” to get patch labels, then flatten per patch to reduce seam stretching. 
**Reference:** Zhao et al., 2022. Code: https://github.com/QingFang1208/DevelopApp

### B) Evolutionary Piecewise Developable Approximations (SIGGRAPH 2023)
**Repo:** `EvoDevelop` (C++/VS2019, OpenMesh+Eigen, OpenMP). CLI: `EvoDevelop.exe [INPUT_OBJ] [OUTPUT_PATH] [DISTORTION_THRESHOLD ~0.025] [CONE_BOUND]`.
**Use‑cases:** Automatic partitioning into few developable patches with explicit distortion budget; returns `output_seg.txt` + fitness logs.
**Operational notes:** Practical when you need a **target #patches or bounded distortion**; deterministic per seed; sensible defaults included.
**Integration tip:** Use as a *constraint driver*: run 2018 energy ↓ until no further decrease, then run EvoDevelop with your acceptable distortion to coarsen patches.
**Reference:** Zhao et al., 2023. Code: https://github.com/USTC-GCL-F/EvoDevelop

### C) Developable Approximation via Gauss Image Thinning (SGP 2021)
**Repo:** ETH – Gauss Thinning (C++/Makefile; deps: libigl, Eigen). CLI: `GaussThinning input.off out.obj examples/bunny 500 2.5` (iter, min‑angle).
**Use‑cases:** Alternates between **Gauss‑image thinning** and surface updates—tends to sharpen and regularize seam curves cleanly.
**Operational notes:** Easy to build; parallel variant available; prefers manifold, reasonably clean inputs.
**Integration tip:** Use as a **seam initializer** before 2018 energies on very noisy data; then switch back to covariance L2/L∞ for final polishing.
**Reference:** Binninger et al., 2021. Code: https://github.com/FloorVerhoeven/DevelopableApproximationViaGaussImageThinning

### D) Developability of Heightfields via Rank Minimization (SIGGRAPH 2020)
**Repo(s):** Official MATLAB + scripts; auxiliary C++ demo. Heightfield‑only (z=f(x,y)).
**Use‑cases:** For DEMs/terrain & scans that are near‑heightfields, produces piecewise‑developable results via convex rank minimization.
**Operational notes:** Not general triangle meshes; good as **preconditioner** to deliver a near‑developable start for 2018 energies.
**Integration tip:** Rasterize mesh → heightfield → run rank‑min → project back to mesh and continue with covariance energy.
**Reference:** Sellán–Aigerman–Jacobson, 2020. Project/code: https://www.dgp.toronto.edu/projects/compressed-developables/

### E) Houdini prototypes (community)
**Status:** VEX/OpenCL experiments computing the **hinge (covariance/combinatorial) energy** and gradients; optimization typically delegated to external solvers. Useful for DCC integration; not parity‑complete.
**Reference:** “Mesh Developability” blog write‑up: https://houdinigubbins.wordpress.com/2018/09/19/mesh-developability/

---

## Practical cross‑method playbook
- **Seam quality:** If seams are jagged after 2018 energies, run **Gauss‑Thinning** for 100–500 iters (min‑angle 2–3°), then resume 2018 **covariance L2**.
- **Patch budgeting:** Use **EvoDevelop** with a `DISTORTION_THRESHOLD` chosen from fabrication tolerances (e.g., 2.5%); its `output_seg.txt` can drive per‑patch flattening.
- **Large models:** Prefer **DevelopApp** (OpenMP) to partition before exporting CNC/fabrication strips.
- **Heightfields:** Use **Rank‑Min** pipeline as an initializer. Works well on scans/terrain.

---



## References
- Stein, O., Grinspun, E., Crane, K. *Developability of Triangle Meshes*. ACM TOG (SIGGRAPH Asia), 2018.

## Local Artefact
- Download path: `/mnt/data/developable-meshes-repro.md`

