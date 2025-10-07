# Limitations and Future Work

## Mesh Topology Operations

Your current TriangleMesh is a simple vertex+face list structure. You need to add edge flip and edge collapse operations to handle degenerate triangles. These are essential for stability.

I can add these operations to TriangleMesh:

1. edgeFlip(v0, v1) - flip edge between two triangles that share it
2. edgeCollapse(v0, v1) - merge v0 and v1, remove degenerate faces
3. detectDegenerateTriangles(minAngleDeg) - find problematic triangles
4. stabilityRemesh() - apply flips/collapses automatically

## Canonization Performance

Canonization should be called something else since it's no longer actually canonization. It's really **graph signature generation** or **graph hashing** - the system creates structural hashes to identify topologically identical computation graphs for kernel reuse. Better names: `GraphHasher`, `GraphSignature`, `GraphFingerprint`, or `StructureHash`.

We should try to make the canonization that's no longer canonization faster.

We need to benchmark canonization - is it possible to get it to be fast enough for *actual* JIT compilation, where we identify what graph the user has created and pick out the best option. We might get away with caching details about the kernel for the next time the caller needs that kernel, but it's gonna be sticky.

If we're doing WebGPU, is there a point to compiling the actual graph? On the other hand, we need a wicked fast method to identify which values can be graphed together, like with the kernels, but the actual compile potentially becomes pointless because we only run each graph once for a very large number of actual values.

## Code Organization Issues

Energy functions contain a lot of weird methods that don't belong there.

Value contains weird code that helps with writing code for forwards and backwards during a compile. Those silly switch cases should be provided into Value.make when we construct the value.
