#!/bin/bash
# Regenerate JIT performance benchmark data

echo "Running JIT benchmarks..."
cd "$(dirname "$0")/../.."
npx tsx src/jit-benchmark.ts

echo ""
echo "Done! View results at docs/jit/index.html"
