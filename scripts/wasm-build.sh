#!/bin/bash
set -e

# Enable SIMD128
export RUSTFLAGS="-C target-feature=+simd128,+bulk-memory"

wasm-pack build \
  --target web \
  --release \
  --out-dir pkg 

echo "WASM build complete with SIMD enabled"