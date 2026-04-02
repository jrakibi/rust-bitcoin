#!/bin/bash
# Benchmark script for x86_64: compares baseline vs SSE4.1 4-way vs AVX2 8-way
# Run this on an x86_64 machine with AVX2 support (e.g. AWS c5.large)
set -e

MERKLE="primitives/src/merkle_tree.rs"
CRYPTO="hashes/src/sha256/crypto.rs"
BENCH_CMD="cargo bench --manifest-path benches/Cargo.toml --bench merkle_tree"

echo "=== Run 1: Baseline (old stack-based path, no SIMD merkle) ==="
# Ensure merkle gate is aarch64 only (default state)
sed -i.bak 's/any(target_arch = "aarch64", target_arch = "x86_64")/target_arch = "aarch64"/g' "$MERKLE"
$BENCH_CMD -- --save-baseline baseline
git checkout -- "$MERKLE"

echo ""
echo "=== Run 2: AVX2 8-way ==="
# Widen merkle gate to include x86_64
sed -i.bak 's/target_arch = "aarch64"/any(target_arch = "aarch64", target_arch = "x86_64")/g' "$MERKLE"
$BENCH_CMD -- --save-baseline avx2-8way
git checkout -- "$MERKLE"

echo ""
echo "=== Run 3: SSE4.1 4-way (disable AVX2 dispatch) ==="
# Widen merkle gate
sed -i.bak 's/target_arch = "aarch64"/any(target_arch = "aarch64", target_arch = "x86_64")/g' "$MERKLE"
# Comment out AVX2 dispatch block
sed -i.bak 's/if std::is_x86_feature_detected!("avx2")/if false \/*avx2 disabled*\//' "$CRYPTO"
$BENCH_CMD -- --save-baseline sse41-4way
git checkout -- "$MERKLE" "$CRYPTO"

echo ""
echo "=== Comparing results ==="
echo "--- AVX2 8-way vs baseline ---"
$BENCH_CMD -- --baseline baseline --load-baseline avx2-8way || true
echo ""
echo "--- SSE4.1 4-way vs baseline ---"
$BENCH_CMD -- --baseline baseline --load-baseline sse41-4way || true
echo ""
echo "--- AVX2 8-way vs SSE4.1 4-way ---"
$BENCH_CMD -- --baseline sse41-4way --load-baseline avx2-8way || true
