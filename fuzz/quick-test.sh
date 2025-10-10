#!/usr/bin/env bash
# Quick test script - runs fuzz tests for 30 seconds each

echo "🔍 Installing honggfuzz fuzzer..."
cargo install --force honggfuzz --no-default-features

echo ""
echo "🧪 Testing consensus_encoding_compact_size (30 seconds)..."
HFUZZ_RUN_ARGS="--run_time 30 --exit_upon_crash -v" cargo hfuzz run consensus_encoding_compact_size

echo ""
echo "🧪 Testing consensus_encoding_array (30 seconds)..."
HFUZZ_RUN_ARGS="--run_time 30 --exit_upon_crash -v" cargo hfuzz run consensus_encoding_array

echo ""
echo "🧪 Testing consensus_encoding_vector (30 seconds)..."
HFUZZ_RUN_ARGS="--run_time 30 --exit_upon_crash -v" cargo hfuzz run consensus_encoding_vector

echo ""
echo "✅ All fuzz tests completed!"
