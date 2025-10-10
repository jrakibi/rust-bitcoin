#!/usr/bin/env bash
# Local testing script for consensus_encoding fuzz tests
# Works on macOS without honggfuzz

set -e

echo "🧪 Running consensus_encoding fuzz tests (unit test mode)"
echo "======================================================"
echo ""

echo "📦 Testing compact_size..."
RUSTFLAGS='--cfg fuzzing' cargo test --bin consensus_encoding_compact_size --quiet

echo "📦 Testing array..."
RUSTFLAGS='--cfg fuzzing' cargo test --bin consensus_encoding_array --quiet

echo "📦 Testing vector..."
RUSTFLAGS='--cfg fuzzing' cargo test --bin consensus_encoding_vector --quiet

echo ""
echo "✅ All consensus_encoding fuzz tests passed!"
echo ""
echo "Note: These are unit tests. Full fuzzing will run in CI on Linux."
echo "To run full fuzzing locally, use Docker or cargo-fuzz."

