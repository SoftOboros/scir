#!/usr/bin/env bash
# scripts/pre-commit.sh - Fast, phased validation prior to commit
set -euo pipefail

echo "[phase 0] format"
cargo fmt --all

echo "[phase 1] clippy (workspace)"
cargo clippy --workspace -- -D warnings

echo "[phase 2] build+test: creator CLI"

echo "[phase 4] docs (nightly)"
export ARTIFACTS_INCLUDE_DIR="$(pwd)/scripts/artifacts/include"
export ARTIFACTS_LIB_DIR="$(pwd)/scripts/artifacts/lib"
export ARTIFACTS_LIB64_DIR="$ARTIFACTS_LIB_DIR"
RUSTDOCFLAGS="--cfg docsrs --cfg nightly" \
    cargo +nightly doc \
    --no-deps


echo "pre-commit: all phases completed"
