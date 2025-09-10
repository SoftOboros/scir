#!/usr/bin/env bash
# scripts/pre-commit.sh - Fast, phased validation prior to commit
set -euo pipefail

echo "[phase 0] format"
cargo fmt --all

echo "[phase 1] clippy (workspace)"
cargo clippy --workspace -- -D warnings

echo "[phase 2] build+test: creator CLI"

echo "[phase 2] docs (stable, deny missing docs)"
# Deny all rustdoc warnings and missing docs on public items
export RUSTDOCFLAGS="-D warnings -D missing-docs -D rustdoc::broken_intra_doc_links -D rustdoc::bare_urls"
cargo doc -p scir-core -p scir-nd -p scir-fft -p scir-signal -p scir-optimize -p scir-gpu -p scir --no-deps


echo "pre-commit: all phases completed"
