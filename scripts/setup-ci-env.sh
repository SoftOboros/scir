#!/usr/bin/env bash
# Setup minimal environment for SciR CI builds.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
git -C "$repo_root" submodule update --init --depth 1 scipy
pip install -r "$repo_root/requirements.txt"

# Install Rust using rustup
curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
source "$HOME/.cargo/env"
rustup component add rust-src llvm-tools-preview rustfmt clippy
