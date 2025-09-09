#!/usr/bin/env bash
# Setup minimal environment for SciR CI builds.
set -euo pipefail

pip install -r requirements.txt

# Install Rust using rustup
curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
source "$HOME/.cargo/env"
rustup component add rust-src llvm-tools-preview rustfmt clippy
