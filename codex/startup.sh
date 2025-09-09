#!/usr/bin/env bash
set -euo pipefail
# SciR development environment setup script
# AGENTS: modify this when changing environment packages with network enabled.
apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    nano \
    cmake \
    ninja-build \
    llvm-dev \
    libclang-dev \
    clang \
    mold \
    sccache \
    pkg-config \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
source "$HOME/.cargo/env"
pip install -r requirements.txt
rustup component add rust-src llvm-tools-preview rustfmt clippy

echo 'export RUSTC_WRAPPER=$(which sccache)' >> ~/.bashrc
export RUSTC_WRAPPER=$(which sccache)
