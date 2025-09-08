# Copy of codex environment configured startup.sh
# AGENTS: modify this when modifying the environment with network enabled.
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
    libsdl2-dev \
    xvfb \
    libxrender1 \
    libfreetype6-dev \
    libx11-dev \
    libxext-dev \
    libgtk-3-dev \
    sccache \
    pkg-config \
    librlottie-dev \
    && rm -rf /var/lib/apt/lists/*
git submodule update --init --recursive --depth 1
curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly
pip install -r requirements.txt
rustup component add rust-src llvm-tools-preview rustfmt clippy
rustup target add thumbv7em-none-eabihf
echo 'export RUSTC_WRAPPER=$(which sccache)' >> ~/.bashrc
export RUSTC_WRAPPER=$(which sccache)
