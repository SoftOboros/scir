GPU CI Runner (Self-hosted) Setup

Overview
- GitHub-hosted runners do not expose GPUs. Use a self-hosted machine with an NVIDIA GPU for CUDA jobs or an Apple Silicon/macOS machine for Metal.

Linux (CUDA) Steps
- Install NVIDIA driver and CUDA Toolkit matching your GPU (e.g., CUDA 12.x).
- Ensure `libcuda.so` (Linux) or `nvcuda.dll` (Windows) is available in the dynamic library search path (CUDA Driver API is required for PTX loading).
- Create a dedicated runner with labels: gpu, nvidia, linux, x64.
- Install Rust toolchain (rustup) and ensure `cargo` is in PATH.
- Verify with `nvidia-smi` and `nvcc --version` (optional for FFI builds).
- Run the GitHub Actions runner service and attach the labels above.

macOS (Metal) Steps (for future wgpu tests)
- Use macOS with Apple Silicon.
- Install Rust toolchain.
- Label runner: gpu, macos, arm64, metal.

Repository Configuration
- GPU jobs are defined in `.github/workflows/gpu.yml` and target self-hosted runners by label.
- CUDA-dependent features are behind `--features cuda` and are off by default.

Validation
- Run `cargo test -p scir-gpu` to validate CPU-backed abstractions.
- For CUDA builds (future), run `cargo test -p scir-gpu --features cuda` on a CUDA-capable runner.

AWS CodeBuild (GPU) Option
- Create a CodeBuild project using a custom ECR image based on an NVIDIA CUDA base image (e.g., `nvidia/cuda:12.X-devel-ubuntu22.04`) with Rust toolchain installed.
- Enable privileged mode in the environment settings so the container can access GPUs (NVIDIA runtime on the host must be configured).
- Choose a GPU-enabled compute configuration as supported by your AWS account and region.
- Add this repo’s `buildspec.gpu.yml` to the project; it runs `cargo build/test` with `--features cuda`.
- Ensure the host AMI and CodeBuild fleet support GPUs and the CUDA driver is installed on the underlying host.
