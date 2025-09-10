scir-gpu

Overview
- GPU foundations for SciR. Provides a minimal `DeviceArray<T>` abstraction, CPU baselines, and optional CUDA paths behind the `cuda` feature.

Features
- cuda: enables CUDA Driver API usage with embedded PTX kernels for f32 elementwise add/mul and batched FIR.

Requirements (CUDA)
- NVIDIA GPU with recent driver installed (libcuda present on host).
- On Linux, ensure `libcuda.so` is visible to the container or process; on Windows, `nvcuda.dll` is required.

Quick start
- CPU tests: `cargo test -p scir-gpu`
- CUDA tests: `cargo test -p scir-gpu --features cuda`

APIs
- `DeviceArray<T>`: shaped arrays with `device` and `dtype`. CPU-backed storage today.
- Elementwise: `add_scalar_auto`, `mul_scalar_auto`, `add_auto` (f32) dispatch to CUDA when available.
- FIR: `fir1d_batched_f32_auto(x, taps, device)` chooses CUDA or CPU and falls back to CPU if CUDA is unavailable.

