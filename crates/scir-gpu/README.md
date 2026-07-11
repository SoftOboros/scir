scir-gpu

Overview
- GPU foundations for SciR. Provides a minimal `DeviceArray<T>` abstraction, CPU baselines, and optional CUDA and wgpu paths behind the `cuda` and `wgpu` features respectively.

Features
- cuda: enables CUDA Driver API usage with embedded PTX kernels for f32 elementwise add/mul and batched FIR.
- wgpu: enables the portable-compute backend (Vulkan/Metal/DX12 via `wgpu`) with WGSL kernels for f32 elementwise add and 2D image resize (`resize2d`, nearest/bilinear). Adds `wgpu`, `pollster` (blocks on wgpu's async adapter/device requests to match this crate's synchronous dispatch style), and `bytemuck` (buffer byte-casting) as dependencies.

Requirements (CUDA)
- NVIDIA GPU with recent driver installed (libcuda present on host).
- On Linux, ensure `libcuda.so` is visible to the container or process; on Windows, `nvcuda.dll` is required.

Requirements (wgpu)
- Any GPU with a Vulkan, Metal, or DX12 driver (AMD/Intel/NVIDIA all work) — no vendor SDK install required. `wgpu_backend::adapter_name()` logs the adapter wgpu selects, useful for confirming which GPU picked up the workload on a multi-GPU host.

Quick start
- CPU tests: `cargo test -p scir-gpu`
- CUDA tests: `cargo test -p scir-gpu --features cuda`
- wgpu tests: `cargo test -p scir-gpu --features wgpu`

APIs
- `DeviceArray<T>`: shaped arrays with `device` and `dtype`. CPU-backed storage today.
- Elementwise: `add_scalar_auto`, `mul_scalar_auto`, `add_auto` (f32) dispatch to CUDA or wgpu when available (`add_auto` is the only op with a wgpu kernel so far; the scalar ops fall back to CPU under `wgpu`).
- FIR: `fir1d_batched_f32_auto(x, taps, device)` chooses CUDA or CPU and falls back to CPU if CUDA is unavailable. No wgpu FIR kernel yet.
- Resize: `DeviceArray<f32>::resize2d_f32(out_h, out_w, mode)` (CPU baseline, `self.shape` = `[height, width]`) and `resize2d_auto(out_h, out_w, mode, device)` (CUDA falls back to CPU — no CUDA kernel yet; wgpu has a real kernel). `ResizeMode::{Nearest, Bilinear}`. Both paths use identical half-pixel-center source-coordinate mapping so GPU/CPU parity holds a tight tolerance (see `tests` in `src/lib.rs`).

Tests always verify GPU output against the CPU baseline within tolerance (PLAN.md §6.4) rather than trusting the GPU path in isolation; each GPU test skips gracefully (with a message) if no compatible adapter/device is found at runtime.

