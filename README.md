# SciR

SciR aims to reimplement core pieces of SciPy in Rust with a parity-first approach and optional GPU backends.

License
- MIT. See the LICENSE file at the repository root. All crates declare `license = "MIT"` in their Cargo.toml.

Quick start
- Build workspace: `cargo build` (first run will fetch dependencies).
- Run tests: `cargo test` (GPU tests are gated and skip without GPUs).

Umbrella crate and GPU feature
- Use `crates/scir` as a convenient umbrella that re-exports core crates.
- Enable the aggregated `gpu` feature to turn on CUDA-backed paths and GPU APIs in dependent crates:
  - CPU: `cargo run -p scir --example fir_gpu`
  - CUDA: `cargo run -p scir --features gpu --example fir_gpu`
  - Elementwise demo: `cargo run -p scir --example elementwise_gpu` (add `--features gpu` to try CUDA)
  - FIR benchmark: `cargo run -p scir --example fir_bench` (add `--features gpu` to time CUDA)

GPU support overview
- GPU backends are off by default and feature-gated.
- Current CUDA coverage in `scir-gpu`: elementwise add/mul and batched FIR (f32) via CUDA Driver API + embedded PTX.
- Auto-dispatch helpers route operations to CUDA when `Device::Cuda` is selected; they fall back to CPU if the backend is unavailable.

Self-hosted GPU CI and AWS CodeBuild
- GitHub Actions for GPUs requires self-hosted runners; see `.github/workflows/gpu.yml` and `docs/gpu-runner.md` for setup and labels.
- Alternatively, use AWS CodeBuild with a CUDA-enabled container image. See:
  - `ci/docker/Dockerfile.cuda` (CUDA+Rust base image)
  - `buildspec.gpu.yml` (build/test steps)
  - `ci/codebuild/project.example.json` and `docs/codebuild-gpu.md` (project template and guide)

Parity via fixtures
- We generate SciPy fixtures and test against them. See `PLAN.md` for details.

This repository currently tracks early scaffolding work, including scripts for generating reference fixtures.

## Getting Started

Run `scripts/setup-ci-env.sh` to install prerequisites and pull the SciPy submodule at `/scipy`.
Install Python deps with `pip install -r requirements.txt` and run tests via `pytest` and `cargo test`.

If you skipped the setup script, initialize the SciPy git submodule (checked out at `/scipy`) with:

```
git submodule update --init --depth 1 scipy
```

Generate reference FFT fixtures with `python scripts/gen_fixtures.py --sizes 8 16` (files land in `fixtures/`, which is git-ignored).
Generate optimization fixtures with `python scripts/gen_optimize_fixtures.py` and signal fixtures with `python scripts/gen_signal_fixtures.py` (Butterworth, Chebyshev, Bessel, filtfilt, and `resample_poly` data). Optimization fixtures cover Nelder–Mead, BFGS, and L-BFGS results.

Fixtures are stored as `.npy` arrays. For each size `<n>`, the script produces:
- `fft_input_<n>.npy`
- `fft_output_<n>.npy`
- `ifft_output_<n>.npy`
- `rfft_output_<n>.npy`
- `irfft_output_<n>.npy`

Complex outputs use NumPy's native complex dtype. Regenerate fixtures as needed; the directory remains untracked.

## License

Dual-licensed under Apache-2.0 and MIT.
