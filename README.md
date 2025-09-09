# SciR

SciR aims to reimplement core pieces of SciPy in Rust with a parity-first approach and future GPU backends.

This repository currently tracks early scaffolding work, including scripts for generating reference fixtures.

## Getting Started

Run `scripts/setup-ci-env.sh` to install prerequisites.
Install Python deps with `pip install -r requirements.txt` and run tests via `pytest` and `cargo test`.

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
