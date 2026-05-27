scir-signal

Overview
- Signal processing for SciR: classic filter design (Butterworth, Chebyshev I, Bessel), second-order-sections filtering, zero-phase `filtfilt`, and `resample_poly`.
- Parity-first: validated against SciPy via fixtures.
- Optional GPU: FIR path can auto-dispatch to CUDA when enabled.

## Release notes

`v0.3.5` (from `v0.3.4`)

- Added notch filtering parity:
  - `iirnotch(w0, q, fs)` with SciPy-closed-form validation.
  - Preset mains rejection helpers:
    - `presets::mains_50hz_notch(fs)`
    - `presets::mains_60hz_notch(fs)`
    - `presets::mains_50hz_notch_with_q(fs, q)`
    - `presets::mains_60hz_notch_with_q(fs, q)`
- Added explicit Bessel normalization control with `BesselNorm::{Phase, Delay, Mag}` and
  `bessel_filter_with_norm(order, norm, kind, fs)`.
- Added `WindowShape` and `window(shape, length)` utilities for STFT taper generation.

Links
- Repository: https://github.com/SoftOboros/scir
- Docs: https://docs.rs/scir-signal
