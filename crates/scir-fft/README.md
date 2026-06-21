scir-fft

Overview
- FFT utilities for SciR with a parity-first approach. Uses fixtures generated from SciPy to validate `fft`, `ifft`, `rfft`, and `irfft` against reference outputs.
- `vision` exposes reusable 2D spectral image kernels: FFT magnitude, centered frequency layout, Hann windowing, log-polar resampling, Fourier-Mellin magnitude signatures, and bin-level rotation/scale pose estimates. Stream/time wrappers remain Streamz-owned; application and Spectral-Pick callers consume these kernels rather than owning the math.

Backends
- CPU: rustfft (complex) and realfft (real transforms).

Links
- Repository: https://github.com/SoftOboros/scir
- Docs: https://docs.rs/scir-fft
