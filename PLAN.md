# SciR (SciPy in Rust) — Development Blueprint (Reviewed & Extended)

> **Status:** Reviewed for consistency with the thread; hallucinations corrected; clarified where thin; CI + phase checklists added.

## 0) Upstream Reference & Constraints (Source of Truth)
- **Upstream project:** SciPy — GitHub: `scipy/scipy`.
- **License:** BSD‑3‑Clause (permissive). We **must not** copy SciPy C/Fortran code; we re‑implement algorithms and validate behavior via tests/fixtures.
- **GPU stance upstream:** SciPy itself is **CPU‑only**; GPU acceleration in Python is typically via **CuPy** or other frameworks (JAX, PyTorch). This is our differentiation target.
- **Test coverage upstream:** No official, single coverage % is published; coverage is broad and actively maintained. Our flow treats SciPy **behavior** as the spec via fixtures.

---

## 1) Goals & Differentiation
- **Parity-first:** Reproduce well-known SciPy APIs (starting with `fft`, `signal`, `optimize`, `linalg`) with Rust ergonomics and explicit types.
- **Mechanical conversion via tests:** Use SciPy to generate canonical **fixtures**; run identical Rust implementations against them. Python behavior becomes the **oracle**.
- **Performance runway:** After parity, add **Rust-native optimizations** (slices, SIMD, Rayon) and opt‑in **GPU backends** (CUDA and/or portable compute via `wgpu`).
- **Marketing edge:** “SciR: SciPy rebuilt for Rust — safe, fast, GPU‑ready.”

---

## 2) Workspace & Module Map (Adjusted for Clarity)
```
scir/                              # Umbrella crate (re-exports)
  Cargo.toml
  crates/
    scir-core/                     # Common: traits, errors, types, tolerance helpers
    scir-nd/                       # ndarray interop & conversions (CPU baseline)
    scir-fft/                      # FFT APIs (CPU: rustfft; GPU backends optional)
    scir-signal/                   # Filters (IIR/FIR), resampling, windows
    scir-optimize/                 # minimize (Nelder–Mead, BFGS/L-BFGS), least_squares
    scir-linalg/                   # Linear algebra (faer / ndarray-linalg / BLAS)
    scir-gpu/                      # (optional) backends & array device mgmt (see §6)
```
**Notes**
- `scir-nd` isolates `ndarray`/shape/stride conversions to keep compute crates focused.
- `scir-gpu` is a *thin* layer: device memory, transfers, and backend dispatch. Compute lives in feature‑gated submodules of each domain crate.

---

## 3) External Dependencies (Corrected & Conservative)
- **Arrays:** `ndarray` (CPU baseline). Consider `ndarray-linalg` for Lapack-backed ops.
- **FFT:** `rustfft` (CPU). (GPU path is custom, see §6.)
- **Linalg (CPU):** `faer` (pure Rust) and/or `ndarray-linalg` (BLAS/LAPACK via `openblas-src`/system). Gate behind features.
- **Parallel:** `rayon`.
- **Numeric types:** `num-traits`, `num-complex`.
- **SIMD:** prefer crates like `wide` (portable); track Rust `portable_simd` stabilization for future.
- **Serde:** optional (`serde`, `serde_json`, `ndarray-npy`) for fixtures.
- **GPU:** see §6. Avoid claiming a specific cuFFT crate; we may write minimal FFI if no maintained bindings exist.

---

## 4) Error Model & API Surface
- One public error enum per crate re-exported through `scir-core`:
```rust
#[derive(thiserror::Error, Debug)]
pub enum ScirError {
    #[error("invalid argument: {0}")] InvalidArgument(String),
    #[error("numerical failure: {0}")] NumericalFailure(String),
    #[error("backend not available: {0}")] BackendUnavailable(String),
}
```
- Functions use **explicit structs** for options to avoid ambiguity (e.g., `FftPlan`, `FilterDesignOpts`, `MinimizeOpts`).
- Complex numbers: `num_complex::Complex<f64>` / `<f32>` with type aliases.
- **Naming:** mirror SciPy function names in `snake_case` and group by module (`scir::signal::filtfilt`, etc.).

---

## 5) Parity via Fixtures (Detailed & Unambiguous)
### 5.1 Fixture Generation (Python)
- Pin versions to ensure deterministic results: e.g., `numpy==1.x`, `scipy==1.y`.
- For each target function, create a Python script that:
  1) Generates representative inputs, including edge cases.
  2) Calls SciPy implementation.
  3) Saves **inputs** and **expected outputs** to `fixtures/` as `.npy` or JSON.
- **Complex data** in JSON: encode as `[re, im]` tuples or use `.npy` to avoid custom encodings.
- **Example:** (pseudo‑snippet)
```python
# export_fixtures_fft.py
import numpy as np
from scipy.fft import fft, ifft, rfft, irfft
np.random.seed(0)
x = np.random.randn(1024).astype(np.float64)
np.save('fixtures/fft/x_f64.npy', x)
np.save('fixtures/fft/fft_expected.npy', fft(x))
```

### 5.2 Parity Testing (Rust)
- Each crate loads its fixtures and asserts tolerances:
```rust
let x: Array1<f64> = read_npy("fixtures/fft/x_f64.npy")?;
let y_ref: Array1<Complex64> = read_npy("fixtures/fft/fft_expected.npy")?;
let y = scir::fft::fft(x.view());
assert_close!(y, y_ref, atol=1e-9, rtol=1e-7);
```
- Provide a macro `assert_close!` in `scir-core` for consistent numeric tolerances.
- **Signal-specific cautions:** `filtfilt` edge handling (default padding scheme in SciPy is critical):
  - `padtype="odd"`, `padlen = 3 * (max(len(a), len(b)) - 1)` unless overridden.
  - Use **Gustafsson** method parity where SciPy does.

### 5.3 Fixture Scope
- `fft`: real/complex, sizes prime/power‑of‑two, shifts.
- `signal`: `butter/cheby1/bessel` design → SOS, `sosfilt`, `filtfilt`, `resample_poly`.
- `optimize`: `rosenbrock`, `himmelblau` with known minima; `least_squares` datasets.
- `linalg`: small/medium problems for `svd`, `qr`, `solve`, conditioned matrices.

---

## 6) GPU Extension (Corrected & Grounded)
**Reality check:** GitHub‑hosted runners generally do **not** expose GPUs. GPU CI requires **self‑hosted** runners or third‑party CI with GPU. Our plan reflects that.

### 6.1 Backend Strategy
- **CUDA (NVIDIA):** Use `cudarc` (if sufficient for our needs) or implement minimal FFI to cuBLAS/cuFFT if maintained crates are unavailable. We will not claim an existing crate until verified; plan for FFI via `bindgen` as fallback.
- **Portable compute:** `wgpu` (Vulkan/Metal/DX12/WebGPU) with WGSL compute kernels. FFT on `wgpu` is non‑trivial; start with pointwise ops, convolutions, GEMM; explore radix‑N FFT kernels as a later milestone.
- **OpenCL:** `ocl` or `opencl3` as an alternative where CUDA is not available.

### 6.2 Device Array Abstraction
Instead of storing raw `wgpu::Buffer` in a generic enum, we define **shaped device arrays** with dtype and strides to mirror CPU semantics:
```rust
pub enum Device {
    Cpu,
    Cuda(CudaCtx),
    Wgpu(WgpuCtx),
}

pub struct DArray {
    pub device: Device,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub dtype: DType,           // F32, F64, C64, C128
    // opaque handle: CPU -> ndarray storage; GPU -> backend-specific buffer
}
```
- Provide transfers `to_cpu()`/`to_device(Device)` with explicit synchronization.
- High‑level APIs accept CPU arrays; GPU paths are **opt‑in** via feature flags and explicit conversions.

### 6.3 GPU Targets & Order
1) **Elementwise / map-reduce** (easy win): scale, add, mul, abs2, reductions.
2) **Convolution / FIR** (1D/2D): batched FIR via tiling; later IIR (with feedback) using parallel prefix or block processing.
3) **GEMM**: leverage cuBLAS (CUDA) or tuned WGSL kernels.
4) **FFT**: CUDA path first (via cuFFT FFI); `wgpu` FFT later (requires custom kernels).

### 6.4 Validation
- Always verify **GPU == CPU** within tolerances.
- Provide deterministic seeds and sized workloads to make cross‑backend comparisons reliable.

---

## 7) CPU Optimizations (Pragmatic)
- **Slicing/borrowing:** prefer `&[T]`/`&mut [T]` hot paths; adapt to `ndarray` views for API.
- **SIMD:** use `wide` or backend‑specific intrinsics; gate with features.
- **Parallel:** `rayon` for embarrassingly parallel operations.
- **Numerical stability:** match SciPy defaults (e.g., bilinear transforms, SOS scaling) to reduce drift.

---

## 8) Benchmarks & Reproducibility
- **`criterion`** benchmarks mirroring fixtures (FFT sizes, filter lengths, optimizer dimensions).
- Compare against **SciPy** (CPU) on the same machine for a reality check; document hardware and versions.
- Record baseline throughput/latency; track regressions via CI trend artifacts.

---

## 9) CI & Platform Testing (Revised & Actionable)
### 9.1 Matrix (CPU)
- **OS:** ubuntu‑latest, macos‑latest, windows‑latest.
- **Rust:** stable, beta, nightly.
- **Features:** `pure-rust` (default), `faer`, `blas`.

### 9.2 GitHub Actions (CPU) — Example
```yaml
name: ci
on: [push, pull_request]

jobs:
  cpu:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, beta, nightly]
        features: ["", "faer", "blas"]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { toolchain: ${{ matrix.rust }} }
      - uses: Swatinem/rust-cache@v2
      - run: cargo build --verbose --features "${{ matrix.features }}"
      - run: cargo test  --verbose --features "${{ matrix.features }}"
```

### 9.3 GPU CI (Self‑Hosted)
```yaml
  gpu:
    if: github.repository_owner == 'YOUR_ORG'  # avoid forks
    runs-on: [self-hosted, linux, x64, gpu, nvidia]
    env:
      RUST_LOG: info
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { toolchain: stable }
      - run: nvidia-smi  # sanity check
      - run: cargo test --features cuda --package scir-fft
```
- Provide docs for setting up the **self-hosted GPU runner** (CUDA drivers, toolkit, labels).
- For `wgpu` tests on macOS with M‑series, run a self‑hosted macOS runner.

### 9.4 Python Parity in CI
- Run Python once to **(re)generate fixtures** (pinned versions) and upload as CI artifact used by Rust jobs.
```yaml
  fixtures:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: python -m pip install --upgrade pip
      - run: pip install numpy==1.X scipy==1.Y pytest
      - run: python tools/export_fixtures.py
      - uses: actions/upload-artifact@v4
        with:
          name: scir-fixtures
          path: fixtures/**
```
- CPU/GPU jobs **download** the `scir-fixtures` artifact before `cargo test`.

### 9.5 Coverage & Reporting
- Use `cargo-llvm-cov` for Rust coverage; `pytest --cov` for Python harness if needed.
- Publish with `codecov/codecov-action@v4` (optional).

---

## 10) Licensing & Compliance (Explicit, Actionable)

> **Intent:** Keep `scir` legally clean and enterprise‑adoptable while leveraging SciPy as a *behavioral* oracle.

### 10.1 Code Provenance (What is / isn’t allowed)
- **Do NOT copy** SciPy implementation code (C/Fortran/Python) or translate line‑by‑line.
- **OK:** Implement algorithms from **papers, standards, and SciPy docs** (concepts ≠ code). Cite sources in `ALGORITHMS.md`.
- **OK:** Generate and store **test fixtures** by running SciPy; fixtures are *data*, not code. Keep provenance (see 10.3).
- **OK:** Use third‑party crates with **permissive licenses** (MIT/Apache‑2.0/BSD‑2/3/ISC/Zlib). Avoid strong/viral copyleft.

### 10.2 Licensing for `scir`
- Pick **one** license for all crates (recommend **Apache‑2.0** for patent grant; MIT also acceptable). Stay consistent repo‑wide.
- Add SPDX headers to every source file:
  ```rust
  // SPDX-License-Identifier: Apache-2.0
  ```
- `Cargo.toml` per crate:
  ```toml
  license = "Apache-2.0"
  # or: license = "MIT"
  ```
- Top‑level `LICENSE` file matching the chosen license; no per‑crate divergence unless absolutely necessary.

### 10.3 Fixture Policy (SciPy as Oracle)
- **Store with provenance** alongside each fixture batch (YAML or JSON):
  ```yaml
  generator: tools/export_fixtures.py
  python: 3.11.x
  numpy: 1.x.y
  scipy: 1.y.z
  seed: 123456
  os: linux-x86_64
  date: 2025-09-08
  script_sha256: <hash-of-exporter>
  ```
- Keep **regeneration scripts** in `tools/` and pin dependency versions.
- Use **synthetic data** for tests; avoid copyrighted datasets unless licenses are compatible and recorded.

### 10.4 Submodules & Vendoring
- Install SciPy via `pip` for local testing/reference; avoid git submodules unless absolutely necessary.
- **Do NOT vendor** SciPy code into `scir` crates. Keep Python tooling under `tools/` and out of publishable crates.
- Ensure crates publish **without** SciPy sources (check `include`/`exclude` in `Cargo.toml`).

### 10.5 Third‑Party Dependencies
- Maintain `license-allowlist.toml` and enforce via CI (`cargo-deny`). Suggested allowlist: `Apache-2.0`, `MIT`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `Zlib`.
- **Avoid** GPL/AGPL. If LGPL/MPL is unavoidable, gate behind **non‑default** feature(s) and document obligations clearly.
- BLAS/LAPACK: prefer **faer** (pure Rust) or **OpenBLAS**/**BLIS** via `ndarray-linalg` (permissive). Do **not** hard‑link to GPL libraries.

### 10.6 GPU Backends & Vendor SDKs
- **CUDA:** Link via FFI; do **not** bundle NVIDIA headers or binaries. Require users to install CUDA under NVIDIA’s EULA; detect at build time.
- **wgpu/OpenCL:** Verify license compatibility; prefer permissive crates. Keep backend code in optional features (`cuda`, `wgpu`, `opencl`).
- Runtime selection must **not** force inclusion of proprietary components in default builds.

### 10.7 Documentation, Examples & Trademarks
- Do **not** copy SciPy docs verbatim. Paraphrase and attribute: “Behavior aligned with SciPy 1.y.z.” Quote only short fragments when necessary with attribution.
- **Branding:** Do **not** imply affiliation. Use a disclaimer in README: “`scir` is not affiliated with or endorsed by the SciPy project.” Avoid using the **SciPy** name in package names.
- Consider licensing documentation under **CC‑BY‑4.0** (optional) and code under Apache‑2.0/MIT; document this split in `README`.

### 10.8 Contributor IP Policy
- Use **DCO** (Developer Certificate of Origin) or a simple CLA. Require `Signed-off-by` in commits via CI.
- PR template must include:
  - Confirmation of **original work** (no copied code from SciPy or other restricted sources).
  - Acknowledgment of project license and fixture policy.
  - Checklist for adding SPDX headers and updating `ALGORITHMS.md` references.

### 10.9 Automated Compliance in CI
- Add `cargo-deny` step (license + bans + sources). Fail CI on violations.
- Add **REUSE**/SPDX validation (e.g., `fsfe/reuse-action`) to ensure all files carry SPDX IDs.
- Ensure `cargo package --list` shows no SciPy or vendor SDK files are included.

### 10.10 Release Hygiene
- Release artifacts must include: `LICENSE`, `NOTICE` (if required), `ALGORITHMS.md`, fixture provenance files, and machine‑readable `CITATION.cff` (optional but useful for academia).
- Changelog entries must note any new dependencies and their licenses.

### 10.11 Quick Compliance Checklist (per PR)
- [ ] No copied upstream code; sources cited in `ALGORITHMS.md`.
- [ ] Fixtures regenerated with pinned SciPy/Numpy; provenance updated.
- [ ] SPDX headers added/maintained; `Cargo.toml` `license` set.
- [ ] `cargo-deny` passes; no disallowed licenses.
- [ ] GPU/vendor SDK usage gated behind non‑default features; no bundled binaries/headers.
- [ ] README contains affiliation disclaimer and licensing summary.
- [ ] `cargo package --list` audited; publishable crates clean of Python/vendor code.

## 11) Versioning & Publishing Versioning & Publishing
- Semantic versioning per crate; synchronize via a workspace release script.
- Publish crates incrementally (`scir-core` → `scir-fft` → `scir-signal` …) as functionality lands.
- Tag releases and attach benchmark summaries and parity status.

---

## 12) Roadmap (Phases with Outcomes)
**Phase 1 — Parity skeletons (CPU):**
- `scir-core`, `scir-nd`, `scir-fft` (fft/ifft/rfft/irfft), basic `signal` windows.
- Fixtures for FFT + simple filters; CI (CPU) running.
- **Outcome:** Green tests vs SciPy for FFT; >80% crate coverage.

**Phase 2 — Signal & Optimize:**
- Filter design (`butter`, `cheby1`, `bessel` → SOS), `sosfilt`, `filtfilt`, `resample_poly`.
- `optimize::minimize` (Nelder–Mead, BFGS/L‑BFGS) on Rosenbrock/Himmelblau.
- **Outcome:** Parity on core signal routines; optimizer matches SciPy within tolerances.

**Phase 3 — Linalg & Backends:**
- Introduce `faer` backend; optional BLAS/LAPACK features via `ndarray-linalg`.
- Initial benchmarks; fuzz harness for edge cases.
- **Outcome:** Stable linalg APIs; CI matrix covers faer/blas features.

**Phase 4 — GPU Foundations:**
- Device array abstraction; elementwise + batched FIR on CUDA and/or `wgpu`.
- Self‑hosted GPU CI; parity checks CPU↔GPU.
- **Outcome:** First GPU win (measurable speedup) with identical API.

**Phase 5 — Advanced GPU & FFT:**
- cuFFT FFI path for CUDA; exploratory `wgpu` FFT kernels.
- GEMM via cuBLAS (CUDA) or tuned WGSL; integrate into `linalg`.
- **Outcome:** Significant GPU acceleration for FFT/signal/GEMM.

**Phase 6 — Optimization & Hardening:**
- SIMD, cache-aware blocking, kernel fusion where applicable.
- Documentation, examples, auto-generated API docs; publish v0.1–v0.3.

---

## 13) Developer Experience (DX) & Contrib
- `just` or `make` targets: `just fixtures`, `just test`, `just bench`.
- `tools/` with Python exporters; pin via `requirements.lock` or `uv`.
- `CONTRIBUTING.md` with style guide, tolerance policy, and fixture provenance rules.

---

## 14) Open Questions (Tracked)
- Which GPU backend to prioritize first (CUDA market share vs `wgpu` portability)?
- Where do we draw the SciPy API boundary (full parity vs pragmatic subset)?
- Tolerance policy per function/domain (documented in `scir-core`).

---

## 15) Phase Completion Checklists

### Phase 1 — CPU FFT Baseline
- [ ] `scir-core` error/tolerance utilities.
- [ ] `scir-nd` ndarray bridges, `.npy` I/O helpers.
- [ ] `scir-fft`: fft/ifft/rfft/irfft (+ shift helpers).
- [ ] Python fixtures for FFT (sizes: 64…65536; real & complex).
- [ ] CI (CPU matrix) green; coverage report uploaded.

### Phase 2 — Signal & Optimize
- [ ] `scir-signal`: `butter`, `cheby1`, `bessel` → SOS; `sosfilt`, `filtfilt`, `resample_poly`.
- [ ] Fixture parity for filter design & filtfilt edge cases.
- [ ] `scir-optimize`: Nelder–Mead, BFGS/L‑BFGS with line search.
- [ ] Optimizer fixtures (Rosenbrock, Himmelblau) pass within tolerances.

### Phase 3 — Linalg & Backends (CPU)
- [ ] `faer` path implemented; `ndarray-linalg` optional behind feature.
- [x] Solve/SVD/QR minimal set with fixtures.
- [x] Benchmarks for FFT, filters, solve.

### Phase 4 — GPU Foundations
- [ ] `scir-gpu` device array abstraction + transfers.
- [ ] CUDA elementwise + batched FIR with parity vs CPU.
- [ ] Self‑hosted GPU CI job green; docs for runner setup.

### Phase 5 — GPU FFT & GEMM
- [ ] cuFFT FFI path wired for 1D FFT; parity vs CPU fixtures.
- [ ] GEMM via cuBLAS (CUDA) or WGSL kernels; linalg integration.
- [ ] Cross‑backend tests (CPU↔CUDA↔WGPU) within tolerances.

### Phase 6 — Optimization & Docs
- [ ] SIMD paths (wide or equivalent) for hot loops.
- [ ] Criterion benchmarks stable; performance notes recorded per release.
- [ ] Docs site published; examples and fixture provenance documented.
- [ ] v0.1…v0.3 releases tagged and crates published.

---

**TL;DR:** The plan now mirrors the thread’s intent, corrects GPU/CI realities (self‑hosted requirement), avoids speculative crates, nails fixture‑driven parity, and adds clear phase gates so this can feel like “shooting fish in a barrel” — with receipts in CI.


## Progress Log
- Initial scaffolding: README, AGENTS, SciPy submodule, script updates.
- Added numpy/scipy deps and bootstrap scir-core/nd crates.

- Implemented assert_close! macro and ndarray Vec conversion helpers.
- Created sample fixture generation script and ignored output in git.
- Documented Python dependency installation steps in README.
- Added slice/complex support to `assert_close!` and parameterized fixture generator.
- Extended `assert_close!` to work with ndarray arrays and documented submodule init in README.
- Added array-based tests in `scir-nd` and switched fixtures to `.npy` with documented format.

- Created scir-fft crate with real FFT and fixture-based parity tests.
- Added CONTRIBUTING guide for contributors.
- Implemented inverse FFT with fixtures for multiple sizes and noted style guidelines in CONTRIBUTING.
- Added parity tests for FFT and IFFT across multiple fixture sizes.
- Added real FFT (rfft/irfft) routines with multi-size fixtures and tests.
- Introduced scir-signal crate with Butterworth design and sosfilt parity.
- Added scir-optimize crate implementing Nelder–Mead and BFGS validated on Rosenbrock and Himmelblau fixtures.
- Extended signal fixtures to include Chebyshev and Bessel designs with Rust APIs and filtfilt scaffolding.
- Added zero-phase filtering and `resample_poly` fixtures with Rust parity tests.
- Implemented L-BFGS optimizer and fixtures for Rosenbrock and Himmelblau.
- Phase 2 complete: core signal routines and L-BFGS optimizer validated.
- Added pre-commit hooks and updated contributing guidelines.
- Removed SciPy git submodule; pip-installed SciPy satisfies fixture generation.
- Phase 3 kick-off: reintroduced SciPy git submodule at `/scipy` and updated documentation references.
- Updated CI setup script to initialize the SciPy submodule automatically.
- Documented submodule path as `/scipy` and made setup script path-agnostic.
 - Added `scir-linalg` crate with BLAS/LAPACK path via `ndarray-linalg` and feature flags for `faer` backend.
 - Implemented `solve`, `svd`, and `qr` APIs with fixture-based tests (BLAS feature).
 - Added `scripts/gen_linalg_fixtures.py` to generate linalg fixtures (`lin_solve_*`, `svd_A.npy`, `qr_A.npy`).
 - Prepared feature gating for `faer` backend (placeholders); enable in a follow-up.
