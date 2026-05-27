# scir-iir-filters

A SciR-maintained Apache-2.0 fork of [`iir_filters`](https://github.com/annoybot/iir_filters), permanently maintained as a **superset** of upstream so SciPy parity remains an explicit goal.

## Why a fork

Upstream `iir_filters@0.1.3` keeps `Sos.sections` and every `SosCoeffs` field `pub(crate)`. Downstream consumers — including `scir-signal`, which `pub use`s `Sos` — can pass a designed `Sos` to `sosfilt` / `filtfilt` but cannot inspect the coefficients. That blocks any realtime DSP runtime that needs to bring its own filter loop (e.g. f32, allocation-free, per-channel-stateful, checkpointable).

The initial fork pass is a minimal divergence: the design code is unchanged; we widen the visibility of `Sos.sections` and `SosCoeffs`'s coefficient fields to `pub` and add `Sos::sections()` and `Sos::to_arrays()` accessors. Future passes will keep adding accessors, design surface, and SciPy-parity coverage as the SciR ecosystem needs them — see SciR's signal-processing roadmap for what lands next.

## Maintenance posture

This crate is **not** intended to converge with upstream and disappear. Even if `annoybot/iir_filters` accepts equivalent accessors, scir-iir-filters remains the SciR-controlled implementation:

- SciR's SciPy-parity contract requires fixture-driven validation against SciPy reference outputs across every API surface. The fork is the place that contract lives in SciR's tree.
- Downstream SciR crates (notably `scir-signal`) depend on this crate's path; pulling those deps back to a registry version creates an external coordination point for every SciPy-parity bug fix.
- The fork is the cheapest place to land additive accessors, lints, edition bumps, and `no_std` work without negotiating each one upstream.

When upstream lands compatible improvements, the fork **rebases or cherry-picks** them in — but maintains the SciR-side delta as a permanent superset.

## Cargo features

| Feature | Default | What it does |
|---|---|---|
| `std` | yes | Pulls `thiserror` for the `Error` derive and adds the standard `impl std::error::Error for Error`. Inherent `f64` transcendentals come from `std`. |
| (none) | — | `--no-default-features` builds for `no_std + alloc` targets (e.g. `thumbv7em-none-eabihf`, `riscv32imac-unknown-none-elf`). `Display` is hand-written; `f64`/`Complex<f64>` transcendentals route through `num-complex`'s `libm` feature. All public API surface (design, ZPK→SOS, biquad filter loop) is identical across both modes. |

The `no_std + alloc` mode is the wire-up point for SciR's static-overlay / build-time coefficient baking story — it lets `scir-signal-build`-generated tables be consumed (or re-derived) on embedded targets without dragging in `std`.

## Versioning

Tracks upstream `iir_filters@0.1.x` lineage but versions independently under SciR's release cadence. Releases:
- `0.1.4` shipped with SciR `v0.3.3` (initial fork — public Sos accessors).
- `0.1.5` shipped with SciR `v0.3.4` (added `cheb1ap` + `cheby1` design path, all four FilterTypes for both prototypes).
- `0.1.6` shipped with SciR `v0.3.4` (added `no_std + alloc` support via the `std` default-feature; `FilterType` derives `Clone, Copy`; added `besselap` + `bessel` via SciPy/mpmath-precomputed phase-norm pole tables for orders 1..=25).
- `0.1.7` shipped with SciR `v0.3.5` (added `BesselNorm::{Phase, Delay, Mag}` and
  per-order precomputed pole + gain tables for `Delay` and `Mag` normalization options).

## License

Apache-2.0, same as upstream. Original copyright preserved (`https://github.com/annoybot`); SciR additions copyright Softoboros Technology Inc.
