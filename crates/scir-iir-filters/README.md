# scir-iir-filters

A SciR-maintained fork of [`iir_filters`](https://github.com/annoybot/iir_filters) (Apache-2.0).

## Why a fork

Upstream `iir_filters@0.1.3` keeps `Sos.sections` and every `SosCoeffs` field `pub(crate)`. Downstream consumers — including `scir-signal`, which `pub use`s `Sos` — can pass a designed `Sos` to `sosfilt` / `filtfilt` but cannot inspect the coefficients. That blocks any realtime DSP runtime that needs to bring its own filter loop (e.g. f32, allocation-free, per-channel-stateful, checkpointable).

This fork is a minimal change: the design code is unchanged; we widen the visibility of `Sos.sections` and `SosCoeffs`'s coefficient fields to `pub` and add an `Sos::sections()` borrow accessor. That is sufficient for `scir-signal` to expose designed coefficients to downstream users without giving up SciPy parity.

If upstream merges equivalent accessors, this crate becomes a thin re-export.

## Versioning

Tracks upstream `iir_filters@0.1.x`. Bumped to `0.1.4` to make the fork visible.

## License

Apache-2.0, same as upstream.
