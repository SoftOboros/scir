//! Window functions for short-time Fourier transform (STFT) analysis.
//!
//! Closed-form windows used to suppress spectral leakage when analyzing
//! a finite-length block of samples. Build-time consumers can emit baked
//! window tables via `scir_signal_build::Emitter::add_window`; the
//! runtime then multiplies samples by the baked array element-wise
//! before FFT, with no `cosf` call required.
//!
//! See `docs/todo/streamz/audio-spectral/TODO-ASA-01-WINDOW-BAKE.md` in
//! the consuming repo for the spec lineage governing this module.

use std::f64::consts::PI;

/// Window shape for STFT analysis.
///
/// Standards Action registration policy — adding a value requires a
/// §15 amendment to the consumer-side ASA spec lineage and is a
/// breaking change for downstream consumers (so they re-acknowledge
/// the new shape's spectral interpretation). Removing or renaming a
/// value is also a breaking change. The enum is intentionally NOT
/// `#[non_exhaustive]`; matching MUST be exhaustive at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WindowShape {
    /// `w[n] = 0.5 (1 − cos(2π n / (N−1)))`. Generic-purpose STFT;
    /// good main-lobe / side-lobe trade-off.
    Hann,
    /// 4-term Blackman-Harris (~92 dB side-lobe rejection). Preferred
    /// for high-dynamic-range analysis (THD+N, IMD).
    BlackmanHarris4Term,
    /// 5-term flat-top per the SRS Stanford Research Systems
    /// definition. Flat passband response; preferred for amplitude-
    /// accurate single-tone measurement.
    FlatTop,
}

/// Evaluate a window's closed-form definition at `length` sample points.
///
/// Returns a `Vec<f64>` of length `length`. The first element is `w[0]`
/// and the last is `w[length-1]`.
///
/// For `length == 0` returns an empty vector. For `length == 1` returns
/// `[1.0]` (degenerate but well-defined; a single sample carries no
/// shaping information).
///
/// Pure function: byte-deterministic for byte-identical inputs. Window
/// math evaluates in `f64`; consumers wanting `f32` storage can narrow
/// at emission time via `Emitter::as_f32`.
///
/// # Examples
///
/// ```
/// use scir_signal::window::{window, WindowShape};
///
/// let hann = window(WindowShape::Hann, 8);
/// assert_eq!(hann.len(), 8);
/// assert!((hann[0] - 0.0).abs() < 1e-12);
/// assert!((hann[7] - 0.0).abs() < 1e-12);
/// // Hann peaks at the midpoint.
/// assert!(hann[3] > 0.9);
/// assert!(hann[4] > 0.9);
/// ```
pub fn window(shape: WindowShape, length: usize) -> Vec<f64> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }
    let denom = (length - 1) as f64;
    (0..length)
        .map(|n| {
            let t = n as f64 / denom;
            match shape {
                WindowShape::Hann => 0.5 * (1.0 - (2.0 * PI * t).cos()),
                WindowShape::BlackmanHarris4Term => {
                    let a0 = 0.35875;
                    let a1 = 0.48829;
                    let a2 = 0.14128;
                    let a3 = 0.01168;
                    a0 - a1 * (2.0 * PI * t).cos()
                        + a2 * (4.0 * PI * t).cos()
                        - a3 * (6.0 * PI * t).cos()
                }
                WindowShape::FlatTop => {
                    let a0 = 0.21557895;
                    let a1 = -0.41663158;
                    let a2 = 0.27726316;
                    let a3 = -0.083578947;
                    let a4 = 0.006947368;
                    a0 + a1 * (2.0 * PI * t).cos()
                        + a2 * (4.0 * PI * t).cos()
                        + a3 * (6.0 * PI * t).cos()
                        + a4 * (8.0 * PI * t).cos()
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_length_returns_empty_vec() {
        assert!(window(WindowShape::Hann, 0).is_empty());
        assert!(window(WindowShape::BlackmanHarris4Term, 0).is_empty());
        assert!(window(WindowShape::FlatTop, 0).is_empty());
    }

    #[test]
    fn length_one_returns_unity() {
        assert_eq!(window(WindowShape::Hann, 1), vec![1.0]);
        assert_eq!(window(WindowShape::BlackmanHarris4Term, 1), vec![1.0]);
        assert_eq!(window(WindowShape::FlatTop, 1), vec![1.0]);
    }

    #[test]
    fn hann_endpoints_are_zero() {
        let w = window(WindowShape::Hann, 1024);
        assert!((w[0] - 0.0).abs() < 1e-12);
        assert!((w[1023] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn hann_midpoint_is_unity() {
        // For odd length, w[(N-1)/2] = 1 exactly.
        let w = window(WindowShape::Hann, 1025);
        assert!((w[512] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn hann_is_symmetric() {
        let n = 1024;
        let w = window(WindowShape::Hann, n);
        for i in 0..n / 2 {
            assert!((w[i] - w[n - 1 - i]).abs() < 1e-12);
        }
    }

    #[test]
    fn blackman_harris_4term_endpoints_match_a0_minus_sum() {
        // At t=0: a0 - a1 + a2 - a3 = 6e-5 (BH4 leakage at endpoints).
        let w = window(WindowShape::BlackmanHarris4Term, 1024);
        let expected = 0.35875 - 0.48829 + 0.14128 - 0.01168;
        assert!((w[0] - expected).abs() < 1e-12);
        assert!((w[1023] - expected).abs() < 1e-12);
    }

    #[test]
    fn blackman_harris_4term_is_symmetric() {
        let n = 1024;
        let w = window(WindowShape::BlackmanHarris4Term, n);
        for i in 0..n / 2 {
            assert!((w[i] - w[n - 1 - i]).abs() < 1e-12);
        }
    }

    #[test]
    fn flat_top_passband_is_amplitude_unity_at_center() {
        // SRS flat-top is normalized so center value (t=0.5) = a0 - a1
        // + a2 - a3 + a4 ≈ 1.0 (within SRS-coefficient rounding).
        let w = window(WindowShape::FlatTop, 1025);
        let expected = 0.21557895 - (-0.41663158) + 0.27726316 - (-0.083578947) + 0.006947368;
        assert!((w[512] - expected).abs() < 1e-12);
        assert!((w[512] - 1.0).abs() < 1e-6, "flat-top peak should be ~1.0, got {}", w[512]);
    }

    #[test]
    fn flat_top_endpoints_match_dc_sum() {
        // At t=0 all cosines are 1, so w[0] = sum of all a coefficients.
        let w = window(WindowShape::FlatTop, 1024);
        let expected = 0.21557895 + (-0.41663158) + 0.27726316 + (-0.083578947) + 0.006947368;
        assert!((w[0] - expected).abs() < 1e-12);
        assert!((w[1023] - expected).abs() < 1e-12);
    }

    #[test]
    fn flat_top_is_symmetric() {
        let n = 1024;
        let w = window(WindowShape::FlatTop, n);
        for i in 0..n / 2 {
            assert!((w[i] - w[n - 1 - i]).abs() < 1e-12);
        }
    }

    #[test]
    fn deterministic_across_calls() {
        let a = window(WindowShape::Hann, 4096);
        let b = window(WindowShape::Hann, 4096);
        assert_eq!(a, b);
    }
}
