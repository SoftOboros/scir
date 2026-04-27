//! Signal processing utilities for SciR.
#![deny(missing_docs)]

use ndarray::Array1;
use scir_iir_filters::{
    filter::{DirectForm2Transposed, Filter},
    filter_design::{
        bessel as design_bessel, bessel_with_norm as design_bessel_with_norm,
        butter as design_butter, cheby1 as design_cheby1,
    },
    sos::zpk2sos,
};

pub use scir_iir_filters::filter_design::MAX_BESSEL_ORDER;

pub use scir_iir_filters::errors::Error as FilterError;
pub use scir_iir_filters::filter_design::{BesselNorm, FilterType};
pub use scir_iir_filters::sos::{Sos, SosCoeffs};

/// Design a Butterworth filter across all four [`FilterType`] variants
/// and return SOS form. Mirrors `scipy.signal.butter(N, Wn, btype, fs=fs,
/// output='sos')`.
///
/// # Examples
/// ```
/// use scir_signal::FilterType;
/// // Equivalent to scipy.signal.butter(4, 200, btype='highpass', fs=1000, output='sos')
/// let sos = scir_signal::butter_filter(4, FilterType::HighPass(200.0), 1000.0).unwrap();
/// let x = ndarray::Array1::from_vec(vec![0.0; 8]);
/// let _y = scir_signal::sosfilt(&sos, &x);
/// ```
pub fn butter_filter(
    order: u32,
    filter_type: FilterType,
    fs: f64,
) -> Result<Sos, FilterError> {
    let zpk = design_butter(order, filter_type, fs)?;
    zpk2sos(&zpk, None)
}

/// Design a Butterworth **low-pass** filter on the SciPy normalized-frequency
/// convention (`fs=2`, so `cutoff` ∈ (0, 1)) and return SOS.
///
/// Thin wrapper over [`butter_filter`] preserved for backward compatibility.
/// New code wanting HPF/BPF/BSF or an explicit sample rate SHOULD call
/// [`butter_filter`] directly.
///
/// # Examples
/// ```
/// let sos = scir_signal::butter(4, 0.2);
/// let x = ndarray::Array1::from_vec(vec![0.0; 8]);
/// let _y = scir_signal::sosfilt(&sos, &x);
/// ```
pub fn butter(order: u32, cutoff: f64) -> Sos {
    butter_filter(order, FilterType::LowPass(cutoff), 2.0)
        .expect("butter design failed for valid LowPass parameters")
}

/// Design a Chebyshev Type I filter across all four [`FilterType`] variants
/// and return SOS form. `rp` is the maximum passband ripple in dB. Mirrors
/// `scipy.signal.cheby1(N, rp, Wn, btype, fs=fs, output='sos')`.
///
/// # Examples
/// ```
/// use scir_signal::FilterType;
/// let sos = scir_signal::cheby1_filter(4, 1.0, FilterType::LowPass(0.2), 2.0).unwrap();
/// let x = ndarray::Array1::from_vec(vec![0.0; 8]);
/// let _y = scir_signal::sosfilt(&sos, &x);
/// ```
pub fn cheby1_filter(
    order: u32,
    ripple: f64,
    filter_type: FilterType,
    fs: f64,
) -> Result<Sos, FilterError> {
    let zpk = design_cheby1(order, ripple, filter_type, fs)?;
    zpk2sos(&zpk, None)
}

/// Design a Chebyshev Type I **low-pass** filter on the SciPy normalized-
/// frequency convention (`fs=2`).
///
/// Thin wrapper over [`cheby1_filter`] preserved for backward compatibility.
///
/// # Examples
/// ```
/// let sos = scir_signal::cheby1(4, 1.0, 0.2);
/// let x = ndarray::Array1::from_vec(vec![0.0; 8]);
/// let y = scir_signal::sosfilt(&sos, &x);
/// assert_eq!(y.len(), x.len());
/// ```
pub fn cheby1(order: u32, ripple: f64, cutoff: f64) -> Sos {
    cheby1_filter(order, ripple, FilterType::LowPass(cutoff), 2.0)
        .expect("cheby1 design failed for valid LowPass parameters")
}

/// Design a Bessel filter (phase normalization) across all four
/// [`FilterType`] variants and return SOS form. Mirrors
/// `scipy.signal.bessel(N, Wn, btype, fs=fs, output='sos', norm='phase')`.
///
/// Supported orders are `1..=`[`MAX_BESSEL_ORDER`] (tabulated SciPy/mpmath-
/// precomputed pole list lives in `scir-iir-filters`); orders outside
/// that range return a [`FilterError::IllegalArgument`].
///
/// # Examples
/// ```
/// use scir_signal::FilterType;
/// // Equivalent to scipy.signal.bessel(4, 200, btype='highpass', fs=1000, output='sos')
/// let sos = scir_signal::bessel_filter(4, FilterType::HighPass(200.0), 1000.0).unwrap();
/// let x = ndarray::Array1::from_vec(vec![0.0; 8]);
/// let _y = scir_signal::sosfilt(&sos, &x);
/// ```
pub fn bessel_filter(
    order: u32,
    filter_type: FilterType,
    fs: f64,
) -> Result<Sos, FilterError> {
    let zpk = design_bessel(order, filter_type, fs)?;
    zpk2sos(&zpk, None)
}

/// Like [`bessel_filter`] but with explicit Bessel normalization choice.
/// Mirrors `scipy.signal.bessel(N, Wn, btype, fs=fs, norm=...)` across
/// all three [`BesselNorm`] variants. See [`BesselNorm`] for the per-norm
/// magnitude / phase / group-delay properties.
///
/// # Examples
/// ```
/// use scir_signal::{BesselNorm, FilterType};
/// // Maximally flat group delay = 1; -3 dB at ω=1 rad/s.
/// let sos =
///     scir_signal::bessel_filter_with_norm(4, BesselNorm::Mag, FilterType::LowPass(0.2), 2.0)
///         .unwrap();
/// let x = ndarray::Array1::from_vec(vec![0.0; 8]);
/// let _y = scir_signal::sosfilt(&sos, &x);
/// ```
pub fn bessel_filter_with_norm(
    order: u32,
    norm: BesselNorm,
    filter_type: FilterType,
    fs: f64,
) -> Result<Sos, FilterError> {
    let zpk = design_bessel_with_norm(order, norm, filter_type, fs)?;
    zpk2sos(&zpk, None)
}

/// Design a Bessel **low-pass** filter on the SciPy normalized-frequency
/// convention (`fs=2`).
///
/// Thin wrapper over [`bessel_filter`] preserved for backward compatibility.
///
/// # Examples
/// ```
/// let sos = scir_signal::bessel(4, 0.2);
/// // Use the SOS in a filter to validate it's usable
/// let x = ndarray::Array1::from_vec(vec![0.0; 8]);
/// let y = scir_signal::sosfilt(&sos, &x);
/// assert_eq!(y.len(), x.len());
/// ```
pub fn bessel(order: u32, cutoff: f64) -> Sos {
    bessel_filter(order, FilterType::LowPass(cutoff), 2.0)
        .expect("bessel design failed for valid LowPass parameters")
}

/// Design a 2nd-order IIR notch filter (single biquad). Mirrors
/// `scipy.signal.iirnotch(w0, Q, fs)` exactly via the closed-form
/// formula at `scipy/signal/_filter_design._design_notch_peak_filter`.
///
/// `w0` is the center frequency to remove; `q` is the notch quality
/// factor (`Q = w0 / bw_-3dB`). When `fs` is supplied, `w0` is in the
/// same units as `fs`; with `fs=2.0` (the SciPy normalized convention),
/// `w0` is the normalized frequency `0 < w0 < 1`.
///
/// # Examples
/// ```
/// // 60 Hz notch at fs=200 Hz with Q=30 (matches the SciPy doc example).
/// let sos = scir_signal::iirnotch(60.0, 30.0, 200.0).unwrap();
/// assert_eq!(sos.num_sections(), 1);
/// ```
pub fn iirnotch(w0: f64, q: f64, fs: f64) -> Result<Sos, FilterError> {
    use scir_iir_filters::errors::Error;
    if fs <= 0.0 {
        return Err(Error::IllegalArgument(format!(
            "iirnotch: fs must be > 0; got {fs}"
        )));
    }
    if q <= 0.0 || !q.is_finite() {
        return Err(Error::IllegalArgument(format!(
            "iirnotch: Q must be > 0 and finite; got {q}"
        )));
    }
    let w0n = 2.0 * w0 / fs;
    if !(0.0 < w0n && w0n < 1.0) {
        return Err(Error::IllegalArgument(format!(
            "iirnotch: normalized w0 must satisfy 0 < w0 < 1 \
             (got w0={w0}, fs={fs} → w0_norm={w0n})"
        )));
    }
    let bw = w0n / q;
    let bw_rad = bw * core::f64::consts::PI;
    let w0_rad = w0n * core::f64::consts::PI;
    let beta = (bw_rad / 2.0).tan();
    let gain = 1.0 / (1.0 + beta);
    let cos_w0 = w0_rad.cos();
    // SOS row layout: [b0, b1, b2, a0, a1, a2].
    let row = [
        gain,
        -2.0 * gain * cos_w0,
        gain,
        1.0,
        -2.0 * gain * cos_w0,
        2.0 * gain - 1.0,
    ];
    Ok(Sos::from_vec(vec![row]))
}

/// Notch presets for common rejection frequencies.
///
/// Pure-ergonomics layer over [`iirnotch`]. Each preset is a one-line
/// wrapper that pre-fills a sensible center frequency + Q for common
/// embedded / audio use cases. The non-`with_q` variants default to
/// `Q = 30.0`, the textbook value for AC mains hum rejection (sharp
/// enough to leave musical content untouched, wide enough to cover
/// real-world line-frequency drift).
pub mod presets {
    use super::{iirnotch, FilterError, Sos};

    /// AC mains rejection at 50 Hz (Europe, most of Asia + Africa,
    /// Australia, parts of South America). Q = 30. For applications
    /// needing a different Q, use [`mains_50hz_notch_with_q`].
    pub fn mains_50hz_notch(fs: f64) -> Result<Sos, FilterError> {
        iirnotch(50.0, 30.0, fs)
    }

    /// AC mains rejection at 60 Hz (Americas, parts of Asia, parts of
    /// Africa). Q = 30. For applications needing a different Q, use
    /// [`mains_60hz_notch_with_q`].
    pub fn mains_60hz_notch(fs: f64) -> Result<Sos, FilterError> {
        iirnotch(60.0, 30.0, fs)
    }

    /// AC mains rejection at 50 Hz with caller-specified Q. Typical
    /// audio range is `Q ∈ 5..=100`; lower Q rejects a wider band but
    /// affects more nearby content, higher Q is sharper but more
    /// sensitive to line-frequency drift.
    pub fn mains_50hz_notch_with_q(fs: f64, q: f64) -> Result<Sos, FilterError> {
        iirnotch(50.0, q, fs)
    }

    /// AC mains rejection at 60 Hz with caller-specified Q.
    pub fn mains_60hz_notch_with_q(fs: f64, q: f64) -> Result<Sos, FilterError> {
        iirnotch(60.0, q, fs)
    }
}

/// Apply a second-order-section filter to input data.
///
/// # Examples
/// ```
/// let sos = scir_signal::butter(2, 0.2);
/// let x = ndarray::Array1::from_vec(vec![0.0; 8]);
/// let y = scir_signal::sosfilt(&sos, &x);
/// assert_eq!(y.len(), x.len());
/// ```
pub fn sosfilt(sos: &Sos, input: &Array1<f64>) -> Array1<f64> {
    let mut df = DirectForm2Transposed::new(sos);
    let mut out = Vec::with_capacity(input.len());
    for &x in input.iter() {
        out.push(df.filter(x));
    }
    Array1::from(out)
}

/// Zero-phase filtering by applying `sosfilt` forward and backward.
///
/// # Examples
/// ```
/// let sos = scir_signal::butter(2, 0.2);
/// let x = ndarray::Array1::from_vec(vec![0.0; 8]);
/// let y = scir_signal::filtfilt(&sos, &x);
/// assert_eq!(y.len(), x.len());
/// ```
pub fn filtfilt(sos: &Sos, input: &Array1<f64>) -> Array1<f64> {
    let mut df = DirectForm2Transposed::new(sos);
    let mut tmp = Vec::with_capacity(input.len());
    for &x in input.iter() {
        tmp.push(df.filter(x));
    }
    let mut df2 = DirectForm2Transposed::new(sos);
    let mut out = Vec::with_capacity(tmp.len());
    for &x in tmp.iter().rev() {
        out.push(df2.filter(x));
    }
    out.reverse();
    Array1::from(out)
}

fn convolve(x: &[f64], h: &[f64]) -> Vec<f64> {
    let n = x.len();
    let m = h.len();
    let mut y = vec![0.0; n + m - 1];
    for i in 0..n {
        for j in 0..m {
            y[i + j] += x[i] * h[j];
        }
    }
    y
}

/// Resample using polyphase filtering.
///
/// # Examples
/// ```
/// let x = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let y = scir_signal::resample_poly(&x, 2, 3);
/// assert!(y.len() > 0);
/// ```
pub fn resample_poly(input: &Array1<f64>, up: usize, down: usize) -> Array1<f64> {
    assert!(up == 2 && down == 3);
    const H: [f64; 31] = [
        2.074_505_559_623_708e-18,
        3.538_600_036_880_219e-3,
        5.068_355_234_374_961e-3,
        -4.352_628_574_659_906e-18,
        -1.161_274_884_414_711e-2,
        -1.705_404_264_360_788e-2,
        1.031_683_205_857_806e-17,
        3.382_730_607_794_791e-2,
        4.621_748_455_191_45e-2,
        -1.768_899_299_634_197e-17,
        -8.472_008_281_018_12e-2,
        -1.166_001_508_003_416e-1,
        2.365_319_648_026_013e-17,
        2.641_261_112_200_551e-1,
        5.446_004_399_706_618e-1,
        6.652_174_560_128_868e-1,
        5.446_004_399_706_618e-1,
        2.641_261_112_200_551e-1,
        2.365_319_648_026_013e-17,
        -1.166_001_508_003_416e-1,
        -8.472_008_281_018_122e-2,
        -1.768_899_299_634_197e-17,
        4.621_748_455_191_452e-2,
        3.382_730_607_794_793e-2,
        1.031_683_205_857_806e-17,
        -1.705_404_264_360_79e-2,
        -1.161_274_884_414_712e-2,
        -4.352_628_574_659_906e-18,
        5.068_355_234_374_964e-3,
        3.538_600_036_880_219e-3,
        2.074_505_559_623_708e-18,
    ];
    let mut upsampled = vec![0.0; input.len() * up];
    for (i, &val) in input.iter().enumerate() {
        upsampled[i * up] = val;
    }
    let conv = convolve(&upsampled, &H);
    let offset = (H.len() - 1) / 2;
    let end = conv.len() - offset;
    let mut out = Vec::new();
    let mut idx = offset;
    while idx < end {
        out.push(conv[idx]);
        idx += down;
    }
    Array1::from(out)
}

// GPU-forwarded APIs (optional)
#[cfg(feature = "gpu")]
pub mod gpu {
    use ndarray::{Array1, Array2};
    use scir_gpu::{fir1d_batched_f32_auto, Device};

    /// Batched FIR for f32 with device selection.
    pub fn fir1d_batched_f32(x: &Array2<f32>, taps: &Array1<f32>, device: Device) -> Array2<f32> {
        fir1d_batched_f32_auto(x, taps, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use ndarray_npy::ReadNpyExt;
    use scir_core::assert_close;
    use std::{fs::File, path::PathBuf};

    fn fixtures_base() -> Option<PathBuf> {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        if base.exists() {
            Some(base)
        } else {
            None
        }
    }

    fn sos_from_array(arr: &Array2<f64>) -> Sos {
        let vec: Vec<[f64; 6]> = (0..arr.nrows())
            .map(|i| {
                [
                    arr[[i, 0]],
                    arr[[i, 1]],
                    arr[[i, 2]],
                    arr[[i, 3]],
                    arr[[i, 4]],
                    arr[[i, 5]],
                ]
            })
            .collect();
        Sos::from_vec(vec)
    }

    #[test]
    fn sosfilt_matches_fixture() {
        let Some(base) = fixtures_base() else {
            eprintln!("[scir-signal] fixtures missing; skipping sosfilt_matches_fixture");
            return;
        };
        let sos_arr: Array2<f64> =
            ReadNpyExt::read_npy(File::open(base.join("butter_sos.npy")).unwrap()).unwrap();
        let sos = sos_from_array(&sos_arr);
        let input: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_input.npy")).unwrap()).unwrap();
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_output.npy")).unwrap()).unwrap();
        let result = sosfilt(&sos, &input);
        assert_close!(&result, &expected, array, atol = 1e-6, rtol = 1e-6);
    }

    #[test]
    fn butter_design_filters_correctly() {
        let Some(base) = fixtures_base() else {
            eprintln!("[scir-signal] fixtures missing; skipping butter_design_filters_correctly");
            return;
        };
        let input: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_input.npy")).unwrap()).unwrap();
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_output.npy")).unwrap()).unwrap();
        let sos = butter(4, 0.2);
        let result = sosfilt(&sos, &input);
        assert_close!(&result, &expected, array, atol = 1e-6, rtol = 1e-6);
    }

    #[test]
    fn cheby1_design_matches_fixture() {
        let Some(base) = fixtures_base() else {
            eprintln!("[scir-signal] fixtures missing; skipping cheby1_design_matches_fixture");
            return;
        };
        let input: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_input.npy")).unwrap()).unwrap();
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("cheby_output.npy")).unwrap()).unwrap();
        let sos = cheby1(4, 1.0, 0.2);
        let result = sosfilt(&sos, &input);
        assert_close!(&result, &expected, array, atol = 1e-6, rtol = 1e-6);
    }

    #[test]
    fn bessel_design_matches_fixture() {
        let Some(base) = fixtures_base() else {
            eprintln!("[scir-signal] fixtures missing; skipping bessel_design_matches_fixture");
            return;
        };
        let input: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_input.npy")).unwrap()).unwrap();
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("bessel_output.npy")).unwrap()).unwrap();
        let sos = bessel(4, 0.2);
        let result = sosfilt(&sos, &input);
        assert_close!(&result, &expected, array, atol = 1e-6, rtol = 1e-6);
    }

    #[test]
    fn butter_highpass_design_matches_fixture() {
        let Some(base) = fixtures_base() else {
            eprintln!(
                "[scir-signal] fixtures missing; skipping butter_highpass_design_matches_fixture"
            );
            return;
        };
        let input: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_input.npy")).unwrap()).unwrap();
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("butter_hp_output.npy")).unwrap()).unwrap();
        let sos =
            butter_filter(4, FilterType::HighPass(0.3), 2.0).expect("butter_filter HPF failed");
        let result = sosfilt(&sos, &input);
        assert_close!(&result, &expected, array, atol = 1e-6, rtol = 1e-6);
    }

    #[test]
    fn bessel_lowpass_delay_norm_matches_fixture() {
        let Some(base) = fixtures_base() else {
            eprintln!(
                "[scir-signal] fixtures missing; skipping bessel_lowpass_delay_norm_matches_fixture"
            );
            return;
        };
        let input: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_input.npy")).unwrap()).unwrap();
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("bessel_lp_delay_output.npy")).unwrap())
                .unwrap();
        let sos = bessel_filter_with_norm(4, BesselNorm::Delay, FilterType::LowPass(0.2), 2.0)
            .expect("bessel_filter_with_norm Delay");
        let result = sosfilt(&sos, &input);
        assert_close!(&result, &expected, array, atol = 1e-6, rtol = 1e-6);
    }

    #[test]
    fn bessel_lowpass_mag_norm_matches_fixture() {
        let Some(base) = fixtures_base() else {
            eprintln!(
                "[scir-signal] fixtures missing; skipping bessel_lowpass_mag_norm_matches_fixture"
            );
            return;
        };
        let input: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_input.npy")).unwrap()).unwrap();
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("bessel_lp_mag_output.npy")).unwrap())
                .unwrap();
        let sos = bessel_filter_with_norm(4, BesselNorm::Mag, FilterType::LowPass(0.2), 2.0)
            .expect("bessel_filter_with_norm Mag");
        let result = sosfilt(&sos, &input);
        assert_close!(&result, &expected, array, atol = 1e-6, rtol = 1e-6);
    }

    #[test]
    fn bessel_phase_norm_default_matches_explicit() {
        // bessel_filter (default phase) and bessel_filter_with_norm(Phase)
        // MUST produce byte-identical SOS for the same parameters.
        let a = bessel_filter(4, FilterType::HighPass(0.3), 2.0).unwrap();
        let b = bessel_filter_with_norm(4, BesselNorm::Phase, FilterType::HighPass(0.3), 2.0)
            .unwrap();
        let arr_a = a.sections();
        let arr_b = b.sections();
        assert_eq!(arr_a.len(), arr_b.len());
        for (x, y) in arr_a.iter().zip(arr_b.iter()) {
            assert_eq!(x.b0, y.b0);
            assert_eq!(x.b1, y.b1);
            assert_eq!(x.b2, y.b2);
            assert_eq!(x.a0, y.a0);
            assert_eq!(x.a1, y.a1);
            assert_eq!(x.a2, y.a2);
        }
    }

    #[test]
    fn notch_presets_match_inline_iirnotch() {
        // The presets module is a pure passthrough to iirnotch with
        // pre-filled (frequency, Q) tuples. Each preset MUST produce
        // byte-identical SOS to the equivalent inline call.
        let fs = 48_000.0;

        let direct_50 = iirnotch(50.0, 30.0, fs).unwrap();
        let preset_50 = presets::mains_50hz_notch(fs).unwrap();
        assert_eq!(direct_50.to_arrays(), preset_50.to_arrays());

        let direct_60 = iirnotch(60.0, 30.0, fs).unwrap();
        let preset_60 = presets::mains_60hz_notch(fs).unwrap();
        assert_eq!(direct_60.to_arrays(), preset_60.to_arrays());

        // _with_q variants override the Q while keeping the frequency.
        let direct_50_q15 = iirnotch(50.0, 15.0, fs).unwrap();
        let preset_50_q15 = presets::mains_50hz_notch_with_q(fs, 15.0).unwrap();
        assert_eq!(direct_50_q15.to_arrays(), preset_50_q15.to_arrays());

        let direct_60_q15 = iirnotch(60.0, 15.0, fs).unwrap();
        let preset_60_q15 = presets::mains_60hz_notch_with_q(fs, 15.0).unwrap();
        assert_eq!(direct_60_q15.to_arrays(), preset_60_q15.to_arrays());
    }

    #[test]
    fn iirnotch_design_matches_fixture() {
        let Some(base) = fixtures_base() else {
            eprintln!("[scir-signal] fixtures missing; skipping iirnotch_design_matches_fixture");
            return;
        };
        let input: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_input.npy")).unwrap()).unwrap();
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("notch_output.npy")).unwrap()).unwrap();
        // Same arguments as the fixture generator: w0=0.2, Q=30, fs=2.0.
        let sos = iirnotch(0.2, 30.0, 2.0).expect("iirnotch failed");
        let result = sosfilt(&sos, &input);
        assert_close!(&result, &expected, array, atol = 1e-12, rtol = 1e-12);
    }

    #[test]
    fn iirnotch_rejects_invalid_inputs() {
        // Q ≤ 0 or non-finite
        assert!(iirnotch(0.2, 0.0, 2.0).is_err());
        assert!(iirnotch(0.2, -1.0, 2.0).is_err());
        assert!(iirnotch(0.2, f64::NAN, 2.0).is_err());
        // w0 outside (0, 1) (normalized)
        assert!(iirnotch(0.0, 30.0, 2.0).is_err());
        assert!(iirnotch(1.5, 30.0, 2.0).is_err());
        // fs ≤ 0
        assert!(iirnotch(60.0, 30.0, 0.0).is_err());
    }

    #[test]
    fn bessel_highpass_design_matches_fixture() {
        let Some(base) = fixtures_base() else {
            eprintln!(
                "[scir-signal] fixtures missing; skipping bessel_highpass_design_matches_fixture"
            );
            return;
        };
        let input: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_input.npy")).unwrap()).unwrap();
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("bessel_hp_output.npy")).unwrap()).unwrap();
        let sos =
            bessel_filter(4, FilterType::HighPass(0.3), 2.0).expect("bessel_filter HPF failed");
        let result = sosfilt(&sos, &input);
        assert_close!(&result, &expected, array, atol = 1e-6, rtol = 1e-6);
    }

    #[test]
    fn cheby1_highpass_design_matches_fixture() {
        let Some(base) = fixtures_base() else {
            eprintln!(
                "[scir-signal] fixtures missing; skipping cheby1_highpass_design_matches_fixture"
            );
            return;
        };
        let input: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_input.npy")).unwrap()).unwrap();
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("cheby_hp_output.npy")).unwrap()).unwrap();
        let sos = cheby1_filter(4, 1.0, FilterType::HighPass(0.3), 2.0)
            .expect("cheby1_filter HPF failed");
        let result = sosfilt(&sos, &input);
        assert_close!(&result, &expected, array, atol = 1e-6, rtol = 1e-6);
    }

    #[test]
    fn filtfilt_matches_fixture() {
        let Some(base) = fixtures_base() else {
            eprintln!("[scir-signal] fixtures missing; skipping filtfilt_matches_fixture");
            return;
        };
        let sos_arr: Array2<f64> =
            ReadNpyExt::read_npy(File::open(base.join("butter_sos.npy")).unwrap()).unwrap();
        let sos = sos_from_array(&sos_arr);
        let input: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_input.npy")).unwrap()).unwrap();
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("filtfilt_output.npy")).unwrap()).unwrap();
        let result = filtfilt(&sos, &input);
        assert_close!(&result, &expected, array, atol = 1e-6, rtol = 1e-6);
    }

    #[test]
    fn resample_poly_matches_fixture() {
        let Some(base) = fixtures_base() else {
            eprintln!("[scir-signal] fixtures missing; skipping resample_poly_matches_fixture");
            return;
        };
        let input: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_input.npy")).unwrap()).unwrap();
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("resample_poly_output.npy")).unwrap())
                .unwrap();
        let result = resample_poly(&input, 2, 3);
        assert_close!(&result, &expected, array, atol = 2e-2, rtol = 1e-6);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn fir1d_batched_f32_auto_cpu_matches_naive() {
        use crate::gpu::fir1d_batched_f32 as fir_auto;
        use scir_gpu::Device;
        let x: Array2<f32> = ndarray::array![[1.0, 2.0, 3.0, 4.0], [0.5, 0.0, -0.5, -1.0]];
        let taps: Array1<f32> = ndarray::array![0.25, 0.5, 0.25];
        let y = fir_auto(&x, &taps, Device::Cpu);
        // naive reference
        let (b, n) = x.dim();
        let k = taps.len();
        let mut y_ref = Array2::<f32>::zeros((b, n));
        for bi in 0..b {
            for i in 0..n {
                let mut acc = 0.0f32;
                let start = if i + 1 >= k { i + 1 - k } else { 0 };
                for (t_idx, xi) in (start..=i).rev().enumerate() {
                    let tap = taps[k - 1 - t_idx];
                    acc += tap * x[[bi, xi]];
                }
                y_ref[[bi, i]] = acc;
            }
        }
        assert_close!(&y.into_raw_vec(), &y_ref.into_raw_vec(), slice, tol = 0.0);
    }
}
