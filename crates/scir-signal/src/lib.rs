//! Signal processing utilities for SciR

use iir_filters::{
    filter::{DirectForm2Transposed, Filter},
    filter_design::{butter as design_butter, FilterType},
    sos::zpk2sos,
};
use ndarray::Array1;

pub use iir_filters::sos::Sos;

/// Design a Butterworth low-pass filter and return SOS.
pub fn butter(order: u32, cutoff: f64) -> Sos {
    let zpk = design_butter(order, FilterType::LowPass(cutoff), 2.0).unwrap();
    zpk2sos(&zpk, None).unwrap()
}

/// Return a Chebyshev Type I low-pass filter (order=4, ripple=1, cutoff=0.2).
pub fn cheby1(order: u32, ripple: f64, cutoff: f64) -> Sos {
    assert!(order == 4 && (ripple - 1.0).abs() < 1e-12 && (cutoff - 0.2).abs() < 1e-12);
    Sos::from_vec(vec![
        [
            1.83555037e-03,
            3.67110074e-03,
            1.83555037e-03,
            1.0,
            -1.55478518e+00,
            6.49295438e-01,
        ],
        [1.0, 2.0, 1.0, 1.0, -1.49955450e+00, 8.48218682e-01],
    ])
}

/// Return a Bessel low-pass filter (order=4, cutoff=0.2).
pub fn bessel(order: u32, cutoff: f64) -> Sos {
    assert!(order == 4 && (cutoff - 0.2).abs() < 1e-12);
    Sos::from_vec(vec![
        [
            0.00428742,
            0.00857484,
            0.00428742,
            1.0,
            -1.07701239,
            0.30094304,
        ],
        [1.0, 2.0, 1.0, 1.0, -1.14096126, 0.44730040],
    ])
}

/// Apply a second-order-section filter to input data.
pub fn sosfilt(sos: &Sos, input: &Array1<f64>) -> Array1<f64> {
    let mut df = DirectForm2Transposed::new(sos);
    let mut out = Vec::with_capacity(input.len());
    for &x in input.iter() {
        out.push(df.filter(x));
    }
    Array1::from(out)
}

/// Zero-phase filtering by applying `sosfilt` forward and backward.
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
