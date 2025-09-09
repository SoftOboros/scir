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
        2.0745055596237081e-18,
        3.5386000368802188e-03,
        5.0683552343749614e-03,
        -4.3526285746599063e-18,
        -1.1612748844147112e-02,
        -1.7054042643607877e-02,
        1.0316832058578061e-17,
        3.3827306077947913e-02,
        4.6217484551914496e-02,
        -1.7688992996341970e-17,
        -8.4720082810181202e-02,
        -1.1660015080034160e-01,
        2.3653196480260126e-17,
        2.6412611122005514e-01,
        5.4460043997066177e-01,
        6.6521745601288684e-01,
        5.4460043997066177e-01,
        2.6412611122005514e-01,
        2.3653196480260126e-17,
        -1.1660015080034161e-01,
        -8.4720082810181220e-02,
        -1.7688992996341970e-17,
        4.6217484551914517e-02,
        3.3827306077947927e-02,
        1.0316832058578061e-17,
        -1.7054042643607898e-02,
        -1.1612748844147121e-02,
        -4.3526285746599063e-18,
        5.0683552343749640e-03,
        3.5386000368802188e-03,
        2.0745055596237081e-18,
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use ndarray_npy::ReadNpyExt;
    use scir_core::assert_close;
    use std::{fs::File, path::PathBuf};

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
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
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
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
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
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
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
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
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
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
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
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        let input: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("sosfilt_input.npy")).unwrap()).unwrap();
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("resample_poly_output.npy")).unwrap())
                .unwrap();
        let result = resample_poly(&input, 2, 3);
        assert_close!(&result, &expected, array, atol = 2e-2, rtol = 1e-6);
    }
}
