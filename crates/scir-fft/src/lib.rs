//! FFT utilities for SciR

use ndarray::Array1;
use num_complex::Complex64;
use rustfft::FftPlanner;
use realfft::RealFftPlanner;

/// Compute the forward FFT of a real-valued array.
pub fn fft(input: &Array1<f64>) -> Array1<Complex64> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(input.len());
    let mut buffer: Vec<Complex64> = input.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    fft.process(&mut buffer);
    Array1::from_vec(buffer)
}

/// Compute the inverse FFT of a complex-valued array.
pub fn ifft(input: &Array1<Complex64>) -> Array1<Complex64> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_inverse(input.len());
    let mut buffer: Vec<Complex64> = input.to_vec();
    fft.process(&mut buffer);
    let n = input.len() as f64;
    Array1::from_vec(buffer.into_iter().map(|v| v / n).collect())
}

/// Compute the forward real FFT of a real-valued array.
pub fn rfft(input: &Array1<f64>) -> Array1<Complex64> {
    let mut planner = RealFftPlanner::<f64>::new();
    let rfft = planner.plan_fft_forward(input.len());
    let mut buffer = input.to_vec();
    let mut spectrum = rfft.make_output_vec();
    rfft.process(&mut buffer, &mut spectrum).unwrap();
    Array1::from_vec(spectrum)
}

/// Compute the inverse real FFT producing a real-valued array.
pub fn irfft(input: &Array1<Complex64>) -> Array1<f64> {
    let n = (input.len() - 1) * 2;
    let mut planner = RealFftPlanner::<f64>::new();
    let irfft = planner.plan_fft_inverse(n);
    let mut buffer = input.to_vec();
    let mut output = irfft.make_output_vec();
    irfft.process(&mut buffer, &mut output).unwrap();
    let n_f64 = n as f64;
    Array1::from_vec(output.into_iter().map(|v| v / n_f64).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_npy::ReadNpyExt;
    use std::{fs::File, path::PathBuf};
    use scir_core::assert_close;

    #[test]
    fn fft_matches_fixtures() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        for &n in &[8, 16] {
            let input: Array1<f64> =
                ReadNpyExt::read_npy(File::open(base.join(format!("fft_input_{n}.npy"))).unwrap()).unwrap();
            let expected: Array1<Complex64> =
                ReadNpyExt::read_npy(File::open(base.join(format!("fft_output_{n}.npy"))).unwrap()).unwrap();
            let result = fft(&input);
            assert_close!(&result, &expected, complex_array, atol = 1e-9, rtol = 1e-9);
        }
    }

    #[test]
    fn ifft_matches_fixtures() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        for &n in &[8, 16] {
            let input: Array1<Complex64> = ReadNpyExt::read_npy(
                File::open(base.join(format!("fft_output_{n}.npy"))).unwrap(),
            )
            .unwrap();
            let expected: Array1<Complex64> = ReadNpyExt::read_npy(
                File::open(base.join(format!("ifft_output_{n}.npy"))).unwrap(),
            )
            .unwrap();
            let result = ifft(&input);
            assert_close!(&result, &expected, complex_array, atol = 1e-9, rtol = 1e-9);
        }
    }

    #[test]
    fn rfft_matches_fixtures() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        for &n in &[8, 16] {
            let input: Array1<f64> =
                ReadNpyExt::read_npy(File::open(base.join(format!("fft_input_{n}.npy"))).unwrap()).unwrap();
            let expected: Array1<Complex64> = ReadNpyExt::read_npy(
                File::open(base.join(format!("rfft_output_{n}.npy"))).unwrap(),
            )
            .unwrap();
            let result = rfft(&input);
            assert_close!(&result, &expected, complex_array, atol = 1e-9, rtol = 1e-9);
        }
    }

    #[test]
    fn irfft_matches_fixtures() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        for &n in &[8, 16] {
            let input: Array1<Complex64> = ReadNpyExt::read_npy(
                File::open(base.join(format!("rfft_output_{n}.npy"))).unwrap(),
            )
            .unwrap();
            let expected: Array1<f64> = ReadNpyExt::read_npy(
                File::open(base.join(format!("fft_input_{n}.npy"))).unwrap(),
            )
            .unwrap();
            let result = irfft(&input);
            assert_close!(&result, &expected, array, atol = 1e-9, rtol = 1e-9);
        }
    }
}
