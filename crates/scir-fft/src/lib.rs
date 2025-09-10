//! FFT utilities for SciR.
#![deny(missing_docs)]

use ndarray::Array1;
use num_complex::Complex64;
use realfft::RealFftPlanner;
use rustfft::FftPlanner;

/// Compute the forward FFT of a real-valued array.
///
/// # Examples
/// ```
/// use ndarray::Array1;
/// let x = Array1::from_vec(vec![0.0, 1.0, 0.0, -1.0]);
/// let y = scir_fft::fft(&x);
/// assert_eq!(y.len(), x.len());
/// ```
pub fn fft(input: &Array1<f64>) -> Array1<Complex64> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(input.len());
    let mut buffer: Vec<Complex64> = input.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    fft.process(&mut buffer);
    Array1::from_vec(buffer)
}

/// Compute the inverse FFT of a complex-valued array.
///
/// # Examples
/// ```
/// use ndarray::Array1;
/// use num_complex::Complex64;
/// let x = Array1::from_vec(vec![Complex64::new(1.0,0.0); 4]);
/// let y = scir_fft::ifft(&x);
/// assert_eq!(y.len(), x.len());
/// ```
pub fn ifft(input: &Array1<Complex64>) -> Array1<Complex64> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_inverse(input.len());
    let mut buffer: Vec<Complex64> = input.to_vec();
    fft.process(&mut buffer);
    let n = input.len() as f64;
    Array1::from_vec(buffer.into_iter().map(|v| v / n).collect())
}

/// Compute the forward real FFT of a real-valued array.
///
/// # Examples
/// ```
/// use ndarray::Array1;
/// let x = Array1::from_vec(vec![0.0, 1.0, 0.0, -1.0]);
/// let y = scir_fft::rfft(&x);
/// assert!(y.len() >= 1);
/// ```
pub fn rfft(input: &Array1<f64>) -> Array1<Complex64> {
    let mut planner = RealFftPlanner::<f64>::new();
    let rfft = planner.plan_fft_forward(input.len());
    let mut buffer = input.to_vec();
    let mut spectrum = rfft.make_output_vec();
    rfft.process(&mut buffer, &mut spectrum).unwrap();
    Array1::from_vec(spectrum)
}

/// Compute the inverse real FFT producing a real-valued array.
///
/// # Examples
/// ```
/// use ndarray::Array1;
/// use num_complex::Complex64;
/// let spec = Array1::from_vec(vec![Complex64::new(1.0,0.0); 3]);
/// let x = scir_fft::irfft(&spec);
/// assert!(x.len() >= 2);
/// ```
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

    #[test]
    fn fft_matches_fixtures() {
        let Some(base) = fixtures_base() else {
            eprintln!("[scir-fft] fixtures missing; skipping fft_matches_fixtures");
            return;
        };
        for &n in &[8, 16] {
            let in_path = base.join(format!("fft_input_{n}.npy"));
            let out_path = base.join(format!("fft_output_{n}.npy"));
            let input: Array1<f64> = match File::open(&in_path)
                .ok()
                .and_then(|f| ReadNpyExt::read_npy(f).ok())
            {
                Some(v) => v,
                None => {
                    eprintln!("[scir-fft] missing {}; skipping", in_path.display());
                    return;
                }
            };
            let expected: Array1<Complex64> = match File::open(&out_path)
                .ok()
                .and_then(|f| ReadNpyExt::read_npy(f).ok())
            {
                Some(v) => v,
                None => {
                    eprintln!("[scir-fft] missing {}; skipping", out_path.display());
                    return;
                }
            };
            let result = fft(&input);
            assert_close!(&result, &expected, complex_array, atol = 1e-9, rtol = 1e-9);
        }
    }

    #[test]
    fn ifft_matches_fixtures() {
        let Some(base) = fixtures_base() else {
            eprintln!("[scir-fft] fixtures missing; skipping ifft_matches_fixtures");
            return;
        };
        for &n in &[8, 16] {
            let in_path = base.join(format!("fft_output_{n}.npy"));
            let exp_path = base.join(format!("ifft_output_{n}.npy"));
            let input: Array1<Complex64> = match File::open(&in_path)
                .ok()
                .and_then(|f| ReadNpyExt::read_npy(f).ok())
            {
                Some(v) => v,
                None => {
                    eprintln!("[scir-fft] missing {}; skipping", in_path.display());
                    return;
                }
            };
            let expected: Array1<Complex64> = match File::open(&exp_path)
                .ok()
                .and_then(|f| ReadNpyExt::read_npy(f).ok())
            {
                Some(v) => v,
                None => {
                    eprintln!("[scir-fft] missing {}; skipping", exp_path.display());
                    return;
                }
            };
            let result = ifft(&input);
            assert_close!(&result, &expected, complex_array, atol = 1e-9, rtol = 1e-9);
        }
    }

    #[test]
    fn rfft_matches_fixtures() {
        let Some(base) = fixtures_base() else {
            eprintln!("[scir-fft] fixtures missing; skipping rfft_matches_fixtures");
            return;
        };
        for &n in &[8, 16] {
            let in_path = base.join(format!("fft_input_{n}.npy"));
            let exp_path = base.join(format!("rfft_output_{n}.npy"));
            let input: Array1<f64> = match File::open(&in_path)
                .ok()
                .and_then(|f| ReadNpyExt::read_npy(f).ok())
            {
                Some(v) => v,
                None => {
                    eprintln!("[scir-fft] missing {}; skipping", in_path.display());
                    return;
                }
            };
            let expected: Array1<Complex64> = match File::open(&exp_path)
                .ok()
                .and_then(|f| ReadNpyExt::read_npy(f).ok())
            {
                Some(v) => v,
                None => {
                    eprintln!("[scir-fft] missing {}; skipping", exp_path.display());
                    return;
                }
            };
            let result = rfft(&input);
            assert_close!(&result, &expected, complex_array, atol = 1e-9, rtol = 1e-9);
        }
    }

    #[test]
    fn irfft_matches_fixtures() {
        let Some(base) = fixtures_base() else {
            eprintln!("[scir-fft] fixtures missing; skipping irfft_matches_fixtures");
            return;
        };
        for &n in &[8, 16] {
            let in_path = base.join(format!("rfft_output_{n}.npy"));
            let exp_path = base.join(format!("fft_input_{n}.npy"));
            let input: Array1<Complex64> = match File::open(&in_path)
                .ok()
                .and_then(|f| ReadNpyExt::read_npy(f).ok())
            {
                Some(v) => v,
                None => {
                    eprintln!("[scir-fft] missing {}; skipping", in_path.display());
                    return;
                }
            };
            let expected: Array1<f64> = match File::open(&exp_path)
                .ok()
                .and_then(|f| ReadNpyExt::read_npy(f).ok())
            {
                Some(v) => v,
                None => {
                    eprintln!("[scir-fft] missing {}; skipping", exp_path.display());
                    return;
                }
            };
            let result = irfft(&input);
            assert_close!(&result, &expected, array, atol = 1e-9, rtol = 1e-9);
        }
    }
}
