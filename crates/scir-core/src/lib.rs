//! Core utilities for SciR.
//!
//! This crate provides shared numeric helpers and the `assert_close!` macro
//! used across SciR crates to compare floating‑point (real/complex) values
//! within absolute/relative tolerances.

#![deny(missing_docs)]

use ndarray::Array1;
use num_complex::Complex64;

/// Assert that two scalars are close within absolute (`atol`) and relative (`rtol`) tolerances.
///
/// # Examples
/// ```
/// scir_core::assert_close_scalar(1.0, 1.0 + 1e-9, 1e-8, 1e-8);
/// ```
pub fn assert_close_scalar(left: f64, right: f64, atol: f64, rtol: f64) {
    let diff = (left - right).abs();
    let tol = atol + rtol * right.abs();
    assert!(
        diff <= tol,
        "expected |{left} - {right}| <= {tol}, got {diff}"
    );
}

/// Assert that two `&[f64]` slices are elementwise close within tolerances.
///
/// # Examples
/// ```
/// let a = [1.0, 2.0, 3.0];
/// let b = [1.0 + 1e-9, 2.0, 3.0];
/// scir_core::assert_close_slice(&a, &b, 1e-8, 1e-8);
/// ```
pub fn assert_close_slice(left: &[f64], right: &[f64], atol: f64, rtol: f64) {
    assert_eq!(left.len(), right.len(), "length mismatch");
    for (l, r) in left.iter().zip(right.iter()) {
        assert_close_scalar(*l, *r, atol, rtol);
    }
}

/// Assert that two `&[Complex64]` slices are elementwise close within tolerances (by magnitude).
///
/// # Examples
/// ```
/// use num_complex::Complex64;
/// let a = [Complex64::new(1.0, 0.0)];
/// let b = [Complex64::new(1.0 + 1e-12, 1e-12)];
/// scir_core::assert_close_complex_slice(&a, &b, 1e-8, 1e-8);
/// ```
pub fn assert_close_complex_slice(left: &[Complex64], right: &[Complex64], atol: f64, rtol: f64) {
    assert_eq!(left.len(), right.len(), "length mismatch");
    for (l, r) in left.iter().zip(right.iter()) {
        let diff = (*l - *r).norm();
        let tol = atol + r.norm() * rtol;
        assert!(diff <= tol, "expected |{l:?} - {r:?}| <= {tol}, got {diff}");
    }
}

/// Assert that two `Array1<f64>` are elementwise close within tolerances.
///
/// # Examples
/// ```
/// use ndarray::Array1;
/// let a = Array1::from_vec(vec![1.0, 2.0]);
/// let b = Array1::from_vec(vec![1.0 + 1e-9, 2.0]);
/// scir_core::assert_close_array1(&a, &b, 1e-8, 1e-8);
/// ```
pub fn assert_close_array1(left: &Array1<f64>, right: &Array1<f64>, atol: f64, rtol: f64) {
    assert_eq!(left.len(), right.len(), "length mismatch");
    for (l, r) in left.iter().zip(right.iter()) {
        assert_close_scalar(*l, *r, atol, rtol);
    }
}

/// Assert that two `Array1<Complex64>` are elementwise close within tolerances (by magnitude).
///
/// # Examples
/// ```
/// use ndarray::Array1;
/// use num_complex::Complex64;
/// let a = Array1::from_vec(vec![Complex64::new(1.0, 0.0)]);
/// let b = Array1::from_vec(vec![Complex64::new(1.0 + 1e-12, 1e-12)]);
/// scir_core::assert_close_complex_array1(&a, &b, 1e-8, 1e-8);
/// ```
pub fn assert_close_complex_array1(
    left: &Array1<Complex64>,
    right: &Array1<Complex64>,
    atol: f64,
    rtol: f64,
) {
    assert_eq!(left.len(), right.len(), "length mismatch");
    for (l, r) in left.iter().zip(right.iter()) {
        let diff = (*l - *r).norm();
        let tol = atol + r.norm() * rtol;
        assert!(diff <= tol, "expected |{l:?} - {r:?}| <= {tol}, got {diff}");
    }
}

/// Assert numeric closeness with a flexible interface (scalars, slices, arrays, real/complex).
///
/// Examples:
/// - `assert_close!(a, b, atol = 1e-8, rtol = 1e-8);`
/// - `assert_close!(&xs, &ys, slice, tol = 1e-9);`
/// - `assert_close!(&xa, &ya, complex_array, atol = 1e-9, rtol = 1e-9);`
#[macro_export]
macro_rules! assert_close {
    ($left:expr, $right:expr, atol = $atol:expr, rtol = $rtol:expr) => {{
        $crate::assert_close_scalar($left as f64, $right as f64, $atol, $rtol);
    }};
    ($left:expr, $right:expr, tol = $tol:expr) => {{
        $crate::assert_close_scalar($left as f64, $right as f64, $tol, 0.0);
    }};
    ($left:expr, $right:expr, slice, atol = $atol:expr, rtol = $rtol:expr) => {{
        $crate::assert_close_slice($left, $right, $atol, $rtol);
    }};
    ($left:expr, $right:expr, slice, tol = $tol:expr) => {{
        $crate::assert_close_slice($left, $right, $tol, 0.0);
    }};
    ($left:expr, $right:expr, complex_slice, atol = $atol:expr, rtol = $rtol:expr) => {{
        $crate::assert_close_complex_slice($left, $right, $atol, $rtol);
    }};
    ($left:expr, $right:expr, complex_slice, tol = $tol:expr) => {{
        $crate::assert_close_complex_slice($left, $right, $tol, 0.0);
    }};
    ($left:expr, $right:expr, array, atol = $atol:expr, rtol = $rtol:expr) => {{
        $crate::assert_close_array1($left, $right, $atol, $rtol);
    }};
    ($left:expr, $right:expr, array, tol = $tol:expr) => {{
        $crate::assert_close_array1($left, $right, $tol, 0.0);
    }};
    ($left:expr, $right:expr, complex_array, atol = $atol:expr, rtol = $rtol:expr) => {{
        $crate::assert_close_complex_array1($left, $right, $atol, $rtol);
    }};
    ($left:expr, $right:expr, complex_array, tol = $tol:expr) => {{
        $crate::assert_close_complex_array1($left, $right, $tol, 0.0);
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use num_complex::Complex64;

    #[test]
    fn macro_works() {
        assert_close!(1.0, 1.0 + 1e-9, atol = 1e-8, rtol = 1e-8);
        assert_close!(1.0f32, 1.0f32 + 1e-6, tol = 1e-5);
    }

    #[test]
    fn slice_macro() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0 + 1e-9, 2.0 - 1e-9, 3.0];
        assert_close!(&a, &b, slice, atol = 1e-8, rtol = 1e-8);
    }

    #[test]
    fn complex_slice_macro() {
        let a = [Complex64::new(1.0, 2.0), Complex64::new(0.5, -0.5)];
        let b = [Complex64::new(1.0, 2.0 + 1e-9), Complex64::new(0.5, -0.5)];
        assert_close!(&a, &b, complex_slice, tol = 1e-8);
    }

    #[test]
    fn array_macro() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![1.0 + 1e-9, 2.0 - 1e-9, 3.0]);
        assert_close!(&a, &b, array, atol = 1e-8, rtol = 1e-8);
    }

    #[test]
    fn complex_array_macro() {
        let a = Array1::from_vec(vec![Complex64::new(1.0, 2.0), Complex64::new(0.5, -0.5)]);
        let b = Array1::from_vec(vec![
            Complex64::new(1.0, 2.0 + 1e-9),
            Complex64::new(0.5, -0.5),
        ]);
        assert_close!(&a, &b, complex_array, tol = 1e-8);
    }
}
