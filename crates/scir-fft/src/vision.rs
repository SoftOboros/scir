//! Vision-oriented 2D spectral kernels.
//!
//! These functions are reusable SciR math kernels: 2D FFT, FFT magnitude,
//! log-polar resampling, Fourier-Mellin magnitude signatures, and bin-level
//! pose-shift estimates. They intentionally do not define Streamz frame/time
//! wrappers or Spectral-Pick fingerprint/commit semantics.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Log-polar sampling grid for a 2D magnitude image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogPolarGrid {
    /// Number of log-radius rows in the output.
    pub radial_bins: usize,
    /// Number of theta columns in the output.
    pub angular_bins: usize,
    /// Input-space center y coordinate.
    pub center_y: f64,
    /// Input-space center x coordinate.
    pub center_x: f64,
    /// Smallest radius sampled. Values less than or equal to zero are clamped.
    pub min_radius: f64,
    /// Largest radius sampled.
    pub max_radius: f64,
}

impl LogPolarGrid {
    /// Build a centered grid for an input shape.
    ///
    /// The grid center is `(height - 1) / 2, (width - 1) / 2`; `max_radius`
    /// is the largest radius that stays inside the shorter half-extent.
    pub fn centered(shape: (usize, usize), radial_bins: usize, angular_bins: usize) -> Self {
        let (height, width) = shape;
        let center_y = height.saturating_sub(1) as f64 / 2.0;
        let center_x = width.saturating_sub(1) as f64 / 2.0;
        let max_radius = center_y.min(center_x).max(1.0);
        Self {
            radial_bins,
            angular_bins,
            center_y,
            center_x,
            min_radius: 1.0,
            max_radius,
        }
    }
}

/// Bin-level circular or linear shift estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShiftEstimate {
    /// Shift in bins that best maps the reference profile onto the candidate.
    pub shift_bins: isize,
    /// Normalized correlation score in `[-1, 1]` for the selected shift.
    pub score: f64,
}

/// Bin-level log-polar pose estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogPolarPoseEstimate {
    /// Rotation estimate as a circular theta-bin shift.
    pub rotation: ShiftEstimate,
    /// Scale estimate as a linear log-radius-bin shift.
    pub log_radius: ShiftEstimate,
}

/// Compute the 2D forward FFT of a real-valued image.
pub fn fft2_real(input: &Array2<f64>) -> Array2<Complex64> {
    let complex = input.mapv(|value| Complex64::new(value, 0.0));
    fft2_complex(&complex)
}

/// Compute the 2D forward FFT of a complex-valued image.
pub fn fft2_complex(input: &Array2<Complex64>) -> Array2<Complex64> {
    let mut output = input.clone();
    process_rows(&mut output, false);
    process_columns(&mut output, false);
    output
}

/// Compute the 2D inverse FFT of a complex-valued spectrum.
pub fn ifft2_complex(input: &Array2<Complex64>) -> Array2<Complex64> {
    let mut output = input.clone();
    process_rows(&mut output, true);
    process_columns(&mut output, true);
    output
}

/// Compute the magnitude of a real-valued image's 2D FFT.
pub fn fft2_magnitude(input: &Array2<f64>) -> Array2<f64> {
    fft2_real(input).mapv(|value| value.norm())
}

/// Shift a real-valued 2D array so the zero-frequency component moves to the center.
pub fn fftshift2_real(input: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = input.dim();
    Array2::from_shape_fn((rows, cols), |(y, x)| {
        let source_y = (y + rows / 2) % rows.max(1);
        let source_x = (x + cols / 2) % cols.max(1);
        input[(source_y, source_x)]
    })
}

/// Build a separable 2D Hann window.
pub fn hann2(height: usize, width: usize) -> Array2<f64> {
    Array2::from_shape_fn((height, width), |(y, x)| {
        hann_value(y, height) * hann_value(x, width)
    })
}

/// Multiply an image by a same-shaped 2D window.
///
/// Panics if the two arrays do not have the same shape.
pub fn apply_window2(input: &Array2<f64>, window: &Array2<f64>) -> Array2<f64> {
    assert_eq!(input.dim(), window.dim(), "2D window shape mismatch");
    input * window
}

/// Resample a 2D magnitude image onto a log-polar grid.
pub fn log_polar_resample(input: &Array2<f64>, grid: LogPolarGrid) -> Array2<f64> {
    if grid.radial_bins == 0 || grid.angular_bins == 0 {
        return Array2::zeros((grid.radial_bins, grid.angular_bins));
    }

    let min_radius = grid.min_radius.max(f64::MIN_POSITIVE);
    let max_radius = grid.max_radius.max(min_radius);
    let log_min = min_radius.ln();
    let log_span = max_radius.ln() - log_min;
    let radial_denom = grid.radial_bins.saturating_sub(1).max(1) as f64;

    Array2::from_shape_fn((grid.radial_bins, grid.angular_bins), |(r, theta)| {
        let radius = (log_min + log_span * (r as f64 / radial_denom)).exp();
        let angle = 2.0 * PI * theta as f64 / grid.angular_bins as f64;
        let y = grid.center_y + radius * angle.sin();
        let x = grid.center_x + radius * angle.cos();
        bilinear_sample(input, y, x)
    })
}

/// Compute a Fourier-Mellin invariant magnitude signature for a real image.
///
/// The image is transformed to 2D FFT magnitude, shifted to centered frequency
/// layout, resampled onto the supplied log-polar grid, then transformed again
/// and reduced to magnitude. The log-polar intermediate should be retained by
/// callers that need pose recovery.
pub fn fourier_mellin_magnitude(input: &Array2<f64>, grid: LogPolarGrid) -> Array2<f64> {
    let magnitude = fftshift2_real(&fft2_magnitude(input));
    let log_polar = log_polar_resample(&magnitude, grid);
    fft2_magnitude(&log_polar)
}

/// Sum a log-polar image over theta to produce a radial profile.
pub fn radial_profile(log_polar: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter(log_polar.rows().into_iter().map(|row| row.sum()))
}

/// Sum a log-polar image over log-radius to produce an angular profile.
pub fn angular_profile(log_polar: &Array2<f64>) -> Array1<f64> {
    let (_, cols) = log_polar.dim();
    Array1::from_iter((0..cols).map(|x| log_polar.column(x).sum()))
}

/// Estimate the circular shift that best maps `reference` onto `candidate`.
///
/// Positive `shift_bins` means `candidate[(i + shift) % n]` best matches
/// `reference[i]`.
pub fn estimate_circular_shift(reference: &Array1<f64>, candidate: &Array1<f64>) -> ShiftEstimate {
    assert_eq!(
        reference.len(),
        candidate.len(),
        "circular profile length mismatch"
    );
    let len = reference.len();
    if len == 0 {
        return ShiftEstimate {
            shift_bins: 0,
            score: 0.0,
        };
    }

    let reference_norm = reference.iter().map(|value| value * value).sum::<f64>();
    let candidate_norm = candidate.iter().map(|value| value * value).sum::<f64>();
    let denominator = (reference_norm * candidate_norm).sqrt();
    if denominator == 0.0 {
        return ShiftEstimate {
            shift_bins: 0,
            score: 0.0,
        };
    }

    let mut best = ShiftEstimate {
        shift_bins: 0,
        score: f64::NEG_INFINITY,
    };
    for shift in 0..len {
        let dot = reference
            .iter()
            .enumerate()
            .map(|(i, value)| value * candidate[(i + shift) % len])
            .sum::<f64>();
        let score = dot / denominator;
        if score > best.score {
            best = ShiftEstimate {
                shift_bins: shift as isize,
                score,
            };
        }
    }
    best
}

/// Estimate the bounded linear shift that best maps `reference` onto `candidate`.
///
/// Positive `shift_bins` means `candidate[i + shift]` best matches
/// `reference[i]` over the overlapping range.
pub fn estimate_linear_shift(
    reference: &Array1<f64>,
    candidate: &Array1<f64>,
    max_abs_shift: usize,
) -> ShiftEstimate {
    assert_eq!(
        reference.len(),
        candidate.len(),
        "linear profile length mismatch"
    );
    let len = reference.len();
    if len == 0 {
        return ShiftEstimate {
            shift_bins: 0,
            score: 0.0,
        };
    }

    let max_shift = max_abs_shift.min(len.saturating_sub(1)) as isize;
    let mut best = ShiftEstimate {
        shift_bins: 0,
        score: f64::NEG_INFINITY,
    };

    for shift in -max_shift..=max_shift {
        let mut dot = 0.0;
        let mut reference_norm = 0.0;
        let mut candidate_norm = 0.0;
        for i in 0..len {
            let candidate_index = i as isize + shift;
            if !(0..len as isize).contains(&candidate_index) {
                continue;
            }
            let left = reference[i];
            let right = candidate[candidate_index as usize];
            dot += left * right;
            reference_norm += left * left;
            candidate_norm += right * right;
        }
        let denominator = (reference_norm * candidate_norm).sqrt();
        let score = if denominator == 0.0 {
            0.0
        } else {
            dot / denominator
        };
        if score > best.score {
            best = ShiftEstimate {
                shift_bins: shift,
                score,
            };
        }
    }
    best
}

/// Estimate rotation and scale-bin shifts from two same-shaped log-polar images.
pub fn estimate_log_polar_pose(
    reference_log_polar: &Array2<f64>,
    candidate_log_polar: &Array2<f64>,
) -> LogPolarPoseEstimate {
    assert_eq!(
        reference_log_polar.dim(),
        candidate_log_polar.dim(),
        "log-polar image shape mismatch"
    );
    let rotation = estimate_circular_shift(
        &angular_profile(reference_log_polar),
        &angular_profile(candidate_log_polar),
    );
    let max_scale_shift = reference_log_polar.nrows().saturating_sub(1);
    let log_radius = estimate_linear_shift(
        &radial_profile(reference_log_polar),
        &radial_profile(candidate_log_polar),
        max_scale_shift,
    );
    LogPolarPoseEstimate {
        rotation,
        log_radius,
    }
}

fn process_rows(buffer: &mut Array2<Complex64>, inverse: bool) {
    let cols = buffer.ncols();
    if cols == 0 {
        return;
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(cols)
    } else {
        planner.plan_fft_forward(cols)
    };
    for mut row in buffer.rows_mut() {
        let slice = row
            .as_slice_mut()
            .expect("owned Array2 rows should be contiguous");
        fft.process(slice);
        if inverse {
            let scale = cols as f64;
            for value in slice.iter_mut() {
                *value /= scale;
            }
        }
    }
}

fn process_columns(buffer: &mut Array2<Complex64>, inverse: bool) {
    let rows = buffer.nrows();
    let cols = buffer.ncols();
    if rows == 0 || cols == 0 {
        return;
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(rows)
    } else {
        planner.plan_fft_forward(rows)
    };
    let mut column = vec![Complex64::new(0.0, 0.0); rows];
    for x in 0..cols {
        for y in 0..rows {
            column[y] = buffer[(y, x)];
        }
        fft.process(&mut column);
        if inverse {
            let scale = rows as f64;
            for value in column.iter_mut() {
                *value /= scale;
            }
        }
        for y in 0..rows {
            buffer[(y, x)] = column[y];
        }
    }
}

fn hann_value(index: usize, length: usize) -> f64 {
    match length {
        0 => 0.0,
        1 => 1.0,
        _ => {
            let phase = 2.0 * PI * index as f64 / (length - 1) as f64;
            0.5 * (1.0 - phase.cos())
        }
    }
}

fn bilinear_sample(input: &Array2<f64>, y: f64, x: f64) -> f64 {
    let (height, width) = input.dim();
    if height == 0
        || width == 0
        || y < 0.0
        || x < 0.0
        || y > (height - 1) as f64
        || x > (width - 1) as f64
    {
        return 0.0;
    }

    let y0 = y.floor() as usize;
    let x0 = x.floor() as usize;
    let y1 = (y0 + 1).min(height - 1);
    let x1 = (x0 + 1).min(width - 1);
    let dy = y - y0 as f64;
    let dx = x - x0 as f64;

    let top = input[(y0, x0)] * (1.0 - dx) + input[(y0, x1)] * dx;
    let bottom = input[(y1, x0)] * (1.0 - dx) + input[(y1, x1)] * dx;
    top * (1.0 - dy) + bottom * dy
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fft2_of_impulse_is_constant() {
        let input = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ];

        let spectrum = fft2_real(&input);

        for value in spectrum.iter() {
            assert!((value.re - 1.0).abs() < 1e-12);
            assert!(value.im.abs() < 1e-12);
        }
    }

    #[test]
    fn ifft2_round_trips_real_image() {
        let input = array![
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0]
        ];

        let recovered = ifft2_complex(&fft2_real(&input));

        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert!((actual.re - expected).abs() < 1e-10);
            assert!(actual.im.abs() < 1e-10);
        }
    }

    #[test]
    fn fft2_magnitude_is_invariant_to_circular_translation() {
        let input = array![
            [0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0, 5.0],
            [6.0, 0.0, 0.0, 7.0]
        ];
        let shifted = circular_shift2(&input, 1, 2);

        let a = fft2_magnitude(&input);
        let b = fft2_magnitude(&shifted);

        for (left, right) in a.iter().zip(b.iter()) {
            assert!((left - right).abs() < 1e-10);
        }
    }

    #[test]
    fn fftshift2_moves_quadrants() {
        let input = array![[1.0, 2.0], [3.0, 4.0]];

        let shifted = fftshift2_real(&input);

        assert_eq!(shifted, array![[4.0, 3.0], [2.0, 1.0]]);
    }

    #[test]
    fn hann2_is_separable_and_zero_at_edges() {
        let window = hann2(5, 5);

        assert_eq!(window[(0, 2)], 0.0);
        assert_eq!(window[(2, 0)], 0.0);
        assert!((window[(2, 2)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn log_polar_resample_preserves_constant_image() {
        let input = Array2::from_elem((9, 9), 3.5);
        let grid = LogPolarGrid::centered(input.dim(), 4, 8);

        let sampled = log_polar_resample(&input, grid);

        for value in sampled.iter() {
            assert!((value - 3.5).abs() < 1e-12);
        }
    }

    #[test]
    fn profiles_sum_expected_axes() {
        let log_polar = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        assert_eq!(radial_profile(&log_polar), array![6.0, 15.0]);
        assert_eq!(angular_profile(&log_polar), array![5.0, 7.0, 9.0]);
    }

    #[test]
    fn circular_shift_estimate_detects_theta_bin_shift() {
        let reference = array![0.0, 1.0, 3.0, 7.0, 0.0];
        let candidate = circular_shift1(&reference, 2);

        let estimate = estimate_circular_shift(&reference, &candidate);

        assert_eq!(estimate.shift_bins, 2);
        assert!((estimate.score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn linear_shift_estimate_detects_log_radius_bin_shift() {
        let reference = array![0.0, 1.0, 3.0, 7.0, 0.0];
        let candidate = linear_shift1(&reference, 1);

        let estimate = estimate_linear_shift(&reference, &candidate, 2);

        assert_eq!(estimate.shift_bins, 1);
        assert!((estimate.score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn log_polar_pose_estimate_reports_rotation_and_scale_bins() {
        let reference = array![
            [0.0, 1.0, 4.0, 1.0],
            [0.0, 2.0, 8.0, 2.0],
            [0.0, 3.0, 12.0, 3.0],
            [0.0, 0.0, 0.0, 0.0]
        ];
        let candidate = circular_shift2(&linear_shift2(&reference, 1), 0, 1);

        let estimate = estimate_log_polar_pose(&reference, &candidate);

        assert_eq!(estimate.rotation.shift_bins, 1);
        assert_eq!(estimate.log_radius.shift_bins, 1);
    }

    #[test]
    fn fourier_mellin_signature_has_requested_shape() {
        let input = Array2::from_shape_fn((8, 8), |(y, x)| {
            if (y == 2 && x == 3) || (y == 5 && x == 6) {
                1.0
            } else {
                0.0
            }
        });
        let grid = LogPolarGrid::centered(input.dim(), 4, 8);

        let signature = fourier_mellin_magnitude(&input, grid);

        assert_eq!(signature.dim(), (4, 8));
    }

    fn circular_shift1(input: &Array1<f64>, shift: usize) -> Array1<f64> {
        let len = input.len();
        Array1::from_shape_fn(len, |index| input[(index + len - shift % len) % len])
    }

    fn linear_shift1(input: &Array1<f64>, shift: usize) -> Array1<f64> {
        Array1::from_shape_fn(input.len(), |index| {
            index
                .checked_sub(shift)
                .map(|source| input[source])
                .unwrap_or(0.0)
        })
    }

    fn circular_shift2(input: &Array2<f64>, shift_y: usize, shift_x: usize) -> Array2<f64> {
        let (rows, cols) = input.dim();
        Array2::from_shape_fn((rows, cols), |(y, x)| {
            let source_y = (y + rows - shift_y % rows) % rows;
            let source_x = (x + cols - shift_x % cols) % cols;
            input[(source_y, source_x)]
        })
    }

    fn linear_shift2(input: &Array2<f64>, shift_y: usize) -> Array2<f64> {
        let (rows, cols) = input.dim();
        Array2::from_shape_fn((rows, cols), |(y, x)| {
            y.checked_sub(shift_y)
                .map(|source_y| input[(source_y, x)])
                .unwrap_or(0.0)
        })
    }
}
