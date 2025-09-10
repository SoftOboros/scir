//! Linear algebra utilities for SciR (solve, SVD, QR)

use ndarray::{Array1, Array2};

/// Solve linear system A x = b for x.
#[cfg(feature = "blas")]
pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    use ndarray_linalg::Solve;
    a.solve_into(b.clone()).expect("solve failed")
}

/// Compute thin SVD: A ≈ U S V^T.
/// Returns (U, S, Vt)
#[cfg(feature = "blas")]
pub fn svd(a: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    use ndarray_linalg::SVD;
    let (u_opt, s, vt_opt) = a.svd(true, true).expect("svd failed");
    let u = u_opt.expect("U requested but not returned");
    let vt = vt_opt.expect("Vt requested but not returned");
    (u, s, vt)
}

/// QR decomposition: A = Q R (economy)
#[cfg(feature = "blas")]
pub fn qr(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    use ndarray_linalg::QRInto;
    let (q, r) = a.clone().qr_into().expect("qr failed");
    (q, r)
}

// --- Faer backend placeholders (to be completed) ---

/// Solve linear system with `faer` feature enabled.
/// Placeholder: native faer-backed routines to be wired in a follow-up.
#[cfg(all(feature = "faer", not(feature = "blas")))]
pub fn solve(_a: &Array2<f64>, _b: &Array1<f64>) -> Array1<f64> {
    panic!("faer backend not yet implemented");
}

/// SVD via `faer` feature (temporary ndarray-linalg path).
#[cfg(all(feature = "faer", not(feature = "blas")))]
pub fn svd(_a: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    panic!("faer backend not yet implemented");
}

/// QR via `faer` feature (temporary ndarray-linalg path).
#[cfg(all(feature = "faer", not(feature = "blas")))]
pub fn qr(_a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    panic!("faer backend not yet implemented");
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    use ndarray_npy::ReadNpyExt;
    use proptest::prelude::*;
    use scir_core::assert_close;
    use std::{fs::File, path::PathBuf};

    fn fixtures_base() -> Option<PathBuf> {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        if base.exists() { Some(base) } else { None }
    }

    #[test]
    #[cfg(feature = "blas")]
    fn solve_matches_fixture() {
        let Some(base) = fixtures_base() else {
            eprintln!("[scir-linalg] fixtures missing; skipping solve_matches_fixture");
            return;
        };
        let a: Array2<f64> =
            ReadNpyExt::read_npy(File::open(base.join("lin_solve_A.npy")).unwrap()).unwrap();
        let b: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("lin_solve_b.npy")).unwrap()).unwrap();
        let x_expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("lin_solve_x.npy")).unwrap()).unwrap();
        let x = solve(&a, &b);
        assert_close!(&x, &x_expected, array, atol = 1e-9, rtol = 1e-9);
    }

    #[test]
    #[cfg(feature = "blas")]
    fn svd_matches_fixture() {
        let Some(base) = fixtures_base() else {
            eprintln!("[scir-linalg] fixtures missing; skipping svd_matches_fixture");
            return;
        };
        let a: Array2<f64> =
            ReadNpyExt::read_npy(File::open(base.join("svd_A.npy")).unwrap()).unwrap();
        let (u, s, vt) = svd(&a);
        // Verify A ≈ U diag(S) Vt
        let m = a.nrows();
        let n = a.ncols();
        let k = s.len();
        let mut s_mat = Array2::<f64>::zeros((m, n));
        for i in 0..k {
            s_mat[[i, i]] = s[i];
        }
        let a_recon = u.dot(&s_mat).dot(&vt);
        let a_recon_vec = a_recon.to_owned().into_raw_vec();
        let a_vec = a.to_owned().into_raw_vec();
        assert_close!(&a_recon_vec, &a_vec, slice, atol = 1e-8, rtol = 1e-8);
    }

    #[test]
    #[cfg(feature = "blas")]
    fn qr_matches_fixture() {
        let Some(base) = fixtures_base() else {
            eprintln!("[scir-linalg] fixtures missing; skipping qr_matches_fixture");
            return;
        };
        let a: Array2<f64> =
            ReadNpyExt::read_npy(File::open(base.join("qr_A.npy")).unwrap()).unwrap();
        let (q, r) = qr(&a);
        let a_recon = q.dot(&r);
        let a_recon_vec = a_recon.to_owned().into_raw_vec();
        let a_vec = a.to_owned().into_raw_vec();
        assert_close!(&a_recon_vec, &a_vec, slice, atol = 1e-8, rtol = 1e-8);
    }

    // Property test: for SPD A, solve(A, b) should satisfy A x ≈ b
    #[test]
    #[cfg(feature = "blas")]
    fn solve_spd_property() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..16 {
            // 5x5 SPD matrix
            let m = 5usize;
            let n = 5usize;
            let vals: Vec<f64> = (0..(m * n)).map(|_| rng.random::<f64>()).collect();
            let a_rand = Array2::from_shape_vec((m, n), vals).unwrap();
            let a = a_rand.t().dot(&a_rand) + Array2::<f64>::eye(m);
            let b_vals: Vec<f64> = (0..m).map(|_| rng.random::<f64>()).collect();
            let b = Array1::from_vec(b_vals);
            let x = solve(&a, &b);
            let b_recon = a.dot(&x);
            assert_close!(&b_recon, &b, array, atol = 1e-8, rtol = 1e-8);
        }
    }
}
