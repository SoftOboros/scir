//! Optimization routines for SciR

use ndarray::{Array1, Array2, Axis};

/// Nelder-Mead simplex algorithm.
pub fn nelder_mead<F>(f: F, start: Array1<f64>, step: f64, max_iter: usize, tol: f64) -> Array1<f64>
where
    F: Fn(&Array1<f64>) -> f64,
{
    let n = start.len();
    let mut simplex: Vec<(Array1<f64>, f64)> = Vec::with_capacity(n + 1);
    simplex.push((start.clone(), f(&start)));
    for i in 0..n {
        let mut p = start.clone();
        p[i] += step;
        simplex.push((p.clone(), f(&p)));
    }
    for _ in 0..max_iter {
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let best = simplex[0].1;
        let worst = simplex[n].1;
        if (worst - best).abs() < tol {
            break;
        }
        // centroid of all but worst
        let mut centroid = Array1::<f64>::zeros(n);
        for i in 0..n {
            centroid = centroid + &simplex[i].0;
        }
        centroid /= n as f64;
        // reflection
        let reflection = &centroid + (&centroid - &simplex[n].0);
        let f_ref = f(&reflection);
        if f_ref < simplex[0].1 {
            let expansion = &centroid + 2.0 * (&reflection - &centroid);
            let f_exp = f(&expansion);
            if f_exp < f_ref {
                simplex[n] = (expansion, f_exp);
            } else {
                simplex[n] = (reflection, f_ref);
            }
        } else if f_ref < simplex[n - 1].1 {
            simplex[n] = (reflection, f_ref);
        } else {
            let contraction = &centroid + 0.5 * (&simplex[n].0 - &centroid);
            let f_con = f(&contraction);
            if f_con < simplex[n].1 {
                simplex[n] = (contraction, f_con);
            } else {
                // shrink
                let best_point = simplex[0].0.clone();
                for i in 1..=n {
                    simplex[i].0 = &best_point + 0.5 * (&simplex[i].0 - &best_point);
                    simplex[i].1 = f(&simplex[i].0);
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    simplex[0].0.clone()
}

/// BFGS optimization with provided gradient.
pub fn bfgs<F, G>(f: F, grad: G, start: Array1<f64>, max_iter: usize, tol: f64) -> Array1<f64>
where
    F: Fn(&Array1<f64>) -> f64,
    G: Fn(&Array1<f64>) -> Array1<f64>,
{
    let n = start.len();
    let mut x = start;
    let mut h = Array2::<f64>::eye(n);
    for _ in 0..max_iter {
        let g = grad(&x);
        if g.iter().map(|v| v * v).sum::<f64>().sqrt() < tol {
            break;
        }
        let p = -h.dot(&g);
        // backtracking line search
        let mut alpha = 1.0;
        let fx = f(&x);
        let mut x_new = &x + &(alpha * &p);
        let mut f_new = f(&x_new);
        while f_new > fx && alpha > 1e-8 {
            alpha *= 0.5;
            x_new = &x + &(alpha * &p);
            f_new = f(&x_new);
        }
        let s = &x_new - &x;
        let g_new = grad(&x_new);
        let y = &g_new - &g;
        let ys = y.dot(&s);
        if ys.abs() < 1e-12 {
            break;
        }
        let rho = 1.0 / ys;
        let i = Array2::<f64>::eye(n);
        let sy = s
            .view()
            .insert_axis(Axis(1))
            .dot(&y.view().insert_axis(Axis(0)));
        let ys_mat = y
            .view()
            .insert_axis(Axis(1))
            .dot(&s.view().insert_axis(Axis(0)));
        h = (i.clone() - rho * sy).dot(&h).dot(&(i - rho * ys_mat))
            + rho
                * s.view()
                    .insert_axis(Axis(1))
                    .dot(&s.view().insert_axis(Axis(0)));
        x = x_new;
    }
    x
}

/// Limited-memory BFGS optimization.
pub fn lbfgs<F, G>(
    f: F,
    grad: G,
    start: Array1<f64>,
    max_iter: usize,
    tol: f64,
    m: usize,
) -> Array1<f64>
where
    F: Fn(&Array1<f64>) -> f64,
    G: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut x = start;
    let mut s_list: Vec<Array1<f64>> = Vec::new();
    let mut y_list: Vec<Array1<f64>> = Vec::new();
    let mut rho: Vec<f64> = Vec::new();
    for _ in 0..max_iter {
        let g = grad(&x);
        let g_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if g_norm < tol {
            break;
        }
        let mut q = g.clone();
        let mut alpha = Vec::new();
        for i in (0..s_list.len()).rev() {
            let a = rho[i] * s_list[i].dot(&q);
            alpha.push(a);
            q = &q - &(a * &y_list[i]);
        }
        let mut r = q; // initial Hessian approx is identity
        for (i, s) in s_list.iter().enumerate() {
            let beta = rho[i] * y_list[i].dot(&r);
            let a = alpha[s_list.len() - 1 - i];
            r = &r + &(s * (a - beta));
        }
        let p = -r;
        // backtracking line search
        let mut step = 1.0;
        let fx = f(&x);
        let mut x_new = &x + &(step * &p);
        let mut f_new = f(&x_new);
        while f_new > fx && step > 1e-8 {
            step *= 0.5;
            x_new = &x + &(step * &p);
            f_new = f(&x_new);
        }
        let s = &x_new - &x;
        let g_new = grad(&x_new);
        let y = &g_new - &g;
        let ys = y.dot(&s);
        if ys.abs() < 1e-12 {
            break;
        }
        if s_list.len() == m {
            s_list.remove(0);
            y_list.remove(0);
            rho.remove(0);
        }
        s_list.push(s.clone());
        y_list.push(y.clone());
        rho.push(1.0 / ys);
        x = x_new;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};
    use ndarray_npy::ReadNpyExt;
    use scir_core::assert_close;
    use std::{fs::File, path::PathBuf};

    fn rosenbrock(x: &Array1<f64>) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }

    fn rosenbrock_grad(x: &Array1<f64>) -> Array1<f64> {
        array![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
            200.0 * (x[1] - x[0].powi(2))
        ]
    }

    fn himmelblau(x: &Array1<f64>) -> f64 {
        (x[0].powi(2) + x[1] - 11.0).powi(2) + (x[0] + x[1].powi(2) - 7.0).powi(2)
    }

    fn himmelblau_grad(x: &Array1<f64>) -> Array1<f64> {
        array![
            4.0 * x[0] * (x[0].powi(2) + x[1] - 11.0) + 2.0 * (x[0] + x[1].powi(2) - 7.0),
            2.0 * (x[0].powi(2) + x[1] - 11.0) + 4.0 * x[1] * (x[0] + x[1].powi(2) - 7.0)
        ]
    }

    #[test]
    fn rosenbrock_nelder_mead_matches_fixture() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("rosenbrock_nelder.npy")).unwrap()).unwrap();
        let result = nelder_mead(rosenbrock, array![-1.2, 1.0], 1.0, 2000, 1e-8);
        assert_close!(&result, &expected, array, atol = 1e-5, rtol = 1e-5);
    }

    #[test]
    fn rosenbrock_bfgs_matches_fixture() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("rosenbrock_bfgs.npy")).unwrap()).unwrap();
        let result = bfgs(rosenbrock, rosenbrock_grad, array![-1.2, 1.0], 2000, 1e-8);
        assert_close!(&result, &expected, array, atol = 1e-5, rtol = 1e-5);
    }

    #[test]
    fn himmelblau_nelder_mead_matches_fixture() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("himmelblau_nelder.npy")).unwrap()).unwrap();
        let result = nelder_mead(himmelblau, array![0.0, 0.0], 0.5, 2000, 1e-8);
        assert_close!(&result, &expected, array, atol = 1e-4, rtol = 1e-5);
    }

    #[test]
    fn himmelblau_bfgs_matches_fixture() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("himmelblau_bfgs.npy")).unwrap()).unwrap();
        let result = bfgs(himmelblau, himmelblau_grad, array![0.0, 0.0], 2000, 1e-8);
        assert_close!(&result, &expected, array, atol = 1e-5, rtol = 1e-5);
    }

    #[test]
    fn rosenbrock_lbfgs_matches_fixture() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("rosenbrock_lbfgs.npy")).unwrap()).unwrap();
        let result = lbfgs(
            rosenbrock,
            rosenbrock_grad,
            array![-1.2, 1.0],
            2000,
            1e-8,
            5,
        );
        assert_close!(&result, &expected, array, atol = 1e-5, rtol = 1e-5);
    }

    #[test]
    fn himmelblau_lbfgs_matches_fixture() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        let expected: Array1<f64> =
            ReadNpyExt::read_npy(File::open(base.join("himmelblau_lbfgs.npy")).unwrap()).unwrap();
        let result = lbfgs(himmelblau, himmelblau_grad, array![0.0, 0.0], 2000, 1e-8, 5);
        assert_close!(&result, &expected, array, atol = 1e-5, rtol = 1e-5);
    }
}
