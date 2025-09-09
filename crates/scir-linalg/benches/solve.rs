use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{array, Array1, Array2};

#[cfg(feature = "blas")]
fn bench_solve(c: &mut Criterion) {
    // Small, well-conditioned system
    let a: Array2<f64> = array![[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0],];
    let b: Array1<f64> = array![1.0, 2.0, 3.0];

    c.bench_function("linalg_solve_3x3", |bencher| {
        bencher.iter(|| {
            let x = scir_linalg::solve(black_box(&a), black_box(&b));
            black_box(x)
        })
    });
}

#[cfg(feature = "blas")]
criterion_group!(benches, bench_solve);
#[cfg(feature = "blas")]
criterion_main!(benches);

// When BLAS feature is not enabled, provide empty bench to avoid build errors
#[cfg(not(feature = "blas"))]
fn dummy(_c: &mut Criterion) {}
#[cfg(not(feature = "blas"))]
criterion_group!(benches, dummy);
#[cfg(not(feature = "blas"))]
criterion_main!(benches);
