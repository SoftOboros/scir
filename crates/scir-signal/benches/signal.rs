use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;

fn gen_input(n: usize) -> Array1<f64> {
    Array1::from_iter((0..n).map(|i| (i as f64) / (n as f64)))
}

fn bench_sosfilt(c: &mut Criterion) {
    let sos = scir_signal::butter(6, 0.15);
    let x = gen_input(8192);
    c.bench_function("sosfilt_8k_butter6", |b| {
        b.iter(|| {
            let y = scir_signal::sosfilt(&sos, black_box(&x));
            black_box(y)
        })
    });
}

fn bench_filtfilt(c: &mut Criterion) {
    let sos = scir_signal::butter(4, 0.2);
    let x = gen_input(4096);
    c.bench_function("filtfilt_4k_butter4", |b| {
        b.iter(|| {
            let y = scir_signal::filtfilt(&sos, black_box(&x));
            black_box(y)
        })
    });
}

criterion_group!(benches, bench_sosfilt, bench_filtfilt);
criterion_main!(benches);
