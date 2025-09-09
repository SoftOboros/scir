use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;

fn gen_input(n: usize) -> Array1<f64> {
    Array1::from_iter((0..n).map(|i| i as f64))
}

fn bench_fft(c: &mut Criterion) {
    let x = gen_input(2048);
    c.bench_function("fft_2048", |b| {
        b.iter(|| {
            let y = scir_fft::fft(black_box(&x));
            black_box(y)
        })
    });
}

fn bench_rfft(c: &mut Criterion) {
    let x = gen_input(4096);
    c.bench_function("rfft_4096", |b| {
        b.iter(|| {
            let y = scir_fft::rfft(black_box(&x));
            black_box(y)
        })
    });
}

criterion_group!(benches, bench_fft, bench_rfft);
criterion_main!(benches);
