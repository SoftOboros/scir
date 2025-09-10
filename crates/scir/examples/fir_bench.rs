use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::env;
use std::time::Instant;

fn make_random(b: usize, n: usize, k: usize, seed: u64) -> (Array2<f32>, Array1<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f32>::zeros((b, n));
    for mut row in x.axis_iter_mut(ndarray::Axis(0)) {
        for v in row.iter_mut() {
            *v = rng.gen::<f32>() * 2.0 - 1.0;
        }
    }
    let taps = Array1::from(
        (0..k)
            .map(|i| 1.0f32 / (i as f32 + 1.0))
            .collect::<Vec<_>>(),
    );
    (x, taps)
}

fn parse_arg_usize(args: &[String], key: &str, default: usize) -> usize {
    let mut i = 0;
    while i + 1 < args.len() {
        if args[i] == key {
            if let Ok(v) = args[i + 1].parse::<usize>() {
                return v;
            }
        }
        i += 1;
    }
    default
}

fn parse_arg_u64(args: &[String], key: &str, default: u64) -> u64 {
    let mut i = 0;
    while i + 1 < args.len() {
        if args[i] == key {
            if let Ok(v) = args[i + 1].parse::<u64>() {
                return v;
            }
        }
        i += 1;
    }
    default
}

fn main() {
    let args: Vec<String> = env::args().collect();
    // Flags: --b <batches> --n <len> --k <taps> --seed <seed>
    let b = parse_arg_usize(&args, "--b", 64);
    let n = parse_arg_usize(&args, "--n", 1 << 14);
    let k = parse_arg_usize(&args, "--k", 31);
    let seed = parse_arg_u64(&args, "--seed", 42);
    let (x, taps) = make_random(b, n, k, seed);

    // CPU baseline
    let t0 = Instant::now();
    let _y_cpu = scir_gpu::fir1d_batched_f32(&x, &taps);
    let dt_cpu = t0.elapsed();
    println!("CPU FIR: {:?} for {}x{} (k={})", dt_cpu, b, n, k);

    // CUDA (if enabled + available); falls back to CPU internally if unavailable
    #[cfg(feature = "gpu")]
    {
        use scir::gpu::{signal as gpu_signal, Device};
        // warmup
        let _ = gpu_signal::fir1d_batched_f32(&x, &taps, Device::Cuda);
        let t1 = Instant::now();
        let y_gpu = gpu_signal::fir1d_batched_f32(&x, &taps, Device::Cuda);
        let dt_gpu = t1.elapsed();
        println!("CUDA FIR (or CPU fallback): {:?}", dt_gpu);

        // Sanity: shapes must match
        assert_eq!(y_cpu.dim(), y_gpu.dim());
    }
}
