#[cfg(feature = "gpu")]
use scir_gpu::Device;
use scir_gpu::{DType, DeviceArray};
use std::env;

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

fn parse_arg_f32(args: &[String], key: &str, default: f32) -> f32 {
    let mut i = 0;
    while i + 1 < args.len() {
        if args[i] == key {
            if let Ok(v) = args[i + 1].parse::<f32>() {
                return v;
            }
        }
        i += 1;
    }
    default
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n = parse_arg_usize(&args, "--n", 8);
    let add_alpha = parse_arg_f32(&args, "--add_alpha", 1.0);
    #[cfg(feature = "gpu")]
    let mul_alpha = parse_arg_f32(&args, "--mul_alpha", 2.0);
    #[cfg(not(feature = "gpu"))]
    let _mul_alpha = parse_arg_f32(&args, "--mul_alpha", 2.0);

    let data = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let arr = DeviceArray::from_cpu_slice(&[n], DType::F32, &data);

    // CPU add-scalar via auto-dispatch (device defaults to CPU)
    let added = arr.add_scalar_auto(add_alpha);
    println!(
        "CPU add_scalar_auto (alpha={}) -> {:?}",
        add_alpha,
        added.to_cpu_vec()
    );

    // Try CUDA path if compiled with the umbrella `gpu` feature
    #[cfg(feature = "gpu")]
    {
        let mut gpu_arr = arr.clone();
        if gpu_arr.to_device(Device::Cuda).is_ok() {
            let scaled = gpu_arr.mul_scalar_auto(mul_alpha);
            println!(
                "CUDA mul_scalar_auto (alpha={}, or CPU fallback) -> {:?}",
                mul_alpha,
                scaled.to_cpu_vec()
            );
        } else {
            eprintln!("CUDA not available; running CPU path only");
        }
    }
}
