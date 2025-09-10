use ndarray::{array, Array1, Array2};

#[cfg(feature = "gpu")]
use scir::gpu::{signal as gpu_signal, Device};

fn main() {
    // Small example: two 1D signals batched as rows
    let x: Array2<f32> = array![[1.0, 2.0, 3.0, 4.0], [0.5, 0.0, -0.5, -1.0]];
    let taps: Array1<f32> = array![0.25, 0.5, 0.25];

    // CPU baseline via scir-gpu helper (auto function exists under scir-gpu)
    let y_cpu = scir_gpu::fir1d_batched_f32(&x, &taps);
    println!("CPU FIR output:\n{:?}", y_cpu);

    #[cfg(feature = "gpu")]
    {
        // Attempt CUDA; falls back internally if unavailable
        let y_gpu = gpu_signal::fir1d_batched_f32(&x, &taps, Device::Cuda);
        println!("CUDA FIR output (or CPU fallback):\n{:?}", y_gpu);
    }
}
