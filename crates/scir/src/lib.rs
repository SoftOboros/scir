//! scir umbrella crate: re-exports core crates and provides feature aggregation.

pub use scir_core as core;
pub use scir_fft as fft;
pub use scir_nd as nd;
pub use scir_signal as signal;

#[cfg(feature = "gpu")]
pub mod gpu {
    pub use scir_gpu::{DType, Device, DeviceArray};
    pub use scir_signal::gpu as signal;
}
