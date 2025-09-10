//! scir umbrella crate: re-exports core crates and provides feature aggregation.
#![deny(missing_docs)]

pub use scir_core as core;
pub use scir_fft as fft;
pub use scir_nd as nd;
pub use scir_signal as signal;

#[cfg(feature = "gpu")]
pub mod gpu {
    //! GPU re-exports (enabled with the `gpu` feature).
    pub use scir_gpu::{DType, Device, DeviceArray};
    pub use scir_signal::gpu as signal;
}
