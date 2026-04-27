//! A Rust re-implementation of some of scipy's digital filters.
//!
//! ## SciR fork notes
//!
//! This is `scir-iir-filters`, the SciR maintenance fork of the
//! Apache-2.0 [`iir_filters`](https://github.com/annoybot/iir_filters)
//! crate. The only meaningful divergence is that `Sos.sections` and
//! `SosCoeffs`'s coefficient fields are widened to `pub`, with a
//! borrow accessor `Sos::sections()`. That lets downstream realtime
//! DSP runtimes read designed coefficients out of an `Sos` produced
//! by [`filter_design::butter`] / [`sos::zpk2sos`] and run their own
//! filter loop. See the crate-level README for the rationale.
//!
//! # Example
//!
//! ```rust
//! use scir_iir_filters::filter_design::FilterType;
//! use scir_iir_filters::filter_design::butter;
//! use scir_iir_filters::sos::zpk2sos;
//! use scir_iir_filters::filter::DirectForm2Transposed;
//! use scir_iir_filters::filter::Filter;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let order = 5;
//!     let cutoff_low = 1.0;
//!     let cutoff_hi= 10.0;
//!     let fs = 81.0;
//!
//!     let zpk = butter(order, FilterType::BandPass(cutoff_low, cutoff_hi),fs)?;
//!     let sos = zpk2sos(&zpk, None)?;
//!
//!     let mut dft2 = DirectForm2Transposed::new(&sos);
//!
//!     let input:Vec<f64>  = vec![1.0, 2.0, 3.0];
//!     let mut output:Vec<f64> = vec![];
//!
//!     for x in input.iter() {
//!         output.push( dft2.filter(*x) );
//!     }
//!
//!     return Ok( () );
//! }
//! ```
//! # Notes
//!
//! See: [scipy.signal: butter()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
//!      and [scipy.signal: cheby1()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html)
//!
//! Currently implements Butterworth and Chebyshev Type I across all four
//! [`filter_design::FilterType`] variants. Bessel and Chebyshev II are
//! tracked as follow-on work in the SciR signal-processing roadmap.
//!

#![cfg_attr(nightly, feature(doc_auto_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(clippy::needless_return)]
#![deny(unsafe_code)]
#![deny(unused_must_use)]
#![deny(clippy::panic)]
#![deny(clippy::expect_used)]
#![deny(clippy::unwrap_used)]
#![warn(missing_docs)]

extern crate alloc;

pub mod errors;
pub mod filter_design;

mod bessel_tables;

#[doc(hidden)]
pub mod macros;

mod cplxreal;
pub mod filter;
pub mod sos;
// Reads npy/text fixture files from disk; only meaningful in std mode and
// only used by `#[cfg(test)]` consumers. Gate the module entirely so the
// crate compiles for `no_std + alloc` targets.
#[cfg(feature = "std")]
mod test_util;
mod util;
mod zpk2tf;
