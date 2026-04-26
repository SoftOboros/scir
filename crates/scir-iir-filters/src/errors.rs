//! Errors which originate in this library.
//!
//! In std mode (default) the `Error` type derives `thiserror::Error`,
//! getting the upstream-style `Display` + `std::error::Error` impl
//! automatically. In `no_std + alloc` mode (`--no-default-features`)
//! `Display` is hand-written and the `std::error::Error` impl is
//! omitted. Variants and payload shapes are identical across both
//! modes so downstream consumers see the same API.

use alloc::string::String;

#[cfg(feature = "std")]
use crate::errors::Error::IllegalArgument;

#[cfg_attr(feature = "std", derive(thiserror::Error))]
#[derive(Debug, PartialEq)]
/// A custom error type for errors which can occur in this library.
pub enum Error {
    /// Errors caused the user providing an illegal argument to a function.
    #[cfg_attr(feature = "std", error("Illegal Argument: {0}"))]
    IllegalArgument(String),

    /// Errors resulting from logic errors in the code. Not recoverable by the user.
    #[cfg_attr(feature = "std", error("Internal error: {0}"))]
    InternalError(String),
}

#[cfg(not(feature = "std"))]
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::IllegalArgument(s) => write!(f, "Illegal Argument: {s}"),
            Error::InternalError(s) => write!(f, "Internal error: {s}"),
        }
    }
}

#[cfg(feature = "std")]
impl From<std::num::ParseFloatError> for Error {
    fn from(err: std::num::ParseFloatError) -> Self {
        IllegalArgument(err.to_string())
    }
}
