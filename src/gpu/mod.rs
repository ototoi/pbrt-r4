//! GPU rendering boundaries.
//!
//! The default CPU build exposes this namespace without compiling a GPU
//! backend. Backend-independent IR and the compile-only WebGPU adapter are
//! enabled by the `webgpu` feature.

#[cfg(feature = "webgpu")]
pub mod compiler;
#[cfg(feature = "webgpu")]
pub mod ir;
#[cfg(feature = "webgpu")]
pub mod webgpu;
