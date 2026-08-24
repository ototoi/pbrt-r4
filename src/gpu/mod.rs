//! GPU rendering boundaries.
//!
//! The default CPU build exposes this namespace without compiling a GPU
//! backend. Backend-independent IR and the WebGPU backend are enabled by the
//! corresponding GPU feature.

#[cfg(any(feature = "cuda", feature = "webgpu"))]
pub mod compiler;
#[cfg(any(feature = "cuda", feature = "webgpu"))]
pub mod ir;
#[cfg(feature = "webgpu")]
pub mod webgpu;
