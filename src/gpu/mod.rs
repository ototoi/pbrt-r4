pub mod ir;
pub mod wavefront;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "webgpu")]
pub mod webgpu;
