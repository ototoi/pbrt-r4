pub mod ir;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "webgpu")]
pub mod webgpu;
