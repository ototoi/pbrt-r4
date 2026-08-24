//! Native WebGPU backend boundary.

mod device;
mod error;
mod geometry;
mod renderer;
mod software;

pub use device::{AccelerationMode, BackendPreference, PowerPreference, PrepareOptions};
pub use error::{BackendError, PlanError};
pub use geometry::{
    index_bytes, light_bytes, material_bytes, primitive_bytes, tlas_transform, transform_bytes,
    vertex_bytes, BlasPlan, ScenePlan, TlasInstancePlan,
};
pub use renderer::{ExecutableScene, Renderer};
pub use software::SoftwareBvhPlan;
