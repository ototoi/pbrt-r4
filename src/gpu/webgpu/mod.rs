//! Native WebGPU backend boundary.

mod device;
mod error;
mod geometry;
mod renderer;
mod software;

pub use device::{AccelerationMode, PowerPreference, PrepareOptions};
pub use error::{BackendError, PlanError};
pub use geometry::{tlas_transform, BlasPlan, ScenePlan, TlasInstancePlan};
pub use renderer::{ExecutableScene, Renderer};
pub use software::SoftwareBvhPlan;
