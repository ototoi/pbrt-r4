#[cfg(feature = "webgpu")]
pub use super::super::webgpu::integrator::WavefrontPathIntegrator;

#[cfg(feature = "cuda")]
pub use super::super::cuda::integrator::WavefrontPathIntegrator;

#[cfg(not(any(feature = "webgpu", feature = "cuda")))]
mod _impl {
    use crate::displays::Display;
    use crate::gpu::ir::flat::Scene;
    use crate::util::error::PbrtError;

    use std::sync::Arc;
    use std::sync::RwLock;
    pub struct WavefrontPathIntegrator {}

    impl WavefrontPathIntegrator {
        pub fn create(_scene: Scene) -> Result<Self, PbrtError> {
            Err(PbrtError::error("GPU integrator not implemented yet"))
        }

        pub fn render(&mut self) -> Result<(), PbrtError> {
            Err(PbrtError::error("GPU integrator not implemented yet"))
        }

        pub fn add_display(&mut self, _display: &Arc<RwLock<dyn Display>>) {
            // No-op
        }
    }
}

#[cfg(not(any(feature = "webgpu", feature = "cuda")))]
pub use _impl::WavefrontPathIntegrator;
