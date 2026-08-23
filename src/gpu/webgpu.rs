//! Initial WebGPU backend boundary.
//!
//! This phase deliberately has no `wgpu` dependency and performs no device
//! work. It proves that a validated semantic IR can be accepted by a
//! backend-specific type. Rendering remains an explicit error until the
//! software-BVH and shader/resource lowering design is implemented.

use super::compiler::GpuCompiledScene;
use super::ir::{GpuRenderConfig, GpuSceneView};

#[derive(Clone, Copy, Debug, Default)]
pub struct WebGpuPrepareOptions;

#[derive(Clone, Debug, PartialEq)]
pub struct WebGpuExecutableScene {
    scene: GpuCompiledScene,
}

impl WebGpuExecutableScene {
    pub fn scene(&self) -> GpuSceneView<'_> {
        self.scene.view()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WebGpuBackendError {
    RenderNotImplemented,
}

#[derive(Default)]
pub struct WebGpuRenderer;

impl WebGpuRenderer {
    pub fn prepare(
        &mut self,
        scene: &GpuCompiledScene,
        _options: &WebGpuPrepareOptions,
    ) -> Result<WebGpuExecutableScene, WebGpuBackendError> {
        Ok(WebGpuExecutableScene {
            scene: scene.clone(),
        })
    }

    pub fn render(
        &mut self,
        _scene: &WebGpuExecutableScene,
        _config: &GpuRenderConfig,
    ) -> Result<(), WebGpuBackendError> {
        Err(WebGpuBackendError::RenderNotImplemented)
    }
}
