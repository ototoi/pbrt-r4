//! Host-side construction boundary for GPU IR.

use super::ir::{GpuSceneIr, GpuSceneView};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct GpuCompiledScene {
    ir: Arc<GpuSceneIr>,
}

impl GpuCompiledScene {
    pub fn new(ir: GpuSceneIr) -> Self {
        Self { ir: Arc::new(ir) }
    }

    pub fn scene(&self) -> &GpuSceneIr {
        &self.ir
    }

    pub fn view(&self) -> GpuSceneView<'_> {
        self.ir.view()
    }
}
