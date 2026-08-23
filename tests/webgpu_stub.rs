#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::compiler::GpuCompiledScene;
use pbrt_r4::gpu::ir::{
    GpuIrVersion, GpuRenderConfig, GpuSceneData, GpuSceneDraft, CURRENT_IR_VERSION,
};
use pbrt_r4::gpu::webgpu::{WebGpuPrepareOptions, WebGpuRenderer};

fn minimal_scene() -> GpuCompiledScene {
    let draft = GpuSceneDraft {
        version: CURRENT_IR_VERSION,
        data: GpuSceneData {
            render: GpuRenderConfig::default(),
        },
    };
    GpuCompiledScene::new(draft.finish().unwrap())
}

#[test]
fn webgpu_prepare_accepts_validated_ir() {
    let mut renderer = WebGpuRenderer;
    let executable = renderer
        .prepare(&minimal_scene(), &WebGpuPrepareOptions)
        .unwrap();
    assert_eq!(executable.scene().version, &CURRENT_IR_VERSION);
}

#[test]
fn webgpu_render_is_explicitly_unimplemented() {
    let mut renderer = WebGpuRenderer;
    let scene = minimal_scene();
    let executable = renderer.prepare(&scene, &WebGpuPrepareOptions).unwrap();
    assert!(matches!(
        renderer.render(&executable, &GpuRenderConfig::default()),
        Err(pbrt_r4::gpu::webgpu::WebGpuBackendError::RenderNotImplemented)
    ));
}

#[test]
fn incompatible_ir_version_is_rejected_before_prepare() {
    let draft = GpuSceneDraft {
        version: GpuIrVersion {
            major: CURRENT_IR_VERSION.major + 1,
            minor: 0,
        },
        data: GpuSceneData {
            render: GpuRenderConfig::default(),
        },
    };
    assert!(draft.finish().is_err());
}
