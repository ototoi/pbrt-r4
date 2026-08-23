#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::compiler::GpuCompiledScene;
use pbrt_r4::gpu::ir::{
    GeometryId, GpuGeometry, GpuIrValidationError, GpuIrVersion, GpuMatrix4x4, GpuPoint3,
    GpuPrimitive, GpuRenderConfig, GpuSceneData, GpuSceneDraft, GpuStaticTransform, GpuTransform,
    GpuTriangleMesh, TransformId, CURRENT_IR_VERSION,
};
use pbrt_r4::gpu::webgpu::{WebGpuPrepareOptions, WebGpuRenderer};

fn minimal_scene() -> GpuCompiledScene {
    let draft = GpuSceneDraft {
        version: CURRENT_IR_VERSION,
        data: GpuSceneData {
            transforms: vec![GpuTransform::Static(GpuStaticTransform {
                render_from_object: GpuMatrix4x4([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]),
                object_from_render: GpuMatrix4x4([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]),
                swaps_handedness: false,
            })],
            geometry: vec![GpuGeometry::TriangleMesh(GpuTriangleMesh {
                positions: vec![
                    GpuPoint3([0.0, 0.0, 0.0]),
                    GpuPoint3([1.0, 0.0, 0.0]),
                    GpuPoint3([0.0, 1.0, 0.0]),
                ],
                indices: vec![[0, 1, 2]],
                normals: None,
                tangents: None,
                uvs: None,
                face_indices: None,
            })],
            primitives: vec![GpuPrimitive {
                geometry: GeometryId(0),
                transform: TransformId(0),
                reverse_orientation: false,
            }],
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
            transforms: Vec::new(),
            geometry: Vec::new(),
            primitives: Vec::new(),
            render: GpuRenderConfig::default(),
        },
    };
    assert!(draft.finish().is_err());
}

#[test]
fn invalid_triangle_index_is_rejected() {
    let draft = GpuSceneDraft {
        version: CURRENT_IR_VERSION,
        data: GpuSceneData {
            transforms: Vec::new(),
            geometry: vec![GpuGeometry::TriangleMesh(GpuTriangleMesh {
                positions: vec![GpuPoint3([0.0, 0.0, 0.0])],
                indices: vec![[0, 1, 0]],
                normals: None,
                tangents: None,
                uvs: None,
                face_indices: None,
            })],
            primitives: Vec::new(),
            render: GpuRenderConfig::default(),
        },
    };
    let errors = draft.finish().unwrap_err();
    assert!(errors
        .issues()
        .iter()
        .any(|issue| matches!(issue, GpuIrValidationError::TriangleIndexOutOfBounds { .. })));
}
