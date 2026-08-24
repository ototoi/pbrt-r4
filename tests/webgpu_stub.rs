#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::compiler::{GpuCompiledScene, GpuSourceMap};
use pbrt_r4::gpu::ir::{
    GeometryId, GpuBounds2i, GpuDiffuseMaterial, GpuGeometry, GpuIrValidationError, GpuIrVersion,
    GpuMaterial, GpuMatrix4x4, GpuPoint3, GpuPrimitive, GpuRenderConfig, GpuRenderOutput,
    GpuRenderRequest, GpuSceneData, GpuSceneDraft, GpuSpectrumResource, GpuSpectrumTexture,
    GpuStaticTransform, GpuTransform, GpuTriangleMesh, MaterialId, SpectrumId, SpectrumTextureId,
    TransformId, CURRENT_IR_VERSION,
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
            spectra: vec![GpuSpectrumResource::Constant { value: 0.5 }],
            float_textures: Vec::new(),
            spectrum_textures: vec![GpuSpectrumTexture::Constant {
                value: SpectrumId(0),
            }],
            texture_mappings: Vec::new(),
            images: Vec::new(),
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
            materials: vec![GpuMaterial::Diffuse(GpuDiffuseMaterial {
                reflectance: SpectrumTextureId(0),
                displacement: None,
                normal_map: None,
            })],
            lights: Vec::new(),
            primitives: vec![GpuPrimitive {
                geometry: GeometryId(0),
                transform: TransformId(0),
                material: Some(MaterialId(0)),
                alpha: None,
                shadow_alpha: None,
                area_light: pbrt_r4::gpu::ir::GpuAreaLightBinding::None,
                reverse_orientation: false,
            }],
            instance_definitions: Vec::new(),
            instances: Vec::new(),
            world_primitives: vec![].into_boxed_slice(),
            world_instances: vec![].into_boxed_slice(),
            render: GpuRenderConfig::default(),
        },
    };
    GpuCompiledScene::new(draft.finish().unwrap(), GpuSourceMap::default())
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
        renderer.render(
            &executable,
            &GpuRenderRequest::new(&GpuRenderConfig::default(), 0, 1).unwrap()
        ),
        Err(pbrt_r4::gpu::webgpu::WebGpuBackendError::RenderNotImplemented)
    ));
}

#[test]
fn webgpu_render_rejects_invalid_sample_range() {
    let mut renderer = WebGpuRenderer;
    let scene = minimal_scene();
    let executable = renderer.prepare(&scene, &WebGpuPrepareOptions).unwrap();
    assert!(matches!(
        renderer.render(
            &executable,
            &GpuRenderRequest {
                sample_start: 1,
                sample_count: 1,
            }
        ),
        Err(pbrt_r4::gpu::webgpu::WebGpuBackendError::InvalidRenderRequest(_))
    ));
}

#[test]
fn render_output_requires_one_rgb_value_per_pixel() {
    let request = GpuRenderRequest {
        sample_start: 0,
        sample_count: 1,
    };
    let error = GpuRenderOutput::new(
        GpuBounds2i {
            min: [0, 0],
            max: [2, 1],
        },
        vec![[0.0, 0.0, 0.0]].into_boxed_slice(),
        request,
    )
    .unwrap_err();
    assert_eq!(
        error,
        pbrt_r4::gpu::ir::GpuRenderOutputError::PixelCountMismatch {
            expected: 2,
            actual: 1,
        }
    );
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
            spectra: vec![GpuSpectrumResource::Constant { value: 0.5 }],
            float_textures: Vec::new(),
            spectrum_textures: Vec::new(),
            texture_mappings: Vec::new(),
            images: Vec::new(),
            geometry: Vec::new(),
            materials: Vec::new(),
            lights: Vec::new(),
            primitives: Vec::new(),
            instance_definitions: Vec::new(),
            instances: Vec::new(),
            world_primitives: Box::new([]),
            world_instances: Box::new([]),
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
            spectra: Vec::new(),
            float_textures: Vec::new(),
            spectrum_textures: Vec::new(),
            texture_mappings: Vec::new(),
            images: Vec::new(),
            geometry: vec![GpuGeometry::TriangleMesh(GpuTriangleMesh {
                positions: vec![GpuPoint3([0.0, 0.0, 0.0])],
                indices: vec![[0, 1, 0]],
                normals: None,
                tangents: None,
                uvs: None,
                face_indices: None,
            })],
            materials: vec![GpuMaterial::Diffuse(GpuDiffuseMaterial {
                reflectance: SpectrumTextureId(0),
                displacement: None,
                normal_map: None,
            })],
            lights: Vec::new(),
            primitives: Vec::new(),
            instance_definitions: Vec::new(),
            instances: Vec::new(),
            world_primitives: Box::new([]),
            world_instances: Box::new([]),
            render: GpuRenderConfig::default(),
        },
    };
    let errors = draft.finish().unwrap_err();
    assert!(errors
        .issues()
        .iter()
        .any(|issue| matches!(issue, GpuIrValidationError::TriangleIndexOutOfBounds { .. })));
}
