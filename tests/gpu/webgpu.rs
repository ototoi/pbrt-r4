#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::compiler::{GpuCompiledScene, GpuSourceMap};
use pbrt_r4::gpu::ir::{
    GeometryId, GpuBounds2i, GpuDiffuseMaterial, GpuGeometry, GpuIrValidationError, GpuIrVersion,
    GpuLight, GpuMaterial, GpuMatrix4x4, GpuPoint3, GpuPointLight, GpuPrimitive, GpuRenderConfig,
    GpuRenderOutput, GpuRenderRequest, GpuSceneData, GpuSceneDraft, GpuSpectrumResource,
    GpuSpectrumTexture, GpuStaticTransform, GpuTransform, GpuTriangleMesh, MaterialId, PrimitiveId,
    SpectrumId, SpectrumTextureId, TransformId, CURRENT_IR_VERSION,
};
use pbrt_r4::gpu::webgpu::{
    index_bytes, light_bytes, material_bytes, primitive_bytes, tlas_transform, transform_bytes,
    vertex_bytes, AccelerationMode, BackendPreference, MaterialReflectancePlan, PlanError,
    PrepareOptions, Renderer, ScenePlan, SoftwareBvhPlan,
};

fn minimal_scene() -> GpuCompiledScene {
    let identity = GpuMatrix4x4([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    let camera_from_camera = GpuMatrix4x4([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    let camera_from_raster = GpuMatrix4x4([
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, -2.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    let static_transform = |render_from_object, object_from_render| {
        GpuTransform::Static(GpuStaticTransform {
            render_from_object,
            object_from_render,
            swaps_handedness: false,
        })
    };
    let draft = GpuSceneDraft {
        version: CURRENT_IR_VERSION,
        data: GpuSceneData {
            transforms: vec![
                static_transform(identity, identity),
                static_transform(
                    camera_from_camera,
                    GpuMatrix4x4([
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, -2.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]),
                ),
                static_transform(
                    camera_from_camera,
                    GpuMatrix4x4([
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, -2.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]),
                ),
            ],
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
            lights: vec![GpuLight::Point(GpuPointLight {
                render_from_light: TransformId(2),
                intensity: SpectrumId(0),
                scale: 1.0,
            })],
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
            world_primitives: vec![PrimitiveId(0)].into_boxed_slice(),
            world_instances: vec![].into_boxed_slice(),
            render: GpuRenderConfig {
                camera: pbrt_r4::gpu::ir::GpuPerspectiveCamera {
                    render_from_camera: TransformId(1),
                    camera_from_raster,
                    lens_radius: 0.0,
                    focal_distance: 1.0,
                    shutter_open: 0.0,
                    shutter_close: 1.0,
                },
                ..Default::default()
            },
        },
    };
    GpuCompiledScene::new(draft.finish().unwrap(), GpuSourceMap::default())
}

#[test]
fn scene_plan_lowers_triangle_geometry_and_transform() {
    let scene = minimal_scene();
    let plan = ScenePlan::from_scene(scene.view()).unwrap();
    assert_eq!(plan.vertices.len(), 3);
    assert_eq!(plan.indices, vec![0, 1, 2]);
    assert_eq!(plan.blases[0].first_vertex, 0);
    assert_eq!(plan.blases[0].vertex_count, 3);
    assert_eq!(plan.blases[0].first_index, 0);
    assert_eq!(plan.blases[0].index_count, 3);
    assert_eq!(plan.tlas_instances[0].blas, 0);
    assert_eq!(plan.tlas_instances[0].custom_data, 0);
    assert_eq!(
        plan.tlas_instances[0].transform,
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    );
    assert_eq!(plan.primitives[0].material, 0);
    assert_eq!(
        plan.materials[0].reflectance,
        MaterialReflectancePlan::Constant([0.5, 0.5, 0.5, 1.0])
    );
    assert_eq!(plan.lights[0].position, [0.0, 0.0, 2.0, 1.0]);
    assert_eq!(plan.lights[0].intensity, [0.5, 0.5, 0.5, 1.0]);
}

#[test]
fn gpu_buffer_serialization_matches_wgsl_layout() {
    let plan = ScenePlan::from_scene(minimal_scene().view()).unwrap();
    let vertices = vertex_bytes(&plan);
    let indices = index_bytes(&plan);
    let primitives = primitive_bytes(&plan);
    let materials = material_bytes(&plan);
    let transforms = transform_bytes(&plan);
    let lights = light_bytes(&plan);

    assert_eq!(vertices.len(), 3 * 32);
    assert_eq!(&vertices[0..4], &0.0f32.to_le_bytes());
    assert_eq!(&vertices[12..16], &0.0f32.to_le_bytes());
    assert_eq!(&vertices[20..24], &0.0f32.to_le_bytes());
    assert_eq!(
        indices,
        [0u32, 1, 2]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(primitives.len(), 16);
    assert_eq!(materials.len(), 32);
    assert_eq!(&materials[0..4], &0.5f32.to_le_bytes());
    assert_eq!(&materials[12..16], &1.0f32.to_le_bytes());
    assert_eq!(transforms.len(), 64);
    assert_eq!(lights.len(), 32);
    assert_eq!(&lights[8..12], &2.0f32.to_le_bytes());
}

#[test]
fn scene_plan_rejects_custom_data_above_24_bits() {
    assert_eq!(ScenePlan::validate_custom_data(0x00ff_ffff), Ok(()));
    assert_eq!(
        ScenePlan::validate_custom_data(0x0100_0000),
        Err(PlanError::LimitExceeded {
            resource: "tlas_instance_custom_data",
            value: 0x0100_0000,
            maximum: 0x00ff_ffff,
        })
    );
}

#[test]
fn tlas_transform_preserves_the_required_three_by_four_rows() {
    assert_eq!(
        tlas_transform([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
}

#[test]
fn software_bvh_plan_has_leaf_ranges_for_triangle_scene() {
    let plan = ScenePlan::from_scene(minimal_scene().view()).unwrap();
    let bvh = SoftwareBvhPlan::from_scene(&plan).unwrap();
    assert_eq!(bvh.nodes.len(), 1);
    assert_eq!(bvh.nodes[0].first, 0);
    assert_eq!(bvh.nodes[0].count, 1);
    assert_eq!(bvh.nodes[0].flags & 1, 1);
    assert_eq!(bvh.primitives[0].primitive, 0);
    assert_eq!(bvh.primitives[0].triangle, 0);
}

#[test]
fn prepare_rejects_zero_texture_limit_before_device_creation() {
    assert!(matches!(
        Renderer::new(&PrepareOptions {
            max_texture_dimension_2d: Some(0),
            ..Default::default()
        }),
        Err(pbrt_r4::gpu::webgpu::BackendError::InvalidPrepareOptions { .. })
    ));
}

#[test]
fn adapter_name_mismatch_does_not_fallback_to_another_adapter() {
    let result = Renderer::new(&PrepareOptions {
        adapter_name: Some("pbrt-r4 adapter that does not exist".to_string()),
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    });
    assert!(matches!(
        result,
        Err(pbrt_r4::gpu::webgpu::BackendError::AdapterRequest(_))
    ));
}

#[test]
fn selected_adapter_info_is_exposed() {
    let renderer = Renderer::new(&PrepareOptions {
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    })
    .unwrap();
    let info = renderer.adapter_info();
    assert!(!info.name.is_empty());
    assert_eq!(renderer.acceleration_mode(), AccelerationMode::SoftwareBvh);
    assert!(renderer.max_texture_dimension_2d() > 0);
}

#[test]
fn explicit_backend_preference_does_not_fallback() {
    let result = Renderer::new(&PrepareOptions {
        backend: BackendPreference::Metal,
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    });
    if cfg!(target_os = "macos") {
        assert!(result.is_ok());
    } else {
        assert!(matches!(
            result,
            Err(pbrt_r4::gpu::webgpu::BackendError::AdapterRequest(_))
        ));
    }
}

#[test]
fn webgpu_prepare_accepts_validated_ir() {
    let Some(mut renderer) = renderer_or_skip() else {
        return;
    };
    let executable = renderer.prepare(&minimal_scene()).unwrap();
    assert_eq!(executable.scene().version, &CURRENT_IR_VERSION);
}

#[test]
fn webgpu_render_returns_a_pixel_buffer() {
    let Some(mut renderer) = renderer_or_skip() else {
        return;
    };
    let scene = minimal_scene();
    let executable = renderer.prepare(&scene).unwrap();
    let output = renderer
        .render(
            &executable,
            &GpuRenderRequest::new(&GpuRenderConfig::default(), 0, 1).unwrap(),
        )
        .unwrap();
    assert_eq!(output.rgb.len(), 1);
    assert!(output.rgb[0][0] > 0.0);
}

#[test]
fn webgpu_render_rejects_invalid_sample_range() {
    let Some(mut renderer) = renderer_or_skip() else {
        return;
    };
    let scene = minimal_scene();
    let executable = renderer.prepare(&scene).unwrap();
    assert!(matches!(
        renderer.render(
            &executable,
            &GpuRenderRequest {
                sample_start: 1,
                sample_count: 1,
            }
        ),
        Err(pbrt_r4::gpu::webgpu::BackendError::InvalidRenderRequest(_))
    ));
}

fn renderer_or_skip() -> Option<Renderer> {
    match Renderer::new(&PrepareOptions::default()) {
        Ok(renderer) => Some(renderer),
        Err(pbrt_r4::gpu::webgpu::BackendError::MissingRayQueryFeature) => {
            eprintln!("skipping hardware WebGPU test: adapter has no ray-query capability");
            None
        }
        Err(error) => panic!("unexpected WebGPU initialization error: {error}"),
    }
}

#[test]
fn webgpu_does_not_fallback_when_ray_query_is_unavailable() {
    match Renderer::new(&PrepareOptions::default()) {
        Ok(_) | Err(pbrt_r4::gpu::webgpu::BackendError::MissingRayQueryFeature) => {}
        Err(error) => panic!("unexpected WebGPU initialization error: {error}"),
    }
}

#[test]
fn software_bvh_renderer_returns_a_pixel_buffer() {
    let mut renderer = Renderer::new(&PrepareOptions {
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    })
    .unwrap();
    let scene = minimal_scene();
    let executable = renderer.prepare(&scene).unwrap();
    let output = renderer
        .render(
            &executable,
            &GpuRenderRequest::new(&GpuRenderConfig::default(), 0, 1).unwrap(),
        )
        .unwrap();
    assert_eq!(output.rgb.len(), 1);
    let expected = 0.25 * (2.0 / (4.125_f32).sqrt()) / 4.125;
    assert!((output.rgb[0][0] - expected).abs() < 1.0e-4);
}

#[test]
fn hardware_and_software_modes_match_the_cpu_reference_scene() {
    let Some(mut hardware) = renderer_or_skip() else {
        return;
    };
    let mut software = Renderer::new(&PrepareOptions {
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    })
    .unwrap();
    let scene = minimal_scene();
    let request = GpuRenderRequest::new(&GpuRenderConfig::default(), 0, 1).unwrap();
    let hardware_scene = hardware.prepare(&scene).unwrap();
    let software_scene = software.prepare(&scene).unwrap();
    let hardware_output = hardware.render(&hardware_scene, &request).unwrap();
    let software_output = software.render(&software_scene, &request).unwrap();
    for (hardware_channel, software_channel) in hardware_output.rgb[0]
        .iter()
        .zip(software_output.rgb[0].iter())
    {
        assert!((hardware_channel - software_channel).abs() < 1.0e-4);
    }
}

#[test]
fn software_renderer_keeps_previous_scene_resources_alive() {
    let mut renderer = Renderer::new(&PrepareOptions {
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    })
    .unwrap();
    let first = renderer.prepare(&minimal_scene()).unwrap();
    let second = renderer.prepare(&minimal_scene()).unwrap();
    let request = GpuRenderRequest::new(&GpuRenderConfig::default(), 0, 1).unwrap();
    assert_eq!(renderer.render(&first, &request).unwrap().rgb.len(), 1);
    assert_eq!(renderer.render(&second, &request).unwrap().rgb.len(), 1);
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
