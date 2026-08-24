#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::compiler::{GpuCompiledScene, GpuSourceMap};
use pbrt_r4::gpu::ir::{
    FloatTextureId, GeometryId, GpuBounds2i, GpuBounds3, GpuColorEncoding, GpuDiffuseMaterial,
    GpuFloatImageChannel, GpuFloatTexture, GpuGeometry, GpuImageChannels, GpuImageFilter,
    GpuImageResource, GpuImageWrapMode, GpuInstance, GpuInstanceDefinition, GpuIrValidationError,
    GpuIrVersion, GpuLight, GpuMaterial, GpuMatrix4x4, GpuMipLevel, GpuPoint3, GpuPointLight,
    GpuPrimitive, GpuRenderConfig, GpuRenderOutput, GpuRenderRequest, GpuSceneData, GpuSceneDraft,
    GpuSpectrumResource, GpuSpectrumTexture, GpuSpectrumType, GpuStaticTransform, GpuTexelStorage,
    GpuTextureMapping, GpuTransform, GpuTriangleMesh, ImageId, InstanceDefinitionId, InstanceId,
    MaterialId, PrimitiveId, SpectrumId, SpectrumTextureId, TextureMappingId, TransformId,
    CURRENT_IR_VERSION,
};
use pbrt_r4::gpu::webgpu::{
    index_bytes, light_bytes, material_bytes, primitive_bytes, texture_bytes, tlas_transform,
    transform_bytes, vertex_bytes, AccelerationMode, BackendPreference, MaterialReflectancePlan,
    PlanError, PrepareOptions, Renderer, ScenePlan, SoftwareBvhPlan,
};

fn minimal_scene_draft() -> GpuSceneDraft {
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
    draft
}

fn minimal_scene() -> GpuCompiledScene {
    GpuCompiledScene::new(
        minimal_scene_draft().finish().unwrap(),
        GpuSourceMap::default(),
    )
}

fn image_scene() -> GpuCompiledScene {
    let mut draft = minimal_scene_draft();
    draft.data.texture_mappings = vec![GpuTextureMapping::Uv {
        su: 1.0,
        sv: 1.0,
        du: 0.0,
        dv: 0.0,
    }];
    let mip = |count| {
        vec![GpuMipLevel {
            resolution: [1, 1],
            texel_offset: 0,
            texel_count: count,
        }]
        .into_boxed_slice()
    };
    draft.data.images = vec![
        GpuImageResource {
            resolution: [1, 1],
            channels: GpuImageChannels::Rgba,
            storage: GpuTexelStorage::U8(vec![255, 128, 0, 64].into_boxed_slice()),
            mip_levels: mip(4),
            color_encoding: GpuColorEncoding::Srgb,
        },
        GpuImageResource {
            resolution: [1, 1],
            channels: GpuImageChannels::R,
            storage: GpuTexelStorage::F16(vec![0x3800].into_boxed_slice()),
            mip_levels: mip(1),
            color_encoding: GpuColorEncoding::Linear,
        },
        GpuImageResource {
            resolution: [1, 1],
            channels: GpuImageChannels::R,
            storage: GpuTexelStorage::F32(vec![0.25].into_boxed_slice()),
            mip_levels: mip(1),
            color_encoding: GpuColorEncoding::Linear,
        },
    ];
    draft.data.spectrum_textures = vec![GpuSpectrumTexture::Image {
        image: ImageId(0),
        mapping: TextureMappingId(0),
        scale: 2.0,
        invert: false,
        swrap: GpuImageWrapMode::Clamp,
        twrap: GpuImageWrapMode::Clamp,
        filter: GpuImageFilter::Bilinear,
        spectrum_type: GpuSpectrumType::Unbounded,
    }];
    draft.data.float_textures = vec![
        GpuFloatTexture::Image {
            image: ImageId(0),
            mapping: TextureMappingId(0),
            scale: 1.0,
            invert: false,
            swrap: GpuImageWrapMode::Repeat,
            twrap: GpuImageWrapMode::Repeat,
            filter: GpuImageFilter::Point,
            channel: GpuFloatImageChannel::Alpha,
        },
        GpuFloatTexture::Image {
            image: ImageId(1),
            mapping: TextureMappingId(0),
            scale: 1.0,
            invert: false,
            swrap: GpuImageWrapMode::Repeat,
            twrap: GpuImageWrapMode::Repeat,
            filter: GpuImageFilter::Point,
            channel: GpuFloatImageChannel::Channel0,
        },
        GpuFloatTexture::Image {
            image: ImageId(2),
            mapping: TextureMappingId(0),
            scale: 1.0,
            invert: false,
            swrap: GpuImageWrapMode::Repeat,
            twrap: GpuImageWrapMode::Repeat,
            filter: GpuImageFilter::Point,
            channel: GpuFloatImageChannel::Channel0,
        },
    ];
    if let GpuGeometry::TriangleMesh(mesh) = &mut draft.data.geometry[0] {
        mesh.uvs = Some(vec![
            pbrt_r4::gpu::ir::GpuPoint2([0.0, 0.0]),
            pbrt_r4::gpu::ir::GpuPoint2([1.0, 0.0]),
            pbrt_r4::gpu::ir::GpuPoint2([0.0, 1.0]),
        ]);
    }
    GpuCompiledScene::new(draft.finish().unwrap(), GpuSourceMap::default())
}

fn mipmap_lod_scene(filter: GpuImageFilter) -> GpuCompiledScene {
    let mut draft = minimal_scene_draft();
    draft.data.texture_mappings = vec![GpuTextureMapping::Uv {
        su: 1.0,
        sv: 1.0,
        du: 0.0,
        dv: 0.0,
    }];
    draft.data.images = vec![GpuImageResource {
        resolution: [2, 1],
        channels: GpuImageChannels::Rgb,
        storage: GpuTexelStorage::F32(
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0].into_boxed_slice(),
        ),
        mip_levels: vec![
            GpuMipLevel {
                resolution: [2, 1],
                texel_offset: 0,
                texel_count: 6,
            },
            GpuMipLevel {
                resolution: [1, 1],
                texel_offset: 6,
                texel_count: 3,
            },
        ]
        .into_boxed_slice(),
        color_encoding: GpuColorEncoding::Linear,
    }];
    draft.data.spectrum_textures = vec![GpuSpectrumTexture::Image {
        image: ImageId(0),
        mapping: TextureMappingId(0),
        scale: 1.0,
        invert: false,
        swrap: GpuImageWrapMode::Repeat,
        twrap: GpuImageWrapMode::Repeat,
        filter,
        spectrum_type: GpuSpectrumType::Unbounded,
    }];
    if let GpuGeometry::TriangleMesh(mesh) = &mut draft.data.geometry[0] {
        mesh.uvs = Some(vec![
            pbrt_r4::gpu::ir::GpuPoint2([0.0, 0.0]),
            pbrt_r4::gpu::ir::GpuPoint2([1.0, 0.0]),
            pbrt_r4::gpu::ir::GpuPoint2([0.0, 1.0]),
        ]);
    }
    GpuCompiledScene::new(draft.finish().unwrap(), GpuSourceMap::default())
}

fn uniform_infinite_scene() -> GpuCompiledScene {
    let mut draft = minimal_scene_draft();
    draft.data.lights = vec![pbrt_r4::gpu::ir::GpuLight::UniformInfinite(
        pbrt_r4::gpu::ir::GpuUniformInfiniteLight {
            radiance: SpectrumId(0),
            scale: 1.0,
        },
    )];
    GpuCompiledScene::new(draft.finish().unwrap(), GpuSourceMap::default())
}

fn alpha_scene(alpha: f32) -> GpuCompiledScene {
    let mut draft = minimal_scene_draft();
    draft.data.float_textures = vec![GpuFloatTexture::Constant { value: alpha }];
    draft.data.primitives[0].alpha = Some(FloatTextureId(0));
    GpuCompiledScene::new(draft.finish().unwrap(), GpuSourceMap::default())
}

fn normal_map_scene() -> GpuCompiledScene {
    let mut draft = minimal_scene_draft();
    draft.data.images = vec![GpuImageResource {
        resolution: [1, 1],
        channels: GpuImageChannels::Rgb,
        storage: GpuTexelStorage::F32(vec![1.0, 0.5, 0.5].into_boxed_slice()),
        mip_levels: vec![GpuMipLevel {
            resolution: [1, 1],
            texel_offset: 0,
            texel_count: 3,
        }]
        .into_boxed_slice(),
        color_encoding: GpuColorEncoding::Linear,
    }];
    if let GpuGeometry::TriangleMesh(mesh) = &mut draft.data.geometry[0] {
        mesh.uvs = Some(vec![
            pbrt_r4::gpu::ir::GpuPoint2([0.0, 0.0]),
            pbrt_r4::gpu::ir::GpuPoint2([1.0, 0.0]),
            pbrt_r4::gpu::ir::GpuPoint2([0.0, 1.0]),
        ]);
    }
    let GpuMaterial::Diffuse(material) = &mut draft.data.materials[0];
    material.normal_map = Some(ImageId(0));
    GpuCompiledScene::new(draft.finish().unwrap(), GpuSourceMap::default())
}

fn bump_map_scene() -> GpuCompiledScene {
    let mut draft = minimal_scene_draft();
    draft.data.texture_mappings = vec![GpuTextureMapping::Uv {
        su: 0.1,
        sv: 0.1,
        du: 0.0,
        dv: 0.0,
    }];
    draft.data.images = vec![GpuImageResource {
        resolution: [2, 1],
        channels: GpuImageChannels::R,
        storage: GpuTexelStorage::F32(vec![0.0, 1.0, 0.5].into_boxed_slice()),
        mip_levels: vec![
            GpuMipLevel {
                resolution: [2, 1],
                texel_offset: 0,
                texel_count: 2,
            },
            GpuMipLevel {
                resolution: [1, 1],
                texel_offset: 2,
                texel_count: 1,
            },
        ]
        .into_boxed_slice(),
        color_encoding: GpuColorEncoding::Linear,
    }];
    draft.data.float_textures = vec![GpuFloatTexture::Image {
        image: ImageId(0),
        mapping: TextureMappingId(0),
        scale: 1.0,
        invert: false,
        swrap: GpuImageWrapMode::Repeat,
        twrap: GpuImageWrapMode::Repeat,
        filter: GpuImageFilter::Bilinear,
        channel: GpuFloatImageChannel::Channel0,
    }];
    if let GpuGeometry::TriangleMesh(mesh) = &mut draft.data.geometry[0] {
        mesh.uvs = Some(vec![
            pbrt_r4::gpu::ir::GpuPoint2([0.0, 0.0]),
            pbrt_r4::gpu::ir::GpuPoint2([1.0, 0.0]),
            pbrt_r4::gpu::ir::GpuPoint2([0.0, 1.0]),
        ]);
    }
    let GpuMaterial::Diffuse(material) = &mut draft.data.materials[0];
    material.displacement = Some(FloatTextureId(0));
    GpuCompiledScene::new(draft.finish().unwrap(), GpuSourceMap::default())
}

fn instance_scene() -> GpuCompiledScene {
    let mut draft = minimal_scene_draft();
    draft
        .data
        .transforms
        .push(GpuTransform::Static(GpuStaticTransform {
            render_from_object: GpuMatrix4x4([
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            object_from_render: GpuMatrix4x4([
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            swaps_handedness: false,
        }));
    draft.data.instance_definitions = vec![GpuInstanceDefinition {
        primitives: vec![PrimitiveId(0)],
        instances: Vec::new(),
        local_bounds: GpuBounds3 {
            min: GpuPoint3([0.0, 0.0, 0.0]),
            max: GpuPoint3([1.0, 1.0, 0.0]),
        },
    }];
    draft.data.instances = vec![GpuInstance {
        definition: InstanceDefinitionId(0),
        transform: TransformId(3),
    }];
    draft.data.world_primitives = Vec::new().into_boxed_slice();
    draft.data.world_instances = vec![InstanceId(0)].into_boxed_slice();
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
fn scene_plan_flattens_static_instances_with_composed_transform() {
    let plan = ScenePlan::from_scene(instance_scene().view()).unwrap();
    assert_eq!(plan.primitives.len(), 1);
    assert_eq!(plan.tlas_instances.len(), 1);
    assert_eq!(plan.transforms[0].render_from_object[0][3], 1.0);
    assert_eq!(plan.tlas_instances[0].transform[3], 1.0);
}

#[test]
fn scene_plan_rejects_static_instance_cycles() {
    let mut draft = minimal_scene_draft();
    draft.data.instance_definitions = vec![GpuInstanceDefinition {
        primitives: Vec::new(),
        instances: vec![InstanceId(0)],
        local_bounds: GpuBounds3 {
            min: GpuPoint3([0.0, 0.0, 0.0]),
            max: GpuPoint3([1.0, 1.0, 1.0]),
        },
    }];
    draft.data.instances = vec![GpuInstance {
        definition: InstanceDefinitionId(0),
        transform: TransformId(0),
    }];
    draft.data.world_primitives = Vec::new().into_boxed_slice();
    draft.data.world_instances = vec![InstanceId(0)].into_boxed_slice();
    let scene = GpuCompiledScene::new(draft.finish().unwrap(), GpuSourceMap::default());

    assert_eq!(
        ScenePlan::from_scene(scene.view()),
        Err(pbrt_r4::gpu::webgpu::BackendError::Plan(
            PlanError::InstanceCycle { instance: 0 }
        ))
    );
}

#[test]
fn image_texture_lowering_preserves_channels_mips_and_encoding() {
    let plan = ScenePlan::from_scene(image_scene().view()).unwrap();
    assert_eq!(plan.images.len(), 3);
    assert_eq!(plan.images[0].channels, GpuImageChannels::Rgba);
    assert_eq!(plan.images[0].mip_levels.len(), 1);
    assert_eq!(plan.images[0].texels.len(), 4);
    assert!(plan.images[0].texels[0] > 0.99);
    assert!(plan.images[0].texels[1] > 0.2 && plan.images[0].texels[1] < 0.3);
    assert!((plan.images[1].texels[0] - 0.5).abs() < 1.0e-6);
    assert!((plan.images[2].texels[0] - 0.25).abs() < 1.0e-6);
    assert_eq!(plan.float_textures.len(), 3);
    assert_eq!(plan.spectrum_textures.len(), 1);
    assert!(texture_bytes(&plan).len() >= 8 * 4);
    assert!(matches!(
        plan.materials[0].reflectance,
        MaterialReflectancePlan::SpectrumTexture(0)
    ));
}

#[test]
fn gpu_buffer_serialization_matches_wgsl_layout() {
    let mut plan = ScenePlan::from_scene(minimal_scene().view()).unwrap();
    plan.transforms[0].render_from_object = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ];
    let vertices = vertex_bytes(&plan);
    let indices = index_bytes(&plan);
    let primitives = primitive_bytes(&plan);
    let materials = material_bytes(&plan);
    let transforms = transform_bytes(&plan);
    let lights = light_bytes(&plan);

    assert_eq!(vertices.len(), 3 * 64);
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
    assert_eq!(primitives.len(), 32);
    assert_eq!(materials.len(), 32);
    assert_eq!(&materials[0..4], &0.5f32.to_le_bytes());
    assert_eq!(&materials[12..16], &1.0f32.to_le_bytes());
    assert_eq!(transforms.len(), 128);
    let matrix_values = transforms[..64]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(
        matrix_values,
        [1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0,]
    );
    assert_eq!(lights.len(), 48);
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
fn software_renderer_evaluates_uniform_infinite_light() {
    let mut renderer = Renderer::new(&PrepareOptions {
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    })
    .unwrap();
    let executable = renderer.prepare(&uniform_infinite_scene()).unwrap();
    let output = renderer
        .render(
            &executable,
            &GpuRenderRequest::new(&GpuRenderConfig::default(), 0, 1).unwrap(),
        )
        .unwrap();
    assert!((output.rgb[0][0] - 0.25).abs() < 1.0e-4);
    assert!((output.rgb[0][1] - 0.25).abs() < 1.0e-4);
    assert!((output.rgb[0][2] - 0.25).abs() < 1.0e-4);
}

#[test]
fn software_renderer_rejects_zero_alpha_intersections() {
    let mut renderer = Renderer::new(&PrepareOptions {
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    })
    .unwrap();
    let executable = renderer.prepare(&alpha_scene(0.0)).unwrap();
    let output = renderer
        .render(
            &executable,
            &GpuRenderRequest::new(&GpuRenderConfig::default(), 0, 1).unwrap(),
        )
        .unwrap();
    assert_eq!(output.rgb[0], [0.0, 0.0, 0.0]);
}

#[test]
fn software_renderer_evaluates_normal_map() {
    let mut renderer = Renderer::new(&PrepareOptions {
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    })
    .unwrap();
    let executable = renderer.prepare(&normal_map_scene()).unwrap();
    let output = renderer
        .render(
            &executable,
            &GpuRenderRequest::new(&GpuRenderConfig::default(), 0, 1).unwrap(),
        )
        .unwrap();
    assert!(output.rgb[0].iter().all(|value| value.abs() < 1.0e-5));
}

#[test]
fn software_renderer_evaluates_bump_map() {
    let mut renderer = Renderer::new(&PrepareOptions {
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    })
    .unwrap();
    let base = renderer
        .prepare(&minimal_scene())
        .and_then(|executable| {
            renderer.render(
                &executable,
                &GpuRenderRequest::new(&GpuRenderConfig::default(), 0, 1).unwrap(),
            )
        })
        .unwrap();
    let bumped = renderer
        .prepare(&bump_map_scene())
        .and_then(|executable| {
            renderer.render(
                &executable,
                &GpuRenderRequest::new(&GpuRenderConfig::default(), 0, 1).unwrap(),
            )
        })
        .unwrap();
    assert!(
        bumped.rgb[0]
            .iter()
            .zip(base.rgb[0].iter())
            .any(|(bumped, base)| (bumped - base).abs() > 1.0e-5),
        "bump map did not change the shading result: base={:?}, bumped={:?}",
        base.rgb[0],
        bumped.rgb[0]
    );
}

#[test]
fn hardware_and_software_bump_map_results_match() {
    let Some(mut hardware) = renderer_or_skip() else {
        return;
    };
    let mut software = Renderer::new(&PrepareOptions {
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    })
    .unwrap();
    let scene = bump_map_scene();
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
fn hardware_and_software_normal_map_results_match() {
    let Some(mut hardware) = renderer_or_skip() else {
        return;
    };
    let mut software = Renderer::new(&PrepareOptions {
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    })
    .unwrap();
    let scene = normal_map_scene();
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
fn scene_plan_rejects_unimplemented_shadow_alpha() {
    let mut draft = minimal_scene_draft();
    draft.data.float_textures = vec![GpuFloatTexture::Constant { value: 0.0 }];
    draft.data.primitives[0].shadow_alpha = Some(FloatTextureId(0));
    let scene = GpuCompiledScene::new(draft.finish().unwrap(), GpuSourceMap::default());
    assert_eq!(
        ScenePlan::from_scene(scene.view()),
        Err(pbrt_r4::gpu::webgpu::BackendError::Plan(
            PlanError::UnsupportedAlpha { primitive: 0 }
        ))
    );
}

#[test]
fn software_renderer_evaluates_spectrum_image_texture() {
    let mut renderer = Renderer::new(&PrepareOptions {
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    })
    .unwrap();
    let executable = renderer.prepare(&image_scene()).unwrap();
    let output = renderer
        .render(
            &executable,
            &GpuRenderRequest::new(&GpuRenderConfig::default(), 0, 1).unwrap(),
        )
        .unwrap();
    assert!(output.rgb[0][0] > output.rgb[0][1]);
    assert!(output.rgb[0][1] > output.rgb[0][2]);
}

#[test]
fn software_renderer_selects_mipmap_level_for_point_and_bilinear() {
    let mut renderer = Renderer::new(&PrepareOptions {
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    })
    .unwrap();
    for filter in [GpuImageFilter::Point, GpuImageFilter::Bilinear] {
        let executable = renderer.prepare(&mipmap_lod_scene(filter)).unwrap();
        let output = renderer
            .render(
                &executable,
                &GpuRenderRequest::new(&GpuRenderConfig::default(), 0, 1).unwrap(),
            )
            .unwrap();
        assert!(
            output.rgb[0][1] > output.rgb[0][0],
            "selected the base mip level for {filter:?}: {:?}",
            output.rgb[0]
        );
        assert!(output.rgb[0][1] > output.rgb[0][2]);
    }
}

#[test]
fn hardware_and_software_mipmap_lod_results_match() {
    let Some(mut hardware) = renderer_or_skip() else {
        return;
    };
    let mut software = Renderer::new(&PrepareOptions {
        acceleration_mode: AccelerationMode::SoftwareBvh,
        ..Default::default()
    })
    .unwrap();
    let scene = mipmap_lod_scene(GpuImageFilter::Bilinear);
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
