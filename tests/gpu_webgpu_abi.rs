use pbrt_r4::gpu::flat;
use pbrt_r4::gpu::webgpu::abi::{
    camera_uniform, inverse_transpose_linear, row_major_to_columns, validate_affine, AttributeRef,
    BSSRDFMaterialRecord, CameraUniform, DenseSpectrum, DispatchIndirectArgs, FilmUniform,
    Geometry, Instance, LightRecord, LightTableUniform, MaterialNode, MaterialTableUniform,
    MeasuredBsdfRecord, MeasuredTableRecord, MediumRecord, PixelSampleState, QueueCounters,
    QueueState, RayWorkItem, RenderError, ShadowRayWorkItem, SurfaceWorkItem, TextureEvalResult,
    TriangleDistributionEntry, Vertex, ViewportUniform, QUEUE_DISPATCH_SLOT_COUNT,
};
use pbrt_r4::gpu::webgpu::sampler::{SAMPLER_UNIFORM_SIZE, SAMPLER_UNIFORM_VARIANT_WORDS_OFFSET};

#[test]
fn webgpu_matrices_are_uploaded_as_column_major() {
    let matrix = row_major_to_columns([
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    ]);
    assert_eq!(matrix[0], [0.0, 4.0, 8.0, 12.0]);
    assert_eq!(matrix[3], [3.0, 7.0, 11.0, 15.0]);
}

#[test]
fn webgpu_affine_validation_accepts_f32_roundoff_but_rejects_projective_rows() {
    let mut matrix = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -8.0e-11, 2.2e-10, -2.3e-10,
        1.0,
    ];
    validate_affine(matrix, "test").unwrap();

    matrix[12] = 1.0e-3;
    assert!(validate_affine(matrix, "test").is_err());
}

#[test]
fn webgpu_storage_struct_sizes_match_shader_layout() {
    assert_eq!(std::mem::size_of::<CameraUniform>(), 256);
    assert_eq!(std::mem::size_of::<ViewportUniform>(), 60);
    assert_eq!(std::mem::size_of::<MaterialTableUniform>(), 72);
    assert_eq!(std::mem::size_of::<LightTableUniform>(), 48);
    assert_eq!(std::mem::size_of::<Vertex>(), 64);
    assert_eq!(std::mem::size_of::<Geometry>(), 32);
    assert_eq!(std::mem::offset_of!(Geometry, intersection_normal_kind), 16);
    assert_eq!(std::mem::size_of::<Instance>(), 160);
    assert_eq!(std::mem::size_of::<MediumRecord>(), 96);
    assert_eq!(std::mem::size_of::<MaterialNode>(), 48);
    assert_eq!(std::mem::offset_of!(MaterialNode, bssrdf_index), 32);
    assert_eq!(std::mem::size_of::<BSSRDFMaterialRecord>(), 16);
    assert_eq!(std::mem::offset_of!(BSSRDFMaterialRecord, table_index), 8);
    assert_eq!(std::mem::size_of::<AttributeRef>(), 8);
    assert_eq!(std::mem::size_of::<MeasuredBsdfRecord>(), 32);
    assert_eq!(std::mem::size_of::<MeasuredTableRecord>(), 80);
    assert_eq!(std::mem::size_of::<DenseSpectrum>(), 1888);
    assert_eq!(std::mem::size_of::<RayWorkItem>(), 176);
    assert_eq!(std::mem::offset_of!(RayWorkItem, medium_id), 164);
    assert_eq!(std::mem::offset_of!(RayWorkItem, medium_segment_index), 168);
    assert_eq!(std::mem::size_of::<ShadowRayWorkItem>(), 176);
    assert_eq!(std::mem::size_of::<SurfaceWorkItem>(), 256);
    assert_eq!(std::mem::size_of::<TextureEvalResult>(), 32);
    assert_eq!(std::mem::size_of::<LightRecord>(), 16);
    assert_eq!(
        std::mem::size_of::<pbrt_r4::gpu::webgpu::abi::LightSamplingModel>(),
        80
    );
    assert_eq!(std::mem::size_of::<QueueState>(), 16);
    assert_eq!(std::mem::size_of::<QueueCounters>(), 304);
    assert_eq!(
        std::mem::offset_of!(QueueCounters, medium_continuation),
        224
    );
    assert_eq!(
        std::mem::offset_of!(QueueCounters, shadow_continuation),
        240
    );
    assert_eq!(std::mem::offset_of!(QueueCounters, medium_active), 256);
    assert_eq!(std::mem::offset_of!(QueueCounters, shadow_active), 272);
    assert_eq!(std::mem::offset_of!(QueueCounters, medium_scatter), 288);
    assert_eq!(std::mem::offset_of!(ShadowRayWorkItem, endpoint), 32);
    assert_eq!(std::mem::offset_of!(ShadowRayWorkItem, transmittance), 112);
    assert_eq!(std::mem::offset_of!(ShadowRayWorkItem, segment_index), 164);
    assert_eq!(std::mem::size_of::<DispatchIndirectArgs>(), 12);
    assert_eq!(std::mem::offset_of!(DispatchIndirectArgs, x), 0);
    assert_eq!(std::mem::offset_of!(DispatchIndirectArgs, y), 4);
    assert_eq!(std::mem::offset_of!(DispatchIndirectArgs, z), 8);
    assert_eq!(QUEUE_DISPATCH_SLOT_COUNT, 15);
    assert_eq!(std::mem::size_of::<RenderError>(), 16);
    assert_eq!(std::mem::offset_of!(RenderError, value), 0);
    assert_eq!(std::mem::offset_of!(RenderError, padding), 4);
    assert_eq!(std::mem::size_of::<PixelSampleState>(), 80);
    assert_eq!(std::mem::size_of::<FilmUniform>(), 32);
    assert_eq!(std::mem::size_of::<TriangleDistributionEntry>(), 16);
    assert_eq!(SAMPLER_UNIFORM_SIZE, 64);
    assert_eq!(SAMPLER_UNIFORM_VARIANT_WORDS_OFFSET, 32);
}

#[test]
fn webgpu_storage_array_strides_are_16_byte_aligned() {
    for size in [
        std::mem::size_of::<Vertex>(),
        std::mem::size_of::<Geometry>(),
        std::mem::size_of::<Instance>(),
        std::mem::size_of::<MediumRecord>(),
        std::mem::size_of::<RayWorkItem>(),
        std::mem::size_of::<ShadowRayWorkItem>(),
        std::mem::size_of::<SurfaceWorkItem>(),
        std::mem::size_of::<PixelSampleState>(),
        std::mem::size_of::<DenseSpectrum>(),
        std::mem::size_of::<MeasuredBsdfRecord>(),
        std::mem::size_of::<MeasuredTableRecord>(),
        std::mem::size_of::<pbrt_r4::gpu::webgpu::abi::LightSamplingModel>(),
    ] {
        assert_eq!(size % 16, 0, "storage stride {size} is not 16-byte aligned");
    }
}

#[test]
fn webgpu_work_item_padding_uses_scalar_shader_fields() {
    let types = include_str!("../src/gpu/webgpu/shaders/types.wgsl");
    assert!(!types.contains("_padding: vec3<u32>"));
    assert!(!types.contains("_padding0: vec3<u32>"));
    assert!(!types.contains("_padding1: vec3<u32>"));
    assert!(!types.contains("_padding: array<u32, 3>"));
    assert!(!types.contains("_padding0: array<u32, 3>"));
    assert!(!types.contains("_padding1: array<u32, 3>"));
    assert!(types.contains("_padding2: u32"));
    assert!(types.contains("segment_index: u32"));
}

#[test]
fn webgpu_normal_matrix_is_inverse_transpose_of_linear_transform() {
    let normal = inverse_transpose_linear(
        [
            2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        "test",
    )
    .unwrap();
    assert_eq!(normal[0][0], 0.5);
    assert_eq!(normal[1][1], 0.25);
    assert_eq!(normal[2][2], 0.125);
}

#[test]
fn perspective_camera_upload_includes_finite_minimum_ray_differentials() {
    let camera = flat::Camera {
        camera_to_world: [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        kind: "perspective".to_string(),
        fov: 45.0,
        lens_radius: 0.0,
        focal_distance: 1.0e6,
        disable_texture_filtering: false,
        disable_pixel_jitter: false,
        screen_window: [-1.0, 1.0, -1.0, 1.0],
        medium: u32::MAX,
    };
    let viewport = flat::Viewport {
        resolution: [64, 32],
        region_offset: [0, 0],
        region_resolution: [64, 32],
    };

    let uniform = camera_uniform(&camera, &viewport).unwrap();

    for differential in [
        uniform.min_dir_differential_x,
        uniform.min_dir_differential_y,
    ] {
        assert!(differential[..3].iter().all(|value| value.is_finite()));
        assert!(differential[..3].iter().any(|value| *value != 0.0));
    }
    assert_eq!(uniform.world_to_camera[0], [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(uniform.world_to_camera[1], [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(uniform.world_to_camera[2], [0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn webgpu_rust_struct_sizes_match_wgsl_types() {
    use pbrt_r4::gpu::webgpu::abi;
    use wgpu::naga;

    // Buffers are allocated with Rust struct sizes while shaders index them
    // with WGSL strides, so any mismatch silently drops out-of-range items.
    let source = include_str!("../src/gpu/webgpu/shaders/types.wgsl");
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    let wgsl_size = |name: &str| {
        module
            .types
            .iter()
            .find_map(|(_, ty)| match ty.inner {
                naga::TypeInner::Struct { span, .. } if ty.name.as_deref() == Some(name) => {
                    Some(span as usize)
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("WGSL struct {name} is missing from types.wgsl"))
    };
    macro_rules! rust_sizes {
        ($($name:ident),* $(,)?) => {
            [$((stringify!($name), std::mem::size_of::<abi::$name>())),*]
        };
    }
    let rust_sizes = rust_sizes![
        CameraUniform,
        ViewportUniform,
        MaterialTableUniform,
        LightTableUniform,
        Vertex,
        Geometry,
        TextureNodeRecord,
        Instance,
        MediumRecord,
        MaterialNode,
        AttributeRef,
        MeasuredBsdfRecord,
        MeasuredTableRecord,
        TextureRootRecord,
        DenseSpectrum,
        RayWorkItem,
        ShadowRayWorkItem,
        SurfaceWorkItem,
        AttributesEvalWorkItem,
        MaterialRoot,
        TextureEvalResult,
        LightRecord,
        LightSamplingModel,
        PortalImageInfiniteRecord,
        PortalDistributionTexel,
        DirectLightSample,
        TriangleDistributionEntry,
        QueueState,
        FilmUniform,
        QueueCounters,
        DispatchIndirectArgs,
        RenderError,
        PixelSampleState,
    ];
    for (name, rust_size) in rust_sizes {
        assert_eq!(
            rust_size,
            wgsl_size(name),
            "{name} size differs between Rust and WGSL"
        );
    }
}
