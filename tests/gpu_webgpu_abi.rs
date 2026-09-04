use pbrt_r4::gpu::webgpu::abi::{
    inverse_transpose_linear, row_major_to_columns, AreaLight, CameraUniform, Geometry, Instance,
    LightRecord, Material, PixelSampleState, PointLight, QueueState, RayWorkItem,
    ShadowRayWorkItem, SurfaceWorkItem, Vertex, ViewportUniform,
};

#[test]
fn webgpu_matrices_are_uploaded_as_column_major() {
    let matrix = row_major_to_columns([
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    ]);
    assert_eq!(matrix[0], [0.0, 4.0, 8.0, 12.0]);
    assert_eq!(matrix[1], [1.0, 5.0, 9.0, 13.0]);
    assert_eq!(matrix[2], [2.0, 6.0, 10.0, 14.0]);
    assert_eq!(matrix[3], [3.0, 7.0, 11.0, 15.0]);
}

#[test]
fn webgpu_storage_struct_sizes_match_shader_layout() {
    assert_eq!(std::mem::size_of::<CameraUniform>(), 128);
    assert_eq!(std::mem::size_of::<ViewportUniform>(), 32);
    assert_eq!(std::mem::size_of::<Vertex>(), 64);
    assert_eq!(std::mem::size_of::<Geometry>(), 16);
    assert_eq!(std::mem::size_of::<Instance>(), 144);
    assert_eq!(std::mem::size_of::<Material>(), 16);
    assert_eq!(std::mem::size_of::<RayWorkItem>(), 80);
    assert_eq!(std::mem::size_of::<ShadowRayWorkItem>(), 80);
    assert_eq!(std::mem::size_of::<SurfaceWorkItem>(), 96);
    assert_eq!(std::mem::size_of::<PointLight>(), 32);
    assert_eq!(std::mem::size_of::<LightRecord>(), 16);
    assert_eq!(std::mem::size_of::<AreaLight>(), 48);
    assert_eq!(std::mem::size_of::<QueueState>(), 16);
    assert_eq!(std::mem::size_of::<PixelSampleState>(), 32);
}

#[test]
fn webgpu_work_item_field_offsets_match_shader_layout() {
    assert_eq!(std::mem::offset_of!(RayWorkItem, origin), 0);
    assert_eq!(std::mem::offset_of!(RayWorkItem, direction), 16);
    assert_eq!(std::mem::offset_of!(RayWorkItem, throughput), 32);
    assert_eq!(std::mem::offset_of!(RayWorkItem, pixel_index), 48);
    assert_eq!(std::mem::offset_of!(RayWorkItem, depth), 52);
    assert_eq!(std::mem::offset_of!(RayWorkItem, inv_w_u), 56);
    assert_eq!(std::mem::offset_of!(RayWorkItem, inv_w_l), 60);
    assert_eq!(std::mem::offset_of!(RayWorkItem, prev_pdf), 64);
    assert_eq!(std::mem::offset_of!(RayWorkItem, padding), 68);

    assert_eq!(std::mem::offset_of!(ShadowRayWorkItem, origin), 0);
    assert_eq!(std::mem::offset_of!(ShadowRayWorkItem, direction), 16);
    assert_eq!(std::mem::offset_of!(ShadowRayWorkItem, max_t), 32);
    assert_eq!(std::mem::offset_of!(ShadowRayWorkItem, padding), 36);
    assert_eq!(std::mem::offset_of!(ShadowRayWorkItem, direct), 48);
    assert_eq!(std::mem::offset_of!(ShadowRayWorkItem, pixel_index), 64);
    assert_eq!(std::mem::offset_of!(ShadowRayWorkItem, reserved), 68);

    assert_eq!(std::mem::offset_of!(SurfaceWorkItem, t), 0);
    assert_eq!(std::mem::offset_of!(SurfaceWorkItem, hit), 4);
    assert_eq!(
        std::mem::offset_of!(SurfaceWorkItem, instance_custom_data),
        8
    );
    assert_eq!(std::mem::offset_of!(SurfaceWorkItem, primitive_index), 12);
    assert_eq!(std::mem::offset_of!(SurfaceWorkItem, barycentric), 16);
    assert_eq!(std::mem::offset_of!(SurfaceWorkItem, position), 32);
    assert_eq!(std::mem::offset_of!(SurfaceWorkItem, position_error), 48);
    assert_eq!(std::mem::offset_of!(SurfaceWorkItem, normal), 64);
    assert_eq!(std::mem::offset_of!(SurfaceWorkItem, material), 80);
    assert_eq!(std::mem::offset_of!(SurfaceWorkItem, flags), 84);
    assert_eq!(std::mem::offset_of!(SurfaceWorkItem, padding), 88);
}

#[test]
fn webgpu_storage_array_strides_are_16_byte_aligned() {
    for size in [
        std::mem::size_of::<Vertex>(),
        std::mem::size_of::<Geometry>(),
        std::mem::size_of::<Instance>(),
        std::mem::size_of::<RayWorkItem>(),
        std::mem::size_of::<ShadowRayWorkItem>(),
        std::mem::size_of::<SurfaceWorkItem>(),
        std::mem::size_of::<PixelSampleState>(),
    ] {
        assert_eq!(size % 16, 0, "storage stride {size} is not 16-byte aligned");
    }
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
    assert_eq!(normal[3], [0.0, 0.0, 0.0, 1.0]);
}
