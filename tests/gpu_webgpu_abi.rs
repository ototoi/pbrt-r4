use pbrt_r4::gpu::webgpu::abi::{
    inverse_transpose_linear, row_major_to_columns, scene_uniform, AreaLight, CameraUniform,
    Geometry, Instance, LightRecord, Material, PixelSampleState, PointLight, QueueState,
    RayWorkItem, SceneUniform, ShadowRayWorkItem, SurfaceWorkItem, Vertex, ViewportUniform,
    INVALID_INDEX, LIGHT_SAMPLER_KIND_UNIFORM,
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
fn scene_uniform_records_absolute_word_offsets() {
    let uniform = scene_uniform(2, 3, 1, 2, 8, 20, 28, 52).unwrap();

    assert_eq!(uniform.material_offset_words, 0);
    assert_eq!(uniform.material_count, 2);
    assert_eq!(uniform.light_record_offset_words, 8);
    assert_eq!(uniform.light_count, 3);
    assert_eq!(uniform.point_light_offset_words, 20);
    assert_eq!(uniform.point_light_count, 1);
    assert_eq!(uniform.area_light_offset_words, 28);
    assert_eq!(uniform.area_light_count, 2);
    assert_eq!(uniform.light_sampler_kind, LIGHT_SAMPLER_KIND_UNIFORM);
    assert_eq!(uniform.light_sampler_data_offset, INVALID_INDEX);
    assert_eq!(uniform.light_bvh_node_offset, INVALID_INDEX);
    assert_eq!(uniform.light_bvh_node_count, 0);
    assert_eq!(uniform.light_leaf_offset, INVALID_INDEX);
    assert_eq!(uniform.light_leaf_count, 0);
    assert_eq!(uniform.scene_data_words, 52);
}

#[test]
fn webgpu_storage_struct_sizes_match_shader_layout() {
    assert_eq!(std::mem::size_of::<CameraUniform>(), 128);
    assert_eq!(std::mem::size_of::<ViewportUniform>(), 32);
    assert_eq!(std::mem::size_of::<SceneUniform>(), 64);
    assert_eq!(std::mem::size_of::<Vertex>(), 64);
    assert_eq!(std::mem::size_of::<Geometry>(), 16);
    assert_eq!(std::mem::size_of::<Instance>(), 144);
    assert_eq!(std::mem::size_of::<Material>(), 16);
    assert_eq!(std::mem::size_of::<RayWorkItem>(), 144);
    assert_eq!(std::mem::size_of::<ShadowRayWorkItem>(), 80);
    assert_eq!(std::mem::size_of::<SurfaceWorkItem>(), 112);
    assert_eq!(std::mem::size_of::<PointLight>(), 32);
    assert_eq!(std::mem::size_of::<LightRecord>(), 16);
    assert_eq!(std::mem::size_of::<AreaLight>(), 48);
    assert_eq!(std::mem::size_of::<QueueState>(), 16);
    assert_eq!(std::mem::size_of::<PixelSampleState>(), 32);
}

#[test]
fn webgpu_work_item_field_offsets_match_shader_layout() {
    assert_eq!(std::mem::offset_of!(ViewportUniform, seed), 16);
    assert_eq!(std::mem::offset_of!(ViewportUniform, padding), 20);
    assert_eq!(std::mem::offset_of!(SceneUniform, material_offset_words), 0);
    assert_eq!(std::mem::offset_of!(SceneUniform, material_count), 4);
    assert_eq!(
        std::mem::offset_of!(SceneUniform, light_record_offset_words),
        8
    );
    assert_eq!(std::mem::offset_of!(SceneUniform, light_count), 12);
    assert_eq!(
        std::mem::offset_of!(SceneUniform, point_light_offset_words),
        16
    );
    assert_eq!(std::mem::offset_of!(SceneUniform, point_light_count), 20);
    assert_eq!(
        std::mem::offset_of!(SceneUniform, area_light_offset_words),
        24
    );
    assert_eq!(std::mem::offset_of!(SceneUniform, area_light_count), 28);
    assert_eq!(std::mem::offset_of!(SceneUniform, light_sampler_kind), 32);
    assert_eq!(
        std::mem::offset_of!(SceneUniform, light_sampler_data_offset),
        36
    );
    assert_eq!(
        std::mem::offset_of!(SceneUniform, light_bvh_node_offset),
        40
    );
    assert_eq!(std::mem::offset_of!(SceneUniform, light_bvh_node_count), 44);
    assert_eq!(std::mem::offset_of!(SceneUniform, light_leaf_offset), 48);
    assert_eq!(std::mem::offset_of!(SceneUniform, light_leaf_count), 52);
    assert_eq!(std::mem::offset_of!(SceneUniform, scene_data_words), 56);
    assert_eq!(std::mem::offset_of!(SceneUniform, reserved), 60);
    assert_eq!(std::mem::offset_of!(Instance, first_area_light), 8);
    assert_eq!(std::mem::offset_of!(AreaLight, emission), 8);
    assert_eq!(std::mem::offset_of!(AreaLight, total_area), 24);
    assert_eq!(std::mem::offset_of!(AreaLight, primitive), 28);

    assert_eq!(std::mem::offset_of!(RayWorkItem, origin), 0);
    assert_eq!(std::mem::offset_of!(RayWorkItem, direction), 16);
    assert_eq!(std::mem::offset_of!(RayWorkItem, throughput), 32);
    assert_eq!(std::mem::offset_of!(RayWorkItem, prev_position), 48);
    assert_eq!(std::mem::offset_of!(RayWorkItem, prev_position_error), 64);
    assert_eq!(std::mem::offset_of!(RayWorkItem, prev_geometric_normal), 80);
    assert_eq!(std::mem::offset_of!(RayWorkItem, prev_shading_normal), 96);
    assert_eq!(std::mem::offset_of!(RayWorkItem, pixel_index), 112);
    assert_eq!(std::mem::offset_of!(RayWorkItem, depth), 116);
    assert_eq!(std::mem::offset_of!(RayWorkItem, inv_w_u), 120);
    assert_eq!(std::mem::offset_of!(RayWorkItem, inv_w_l), 124);
    assert_eq!(std::mem::offset_of!(RayWorkItem, prev_pdf), 128);
    assert_eq!(std::mem::offset_of!(RayWorkItem, padding), 132);

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
    assert_eq!(std::mem::offset_of!(SurfaceWorkItem, geometric_normal), 80);
    assert_eq!(std::mem::offset_of!(SurfaceWorkItem, material), 96);
    assert_eq!(std::mem::offset_of!(SurfaceWorkItem, flags), 100);
    assert_eq!(std::mem::offset_of!(SurfaceWorkItem, padding), 104);
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
