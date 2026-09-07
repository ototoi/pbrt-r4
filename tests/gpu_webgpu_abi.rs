use pbrt_r4::gpu::webgpu::abi::{
    inverse_transpose_linear, row_major_to_columns, AttributeRef, CameraUniform, DenseSpectrum,
    FilmUniform, Geometry, Instance, LightRecord, LightTableUniform, MaterialRecord,
    MaterialTableUniform, PixelSampleState, QueueCounters, QueueState, RayWorkItem, RenderError,
    ScatteringModelRecord, ScatteringNodeRecord, ShadowRayWorkItem, SurfaceWorkItem,
    TriangleDistributionEntry, Vertex, ViewportUniform,
};

#[test]
fn webgpu_matrices_are_uploaded_as_column_major() {
    let matrix = row_major_to_columns([
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    ]);
    assert_eq!(matrix[0], [0.0, 4.0, 8.0, 12.0]);
    assert_eq!(matrix[3], [3.0, 7.0, 11.0, 15.0]);
}

#[test]
fn webgpu_storage_struct_sizes_match_shader_layout() {
    assert_eq!(std::mem::size_of::<CameraUniform>(), 128);
    assert_eq!(std::mem::size_of::<ViewportUniform>(), 32);
    assert_eq!(std::mem::size_of::<MaterialTableUniform>(), 72);
    assert_eq!(std::mem::size_of::<LightTableUniform>(), 48);
    assert_eq!(std::mem::size_of::<Vertex>(), 64);
    assert_eq!(std::mem::size_of::<Geometry>(), 16);
    assert_eq!(std::mem::size_of::<Instance>(), 144);
    assert_eq!(std::mem::size_of::<MaterialRecord>(), 16);
    assert_eq!(std::mem::size_of::<ScatteringModelRecord>(), 16);
    assert_eq!(std::mem::size_of::<ScatteringNodeRecord>(), 32);
    assert_eq!(std::mem::size_of::<AttributeRef>(), 8);
    assert_eq!(std::mem::size_of::<DenseSpectrum>(), 1888);
    assert_eq!(std::mem::size_of::<RayWorkItem>(), 144);
    assert_eq!(std::mem::size_of::<ShadowRayWorkItem>(), 80);
    assert_eq!(std::mem::size_of::<SurfaceWorkItem>(), 112);
    assert_eq!(std::mem::size_of::<LightRecord>(), 16);
    assert_eq!(
        std::mem::size_of::<pbrt_r4::gpu::webgpu::abi::LightSamplingModel>(),
        32
    );
    assert_eq!(std::mem::size_of::<QueueState>(), 16);
    assert_eq!(std::mem::size_of::<QueueCounters>(), 96);
    assert_eq!(std::mem::size_of::<RenderError>(), 16);
    assert_eq!(std::mem::offset_of!(RenderError, value), 0);
    assert_eq!(std::mem::offset_of!(RenderError, padding), 4);
    assert_eq!(std::mem::size_of::<PixelSampleState>(), 80);
    assert_eq!(std::mem::size_of::<FilmUniform>(), 32);
    assert_eq!(std::mem::size_of::<TriangleDistributionEntry>(), 16);
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
        std::mem::size_of::<DenseSpectrum>(),
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
}
