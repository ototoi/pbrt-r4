use pbrt_r4::gpu::webgpu::scene::validate_scene_data_size;

#[test]
fn scene_data_size_must_fit_both_device_limits() {
    assert_eq!(validate_scene_data_size(16, 128, 64).unwrap(), 64);

    let buffer_error = validate_scene_data_size(17, 64, 128).unwrap_err();
    assert!(format!("{buffer_error:?}").contains("max_buffer_size"));

    let binding_error = validate_scene_data_size(17, 128, 64).unwrap_err();
    assert!(format!("{binding_error:?}").contains("max_storage_buffer_binding_size"));
}
