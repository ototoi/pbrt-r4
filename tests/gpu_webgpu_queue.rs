use pbrt_r4::gpu::webgpu::queue::TypedQueueSizes;

#[test]
fn typed_queue_sizes_follow_the_host_abi() {
    let sizes = TypedQueueSizes::new(3).unwrap();

    assert_eq!(sizes.surfaces, 3 * 112);
    assert_eq!(sizes.pixel_sample_states, 3 * 48);
    assert_eq!(sizes.current_rays, 3 * 144);
    assert_eq!(sizes.next_rays, 3 * 144);
    assert_eq!(sizes.shadow_rays, 3 * 80);
    assert_eq!(sizes.material_ray_indices, 3 * 4);
    assert_eq!(sizes.hit_area_ray_indices, 3 * 4);
    assert_eq!(sizes.escaped_ray_indices, 3 * 4);
}

#[test]
fn typed_queue_sizes_must_fit_shader_u32_indices() {
    assert!(TypedQueueSizes::new(u64::from(u32::MAX) + 1).is_err());
}
