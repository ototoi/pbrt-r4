#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::webgpu::queue::TypedQueueSizes;

#[test]
fn typed_queue_sizes_follow_the_host_abi() {
    let sizes = TypedQueueSizes::new(3, 3, 2).unwrap();

    assert_eq!(sizes.surfaces, 3 * 256);
    assert_eq!(sizes.pixel_sample_states, 3 * 96);
    assert_eq!(sizes.current_rays, 3 * 176);
    assert_eq!(sizes.next_rays, 3 * 176);
    assert_eq!(sizes.shadow_rays, 3 * 176);
    assert_eq!(sizes.active_medium_indices, 3 * 4);
    assert_eq!(sizes.next_medium_indices, 3 * 4);
    assert_eq!(sizes.medium_scatter_indices, 3 * 4);
    assert_eq!(sizes.active_shadow_indices, 3 * 4);
    assert_eq!(sizes.next_shadow_indices, 3 * 4);
    assert_eq!(sizes.material_ray_indices, 3 * 4);
    assert_eq!(sizes.attributes_eval_work_items, 3 * 3 * 192);
    assert_eq!(sizes.texture_eval_results, 3 * 2 * 32);
    assert_eq!(sizes.hit_area_ray_indices, 3 * 4);
    assert_eq!(sizes.escaped_ray_indices, 3 * 4);
    assert_eq!(sizes.light_sample_work_items, 3 * 3 * 16);
}

#[test]
fn typed_queue_sizes_must_fit_shader_u32_indices() {
    let max_pixels = u64::from(u32::MAX / 3);
    let sizes = TypedQueueSizes::new(max_pixels, 1, 1).unwrap();
    assert_eq!(sizes.light_sample_work_items, max_pixels * 3 * 16);
    assert!(TypedQueueSizes::new(max_pixels + 1, 1, 1).is_err());
}

#[test]
fn attributes_eval_stride_must_not_be_zero() {
    assert!(TypedQueueSizes::new(1, 0, 1).is_err());
    assert!(TypedQueueSizes::new(1, 1, 0).is_err());
}

#[test]
fn attributes_eval_work_items_scale_with_the_per_surface_stride() {
    let sizes = TypedQueueSizes::new(8, 7, 5).unwrap();
    assert_eq!(sizes.attributes_eval_work_items, 8 * 7 * 192);
    assert_eq!(sizes.current_rays, 8 * 176);
}
