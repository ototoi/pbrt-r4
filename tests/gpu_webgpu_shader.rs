use pbrt_r4::gpu::webgpu::shader::compose_source;

const INTERSECT_SHADOW_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/intersect_shadow.wgsl");

#[test]
fn shadow_direction_is_loaded_from_its_vec4_aligned_queue_slot() {
    let source = compose_source(INTERSECT_SHADOW_SHADER);

    assert!(source.contains("const SHADOW_DIRECTION_WORD: u32 = 4u;"));
    assert!(source.contains("load_shadow_direction(ray_index)"));
    assert!(!source.contains("load_shadow_vec3(ray_index, 3u)"));
}

#[test]
fn escaped_queue_follows_the_classification_queues() {
    let source = compose_source(INTERSECT_SHADOW_SHADER);
    let escaped_offset = source
        .split("fn escaped_data_offset() -> u32 {")
        .nth(1)
        .and_then(|tail| tail.split('}').next())
        .expect("escaped_data_offset must be present in the composed shader");

    assert!(escaped_offset.contains("shadow_data_offset()"));
    assert!(escaped_offset.contains("pixel_count() * SHADOW_WORDS"));
    assert!(escaped_offset.contains("classification_capacity() * 2u"));
    assert!(!escaped_offset.contains("RAY_WORDS"));
}
