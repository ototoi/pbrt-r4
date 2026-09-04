use pbrt_r4::gpu::webgpu::shader::compose_source;

const INTERSECT_SHADOW_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/intersect_shadow.wgsl");
const EVALUATE_MATERIALS_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/evaluate_materials.wgsl");
const GENERATE_PRIMARY_RAYS_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/generate_primary_rays.wgsl");
const SAMPLE_DIFFUSE_BOUNCE_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/sample_diffuse_bounce.wgsl");

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

#[test]
fn wavefront_stages_use_persisted_sample_dimensions() {
    let evaluate = compose_source(EVALUATE_MATERIALS_SHADER);
    assert!(evaluate.contains("let samples = load_ray_samples(pixel_index);"));
    assert!(evaluate.contains("sample_uniform_light(samples.direct.x)"));
    assert!(evaluate.contains("let selector = samples.direct.y;"));
    assert!(evaluate.contains("let su = sqrt(samples.direct.z);"));
    assert!(evaluate.contains("let bv = samples.direct.w;"));
    assert!(!EVALUATE_MATERIALS_SHADER.contains("random01("));

    let bounce = compose_source(SAMPLE_DIFFUSE_BOUNCE_SHADER);
    assert!(bounce.contains("let u = vec2<f32>(samples.indirect.z, samples.indirect.w);"));
    assert!(bounce.contains("if (samples.indirect.y < q)"));
    assert!(bounce.contains("generate_ray_samples(pixel_index, ray.depth + 1u)"));
    assert!(!SAMPLE_DIFFUSE_BOUNCE_SHADER.contains("random01("));
}

#[test]
fn primary_rays_initialize_depth_zero_sample_state() {
    assert!(GENERATE_PRIMARY_RAYS_SHADER
        .contains("store_ray_samples(pixel_index, generate_ray_samples(pixel_index, 0u));"));
}
