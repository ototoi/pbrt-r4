use pbrt_r4::gpu::webgpu::shader::compose_source;

const INTERSECT_SHADOW_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/intersect_shadow.wgsl");
const EVALUATE_MATERIALS_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/evaluate_materials.wgsl");
const GENERATE_PRIMARY_RAYS_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/generate_primary_rays.wgsl");
const HANDLE_EMISSIVE_SHADER: &str = include_str!("../src/gpu/webgpu/shaders/handle_emissive.wgsl");
const SAMPLE_DIFFUSE_BOUNCE_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/sample_diffuse_bounce.wgsl");
const SHADE_SURFACE_SHADER: &str = include_str!("../src/gpu/webgpu/shaders/shade_surface.wgsl");
const COMMON_SHADER: &str = include_str!("../src/gpu/webgpu/shaders/common.wgsl");

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
    assert!(evaluate.contains("sample_triangle_for_context("));
    assert!(evaluate.contains("vec2<f32>(samples.direct.y, samples.direct.z)"));
    assert!(evaluate.contains("sample_spherical_triangle("));
    assert!(evaluate.contains("MIN_SPHERICAL_SAMPLE_AREA: f32 = 3e-4"));
    assert!(evaluate.contains("MAX_SPHERICAL_SAMPLE_AREA: f32 = 6.22"));
    assert!(evaluate.contains("max(ray.inv_w_u, 1e-7) * sampled_light_pdf"));
    assert!(!EVALUATE_MATERIALS_SHADER.contains("random01("));

    let bounce = compose_source(SAMPLE_DIFFUSE_BOUNCE_SHADER);
    assert!(bounce.contains("let u = vec2<f32>(samples.indirect.y, samples.indirect.z);"));
    assert!(bounce.contains("if (samples.indirect.w < q)"));
    assert!(bounce.contains("generate_ray_samples(pixel_index, ray.depth + 1u)"));
    assert!(!SAMPLE_DIFFUSE_BOUNCE_SHADER.contains("random01("));
}

#[test]
fn emissive_hit_resolves_the_triangle_light_handle() {
    assert!(HANDLE_EMISSIVE_SHADER.contains("instance.first_area_light + surface.primitive_index"));
    assert!(HANDLE_EMISSIVE_SHADER.contains("load_light_payload(light_handle)"));
    assert!(HANDLE_EMISSIVE_SHADER.contains("uniform_light_pmf_for_handle(light_handle)"));
    assert!(!COMMON_SHADER.contains("fn light_pmf_for_area"));
}

#[test]
fn primary_rays_initialize_depth_zero_sample_state() {
    assert!(GENERATE_PRIMARY_RAYS_SHADER
        .contains("store_ray_samples(pixel_index, generate_ray_samples(pixel_index, 0u));"));
    assert!(GENERATE_PRIMARY_RAYS_SHADER.contains("vec4<f32>(0.0),\n        pixel_index,"));
    assert!(
        SAMPLE_DIFFUSE_BOUNCE_SHADER.contains("surface.position,\n        vec4<f32>(normal, 0.0),")
    );
}

#[test]
fn emissive_mis_uses_the_unoffset_previous_interaction_context() {
    assert!(HANDLE_EMISSIVE_SHADER.contains("ray.prev_position.xyz"));
    assert!(HANDLE_EMISSIVE_SHADER.contains("ray.prev_shading_normal.xyz"));
    assert!(!HANDLE_EMISSIVE_SHADER.contains("ray.origin.xyz - surface.position.xyz"));
}

#[test]
fn triangle_hit_position_is_reconstructed_from_barycentrics() {
    assert!(SHADE_SURFACE_SHADER.contains("let position = p0 * b0 + p1 * b1 + p2 * b2;"));
    assert!(!SHADE_SURFACE_SHADER
        .contains("let position = ray.origin.xyz + ray.direction.xyz * surface.t;"));
}

#[test]
fn random_samples_are_independent_across_pixel_sample_and_depth() {
    let random01 = COMMON_SHADER
        .split("fn random01(")
        .nth(1)
        .and_then(|tail| tail.split("fn generate_ray_samples").next())
        .expect("random01 must be defined before generate_ray_samples");

    assert!(random01.contains("pixel_index"));
    assert!(random01.contains("viewport.sample_index"));
    assert!(random01.contains("dimension + depth * 8u"));
}
