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
const SAMPLE_DIELECTRIC_BOUNCE_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/sample_dielectric_bounce.wgsl");
const SAMPLE_LAYERED_BOUNCE_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/sample_layered_bounce.wgsl");
const SAMPLE_THIN_DIELECTRIC_BOUNCE_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/sample_thin_dielectric_bounce.wgsl");
const SHADE_SURFACE_SHADER: &str = include_str!("../src/gpu/webgpu/shaders/shade_surface.wgsl");
const COMMON_SHADER: &str = include_str!("../src/gpu/webgpu/shaders/common.wgsl");

#[test]
fn immutable_scene_metadata_is_separate_from_viewport_state() {
    let viewport = COMMON_SHADER
        .split("struct ViewportUniform {")
        .nth(1)
        .and_then(|tail| tail.split("};").next())
        .unwrap();
    assert!(!viewport.contains("light_count"));
    assert!(COMMON_SHADER.contains("struct MaterialTableUniform {"));
    assert!(COMMON_SHADER.contains("struct LightTableUniform {"));
    assert!(COMMON_SHADER.contains("@group(0) @binding(11)"));
    assert!(COMMON_SHADER.contains("var<uniform> material_table: MaterialTableUniform;"));
    assert!(COMMON_SHADER.contains("var<uniform> light_table: LightTableUniform;"));
    assert!(COMMON_SHADER.contains("struct MaterialRecord {"));
    assert!(COMMON_SHADER.contains("@group(0) @binding(13)"));
    assert!(COMMON_SHADER.contains("var<storage, read> materials: array<MaterialRecord>;"));
    assert!(COMMON_SHADER.contains("struct MaterialAttributeRef {"));
    assert!(COMMON_SHADER
        .contains("var<storage, read> material_attributes: array<MaterialAttributeRef>;"));
    assert!(COMMON_SHADER.contains("var<storage, read> scalar_attributes: array<f32>;"));
    assert!(!COMMON_SHADER.contains("scene_data"));
    assert!(COMMON_SHADER.contains("var<storage, read> light_records: array<LightRecord>;"));
    assert!(!COMMON_SHADER.contains("material_light_data"));
}

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
    assert!(evaluate.contains(
        "sample_scene_light(samples.direct.x, surface.position.xyz, surface.normal.xyz)"
    ));
    assert!(evaluate.contains("sample_light_bvh(selector, p, n)"));
    assert!(evaluate.contains("cos_sub_clamped("));
    let emissive = compose_source(HANDLE_EMISSIVE_SHADER);
    assert!(emissive.contains("light_pmf_for_handle("));
    assert!(evaluate.contains("select_area_triangle(light_payload, samples.direct.y)"));
    assert!(evaluate.contains("vec2<f32>(samples.direct.z, samples.direct.w)"));
    assert!(evaluate.contains("sample_uniform_triangle_for_context("));
    assert!(!evaluate.contains("sample_spherical_triangle("));
    assert!(!evaluate.contains("MIN_SPHERICAL_SAMPLE_AREA"));
    assert!(!evaluate.contains("MAX_SPHERICAL_SAMPLE_AREA"));
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
    assert!(HANDLE_EMISSIVE_SHADER.contains("let light_handle = instance.area_light;"));
    assert!(HANDLE_EMISSIVE_SHADER.contains("triangle_selection.pmf * triangle_pdf"));
    assert!(HANDLE_EMISSIVE_SHADER.contains("load_light_payload(light_handle)"));
    assert!(HANDLE_EMISSIVE_SHADER.contains("light_pmf_for_handle(light_handle"));
    assert!(!COMMON_SHADER.contains("fn light_pmf_for_area"));
}

#[test]
fn area_light_sampling_uses_the_group_cdf_and_area_pmf() {
    let source = compose_source(EVALUATE_MATERIALS_SHADER);
    assert!(source.contains(
        "let triangle_selection = select_area_triangle(light_payload, samples.direct.y)"
    ));
    assert!(source.contains("sample_uniform_triangle_for_context"));
    assert!(source.contains("triangle.orientation_flags & 1u"));
    assert!(source.contains("triangle_selection.pmf * triangle_sample.w"));
    assert!(source.contains("load_area_distribution_count(light_payload)"));
}

#[test]
fn diffuse_shaders_load_type_specific_reflectance() {
    assert!(COMMON_SHADER.contains("fn load_diffuse_reflectance(material_index: u32)"));
    assert!(EVALUATE_MATERIALS_SHADER.contains("reflectance = load_diffuse_reflectance"));
    assert!(EVALUATE_MATERIALS_SHADER.contains("reflectance / PI"));
    assert!(SAMPLE_DIFFUSE_BOUNCE_SHADER.contains("ray.throughput * vec4<f32>(reflectance, 1.0)"));
}

#[test]
fn non_layered_stage_does_not_include_layered_module() {
    let source = compose_source(GENERATE_PRIMARY_RAYS_SHADER);
    assert!(!source.contains("pbrt-v4 bxdfs.h: LayeredParams"));
    let layered = compose_source(SAMPLE_LAYERED_BOUNCE_SHADER);
    assert!(layered.contains("pbrt-v4 bxdfs.h: LayeredBxDF"));
}

#[test]
fn layered_shader_resolves_top_and_bottom_nodes() {
    assert!(COMMON_SHADER.contains("const MATERIAL_KIND_LAYERED: u32 = 4u;"));
    assert!(COMMON_SHADER.contains("fn load_layered_bxdf(material_index: u32)"));
    assert!(COMMON_SHADER.contains("fn load_layered_bottom_reflectance"));
    assert!(COMMON_SHADER.contains("load_scattering_child(root, 0u)"));
    assert!(SAMPLE_LAYERED_BOUNCE_SHADER.contains("load_layered_bottom_reflectance"));
    assert!(SAMPLE_LAYERED_BOUNCE_SHADER.contains("layered_sample("));
}

#[test]
fn dielectric_shader_uses_eta_for_reflection_and_transmission() {
    assert!(COMMON_SHADER.contains("const MATERIAL_KIND_DIELECTRIC: u32 = 3u;"));
    assert!(COMMON_SHADER.contains("fn load_dielectric_eta(node_index: u32)"));
    assert!(SAMPLE_DIELECTRIC_BOUNCE_SHADER.contains("load_dielectric_eta"));
    assert!(SAMPLE_DIELECTRIC_BOUNCE_SHADER.contains("fresnel"));
    assert!(SAMPLE_DIELECTRIC_BOUNCE_SHADER.contains("refract(-wo, normal, eta_ratio)"));
    assert!(SAMPLE_DIELECTRIC_BOUNCE_SHADER.contains("reflect(-wo, normal)"));
}

#[test]
fn thin_dielectric_shader_uses_thin_interface_transport() {
    assert!(COMMON_SHADER.contains("const MATERIAL_KIND_THIN_DIELECTRIC: u32 = 5u;"));
    assert!(SAMPLE_THIN_DIELECTRIC_BOUNCE_SHADER.contains("MATERIAL_KIND_THIN_DIELECTRIC"));
    assert!(SAMPLE_THIN_DIELECTRIC_BOUNCE_SHADER.contains("direction = select(-wo"));
    assert!(SAMPLE_THIN_DIELECTRIC_BOUNCE_SHADER.contains("r0 + (1.0 - r0)"));
}

#[test]
fn primary_rays_initialize_depth_zero_sample_state() {
    assert!(GENERATE_PRIMARY_RAYS_SHADER
        .contains("store_ray_samples(pixel_index, generate_ray_samples(pixel_index, 0u));"));
    assert!(GENERATE_PRIMARY_RAYS_SHADER.contains("vec4<f32>(0.0),\n        pixel_index,"));
    assert!(
        SAMPLE_DIFFUSE_BOUNCE_SHADER.contains(
            "surface.position,\n        surface.position_error,\n        surface.geometric_normal,\n        vec4<f32>(normal, 0.0),"
        )
    );
}

#[test]
fn emissive_mis_uses_the_unoffset_previous_interaction_context() {
    assert!(HANDLE_EMISSIVE_SHADER.contains("ray.prev_position.xyz"));
    assert!(HANDLE_EMISSIVE_SHADER.contains("ray.prev_shading_normal.xyz"));
    assert!(HANDLE_EMISSIVE_SHADER.contains("ray.direction.xyz"));
    assert!(!HANDLE_EMISSIVE_SHADER.contains("ray.origin.xyz - surface.position.xyz"));
}

#[test]
fn bounce_rays_preserve_the_complete_previous_interaction_context() {
    assert!(SAMPLE_DIFFUSE_BOUNCE_SHADER.contains("surface.position,"));
    assert!(SAMPLE_DIFFUSE_BOUNCE_SHADER.contains("surface.position_error,"));
    assert!(SAMPLE_DIFFUSE_BOUNCE_SHADER.contains("surface.geometric_normal,"));
    assert!(SAMPLE_DIFFUSE_BOUNCE_SHADER.contains("vec4<f32>(normal, 0.0),"));
}

#[test]
fn triangle_area_fallback_uses_the_v4_folding_map() {
    let source = compose_source(EVALUATE_MATERIALS_SHADER);
    assert!(source.contains("fn sample_uniform_triangle(u: vec2<f32>)"));
    assert!(source.contains("if (u.x < u.y)"));
    assert!(!source.contains("let su = sqrt(u.x)"));
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
