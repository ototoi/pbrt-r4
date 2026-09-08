use pbrt_r4::gpu::webgpu::shader::{compose_source, required_limits_for_sources};
use pbrt_r4::gpu::webgpu::stages::canonical_wavefront_bindings;

const INTERSECT_SHADOW_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/intersect_shadow.wgsl");
const EVALUATE_MATERIALS_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/evaluate_materials.wgsl");
const GENERATE_PRIMARY_RAYS_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/generate_primary_rays.wgsl");
const ESCAPED_TEST_SHADER: &str =
    r#"@compute @workgroup_size(1) fn test_stage() { append_escaped_ray(0u); }"#;
const HANDLE_EMISSIVE_SHADER: &str = include_str!("../src/gpu/webgpu/shaders/handle_emissive.wgsl");
const SAMPLE_DIFFUSE_BOUNCE_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/sample_diffuse_bounce.wgsl");
const SAMPLE_DIELECTRIC_BOUNCE_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/sample_dielectric_bounce.wgsl");
const SAMPLE_THIN_DIELECTRIC_BOUNCE_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/sample_thin_dielectric_bounce.wgsl");
const SAMPLE_CONDUCTOR_BOUNCE_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/sample_conductor_bounce.wgsl");
const SAMPLE_COMPOSITE_BOUNCE_SHADER: &str =
    include_str!("../src/gpu/webgpu/shaders/sample_composite_bounce.wgsl");
const SHADE_SURFACE_SHADER: &str = include_str!("../src/gpu/webgpu/shaders/shade_surface.wgsl");
const COMMON_SHADER: &str = concat!(
    include_str!("../src/gpu/webgpu/shaders/types.wgsl"),
    include_str!("../src/gpu/webgpu/shaders/wavefront.wgsl")
);
const RESOURCES_SHADER: &str = include_str!("../src/gpu/webgpu/shaders/resources.wgsl");
const SPECTRUM_TEST_SHADER: &str = r#"
    @compute @workgroup_size(1) fn test_stage() {
        let value = evaluate_spectrum(0u, vec4<f32>(360.0, 400.5, 700.0, 830.0));
        let divided = safe_div_spectrum(value, vec4<f32>(1.0, 0.0, 2.0, 4.0));
        if (spectrum_is_constant(0u) && average_spectrum(divided) > max_spectrum(divided)) {
            set_render_error();
        }
    }
"#;

#[test]
fn dense_spectrum_module_declares_one_structured_table() {
    let source = compose_source(SPECTRUM_TEST_SHADER);
    assert!(source.contains("var<storage, read> spectrum_attributes: array<DenseSpectrum>;"));
    assert!(source.contains("struct DenseSpectrum"));
    assert!(source.contains("fn safe_div_spectrum"));
    assert!(!source.contains("var<storage, read> materials: array<MaterialRecord>;"));
}

#[test]
fn required_limits_are_derived_from_each_composed_stage() {
    let bindings = canonical_wavefront_bindings();
    let limits = required_limits_for_sources(&bindings, &[GENERATE_PRIMARY_RAYS_SHADER]).unwrap();

    assert_eq!(limits.storage_buffers_per_shader_stage, 8);
    assert_eq!(limits.uniform_buffers_per_shader_stage, 2);
    assert_eq!(limits.bind_groups, 1);
}

#[test]
fn required_limits_reject_unregistered_group_zero_bindings() {
    let bindings = canonical_wavefront_bindings();
    let source = r#"
        @group(0) @binding(99) var<storage, read> unknown_resource: array<u32>;
        @compute @workgroup_size(1) fn test_stage() {
            let value = unknown_resource[0];
        }
    "#;

    let error = required_limits_for_sources(&bindings, &[source]).unwrap_err();
    assert!(error
        .to_string()
        .contains("unregistered group 0 binding 99"));
}

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
    assert!(RESOURCES_SHADER.contains("@group(0) @binding(19)"));
    assert!(RESOURCES_SHADER.contains("var<uniform> material_table: MaterialTableUniform;"));
    assert!(RESOURCES_SHADER.contains("var<uniform> light_table: LightTableUniform;"));
    assert!(COMMON_SHADER.contains("struct MaterialRecord {"));
    assert!(COMMON_SHADER.contains("tree_size: u32"));
    assert!(RESOURCES_SHADER.contains("@group(0) @binding(21)"));
    assert!(RESOURCES_SHADER.contains("var<storage, read> materials: array<MaterialRecord>;"));
    assert!(COMMON_SHADER.contains("struct AttributeRef {"));
    assert!(RESOURCES_SHADER.contains("var<storage, read> attribute_refs: array<AttributeRef>;"));
    assert!(RESOURCES_SHADER.contains("var<storage, read> scalar_attributes: array<f32>;"));
    assert!(!COMMON_SHADER.contains("scene_data"));
    assert!(RESOURCES_SHADER.contains("var<storage, read> light_records: array<LightRecord>;"));
    assert!(!COMMON_SHADER.contains("material_light_data"));
}

#[test]
fn composed_stage_contains_only_referenced_resources() {
    let source = compose_source(GENERATE_PRIMARY_RAYS_SHADER);
    assert!(source.contains("var<uniform> camera: CameraUniform;"));
    assert!(source.contains("var<uniform> viewport: ViewportUniform;"));
    assert!(source.contains("var<storage, read_write> queue_counters: QueueCounters;"));
    assert!(source.contains("var<storage, read_write> current_rays: array<RayWorkItem>;"));
    assert!(
        source.contains("var<storage, read_write> pixel_sample_states: array<PixelSampleState>;")
    );
    assert!(!source.contains("var<storage, read> light_records: array<LightRecord>;"));
    assert!(!source.contains("var<storage, read> materials: array<MaterialRecord>;"));
}

#[test]
fn shadow_direction_is_loaded_from_the_typed_shadow_queue() {
    let source = compose_source(INTERSECT_SHADOW_SHADER);

    assert!(source.contains("load_shadow_direction(ray_index)"));
    assert!(source.contains("return shadow_rays[index].direction.xyz;"));
    assert!(!source.contains("bitcast<f32>(atomicLoad"));
}

#[test]
fn escaped_queue_is_a_typed_ray_index_queue() {
    let source = compose_source(ESCAPED_TEST_SHADER);
    assert!(source.contains("escaped_ray_indices[index] = ray_index;"));
    assert!(source.contains("queue_counters.escaped.capacity"));
    assert!(!source.contains("escaped_data_offset"));
}

#[test]
fn classification_queues_resolve_current_rays_in_constant_time() {
    let evaluate = compose_source(EVALUATE_MATERIALS_SHADER);
    assert!(evaluate.contains("let ray_index = load_material_eval_ray(queue_index);"));
    assert!(evaluate.contains("let ray = load_current_ray(ray_index);"));
    assert!(!evaluate.contains("find_current_ray_for_pixel"));

    let emissive = compose_source(HANDLE_EMISSIVE_SHADER);
    assert!(emissive.contains("let ray_index = load_hit_area_ray(queue_index);"));
    assert!(!emissive.contains("find_current_ray_for_pixel"));
}

#[test]
fn shadow_queue_carries_the_complete_rgb_contribution() {
    let evaluate = compose_source(EVALUATE_MATERIALS_SHADER);
    assert!(evaluate.contains("ray.throughput * direct"));

    let shadow = compose_source(INTERSECT_SHADOW_SHADER);
    assert!(shadow.contains("load_sample_radiance(pixel_index) + shadow_direct"));
    assert!(!shadow.contains("load_current_ray"));
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
    assert!(COMMON_SHADER
        .contains("fn load_diffuse_reflectance(material_index: u32, lambda: vec4<f32>)"));
    assert!(EVALUATE_MATERIALS_SHADER.contains("reflectance = load_diffuse_reflectance"));
    assert!(EVALUATE_MATERIALS_SHADER.contains("reflectance / PI"));
    assert!(SAMPLE_DIFFUSE_BOUNCE_SHADER.contains("ray.throughput * reflectance"));
}

#[test]
fn dielectric_shader_uses_eta_for_reflection_and_transmission() {
    assert!(COMMON_SHADER.contains("const MATERIAL_KIND_DIELECTRIC: u32 = 3u;"));
    assert!(
        COMMON_SHADER.contains("fn load_dielectric_eta(material_index: u32, lambda: vec4<f32>)")
    );
    assert!(SAMPLE_DIELECTRIC_BOUNCE_SHADER.contains("load_dielectric_eta"));
    assert!(SAMPLE_DIELECTRIC_BOUNCE_SHADER.contains("fresnel"));
    assert!(SAMPLE_DIELECTRIC_BOUNCE_SHADER.contains("refract(-wo, normal, eta_ratio)"));
    assert!(SAMPLE_DIELECTRIC_BOUNCE_SHADER.contains("reflect(-wo, normal)"));
}

#[test]
fn conductor_shader_uses_complex_fresnel_attributes() {
    let source = compose_source(SAMPLE_CONDUCTOR_BOUNCE_SHADER);
    assert!(source.contains("MATERIAL_KIND_CONDUCTOR"));
    assert!(source.contains("evaluated.values[0]"));
    assert!(source.contains("evaluated.values[1]"));
    assert!(source.contains("conductor_fresnel"));
    assert!(source.contains("fn sample_conductor_bounce"));
}

#[test]
fn composite_shader_uses_evaluated_material_tree() {
    let source = compose_source(SAMPLE_COMPOSITE_BOUNCE_SHADER);
    assert!(source.contains("MATERIAL_KIND_COATED_DIFFUSE"));
    assert!(source.contains("load_evaluated_attributes"));
    assert!(source.contains("queue_counters.next"));
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
