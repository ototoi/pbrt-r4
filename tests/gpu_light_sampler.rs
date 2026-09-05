use pbrt_r4::gpu::ir::flat::{LightKind, LightRecord, RenderSettings};
use pbrt_r4::gpu::webgpu::light_sampler::{
    resolve_light_sampler, resolve_scene_light_sampler, CompactLightBounds, LightSamplerKind,
    LIGHT_BVH_INDEX_MAX,
};

fn settings(light_sampler: &str) -> RenderSettings {
    RenderSettings {
        samples_per_pixel: 1,
        max_depth: 1,
        seed: 0,
        light_sampler: light_sampler.to_string(),
    }
}

fn lights(count: usize) -> Vec<LightRecord> {
    (0..count)
        .map(|payload| LightRecord {
            kind: LightKind::Point,
            payload: payload as u32,
        })
        .collect()
}

#[test]
fn resolves_supported_light_sampler_names() {
    assert_eq!(
        resolve_light_sampler("uniform", 2).unwrap(),
        LightSamplerKind::Uniform
    );
    assert_eq!(
        resolve_light_sampler("bvh", 2).unwrap(),
        LightSamplerKind::Bvh
    );
}

#[test]
fn one_registered_light_forces_uniform() {
    for requested in ["uniform", "bvh", "power", "exhaustive", "unknown"] {
        assert_eq!(
            resolve_light_sampler(requested, 1).unwrap(),
            LightSamplerKind::Uniform
        );
    }
}

#[test]
fn unknown_name_reports_and_uses_bvh() {
    assert_eq!(
        resolve_light_sampler("unknown", 2).unwrap(),
        LightSamplerKind::Bvh
    );
}

#[test]
fn scene_resolution_uses_render_settings_and_registered_light_count() {
    assert_eq!(
        resolve_scene_light_sampler(&settings("bvh"), &lights(1)).unwrap(),
        LightSamplerKind::Uniform
    );
    assert_eq!(
        resolve_scene_light_sampler(&settings("uniform"), &lights(2)).unwrap(),
        LightSamplerKind::Uniform
    );
    assert!(resolve_scene_light_sampler(&settings("power"), &lights(2)).is_err());
}

#[test]
fn unimplemented_v4_samplers_are_rejected() {
    for requested in ["power", "exhaustive"] {
        let error = resolve_light_sampler(requested, 2).unwrap_err();
        assert!(format!("{error:?}").contains("not implemented"));
    }
}

#[test]
fn compact_light_bounds_use_conservative_quantization_and_packed_tags() {
    let bounds = CompactLightBounds::pack(
        [-1.0, 0.0, 2.0],
        [1.0, 2.0, 4.0],
        [-1.0, 0.0, 2.0],
        [1.0, 2.0, 4.0],
        [0.0, 0.0, 1.0],
        3.0,
        -1.0,
        0.0,
        false,
    )
    .unwrap();
    let words = bounds.to_words(7, true).unwrap();

    assert_eq!(words[0] & 0xffff, 0);
    assert_eq!((words[2] >> 16) & 0xffff, u32::from(u16::MAX));
    assert_eq!(words[4], 3.0f32.to_bits());
    assert_eq!((words[6] >> 31) & 1, 1);
    assert_eq!(words[6] & 0x7fff_ffff, 7);
    assert!(bounds.to_words(LIGHT_BVH_INDEX_MAX + 1, false).is_err());
}

#[test]
fn compact_light_bounds_handle_degenerate_global_axis() {
    let bounds = CompactLightBounds::pack(
        [1.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
        [0.0, 1.0, 0.0],
        1.0,
        0.0,
        0.0,
        true,
    )
    .unwrap();
    assert_eq!(bounds.q_min[0], 0);
    assert_eq!(bounds.q_max[0], 0);
}
