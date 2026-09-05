use pbrt_r4::gpu::ir::flat::{LightKind, LightRecord, RenderSettings};
use pbrt_r4::gpu::webgpu::light_sampler::{
    resolve_light_sampler, resolve_scene_light_sampler, LightSamplerKind,
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
fn unknown_name_is_rejected() {
    assert!(resolve_light_sampler("unknown", 2).is_err());
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
