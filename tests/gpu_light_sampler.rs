#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::flat::flatten_node;
use pbrt_r4::gpu::flat::{
    IntegratorKind, Light, LightKind, RenderSettings, SamplerKind, SamplerRandomization,
};
use pbrt_r4::gpu::node::{prepare_triangle_meshes, tessellate_shapes};
use pbrt_r4::gpu::webgpu::light_sampler::{
    resolve_light_sampler, resolve_scene_light_sampler, LightSamplerKind,
};
use pbrt_r4::parser::parse_string;
use pbrt_r4::parser::scene_builder::SceneBuilder;
use std::sync::Arc;

fn settings(light_sampler: &str) -> RenderSettings {
    RenderSettings {
        integrator_kind: IntegratorKind::Path,
        sampler_kind: SamplerKind::Independent,
        randomization: SamplerRandomization::None,
        samples_per_pixel: 1,
        x_samples: 1,
        y_samples: 1,
        jitter: false,
        max_depth: 1,
        seed: 0,
        light_sampler: light_sampler.to_string(),
        disable_wavelength_jitter: false,
    }
}

fn lights(count: usize) -> Vec<Light> {
    (0..count)
        .map(|payload| Light {
            kind: LightKind::Point,
            attributes: Vec::new(),
            sampling_model: payload as u32,
            image_index: u32::MAX,
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
    assert_eq!(
        resolve_light_sampler("exhaustive", 2).unwrap(),
        LightSamplerKind::Exhaustive
    );
    assert_eq!(
        resolve_light_sampler("power", 2).unwrap(),
        LightSamplerKind::Power
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
    assert_eq!(
        resolve_scene_light_sampler(&settings("exhaustive"), &lights(2)).unwrap(),
        LightSamplerKind::Exhaustive
    );
    assert_eq!(
        resolve_scene_light_sampler(&settings("power"), &lights(2)).unwrap(),
        LightSamplerKind::Power
    );
}

#[test]
fn power_scene_uses_concrete_cpu_phi_weights_in_flat_handle_order() {
    let scene_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/scenes/gpu-wavefront-multiple-area-diffuse.pbrt");
    let source = std::fs::read_to_string(scene_path).unwrap().replace(
        "Integrator \"volpath\"",
        "Integrator \"volpath\" \"string lightsampler\" [ \"power\" ]",
    );
    let mut builder = SceneBuilder::new();
    parse_string(&source, &mut builder).unwrap();
    let root = builder.build_gpu_ir_node().unwrap();
    {
        let mut root = root.write().unwrap();
        tessellate_shapes(&mut root).unwrap();
        prepare_triangle_meshes(&mut root).unwrap();
    }
    let flat = flatten_node(Arc::clone(&root)).unwrap();
    let weights = flat.lights.power_weights.as_ref().unwrap();
    assert_eq!(
        weights.len(),
        flat.lights.lights.len() + flat.lights.infinite_lights.len()
    );
    assert_eq!(weights.len(), 2);
    assert!(weights
        .iter()
        .all(|weight| weight.is_finite() && *weight > 0.0));
}

#[test]
fn power_scene_orders_finite_and_infinite_weights_as_uploaded_handles() {
    let scene_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/scenes/gpu-spot-distant-smoke.pbrt");
    let source = std::fs::read_to_string(scene_path).unwrap().replace(
        "Integrator \"path\"",
        "Integrator \"path\" \"string lightsampler\" [ \"power\" ]",
    );
    let mut builder = SceneBuilder::new();
    parse_string(&source, &mut builder).unwrap();
    let root = builder.build_gpu_ir_node().unwrap();
    {
        let mut root = root.write().unwrap();
        tessellate_shapes(&mut root).unwrap();
        prepare_triangle_meshes(&mut root).unwrap();
    }
    let flat = flatten_node(Arc::clone(&root)).unwrap();
    let weights = flat.lights.power_weights.as_ref().unwrap();
    assert_eq!(flat.lights.lights.len(), 1);
    assert_eq!(flat.lights.infinite_lights.len(), 1);
    assert_eq!(weights.len(), 2);
    assert!(weights
        .iter()
        .all(|weight| weight.is_finite() && *weight > 0.0));
}
