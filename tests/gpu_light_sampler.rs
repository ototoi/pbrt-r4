#![cfg(feature = "webgpu")]

use image::{ImageBuffer, Rgb};
use pbrt_r4::base::light::Light as CpuLight;
use pbrt_r4::gpu::flat::{flatten_node, Scene as FlatScene};
use pbrt_r4::gpu::flat::{
    IntegratorKind, Light, LightKind, RenderSettings, SamplerKind, SamplerRandomization,
};
use pbrt_r4::gpu::node::{prepare_triangle_meshes, tessellate_shapes};
use pbrt_r4::gpu::webgpu::light_sampler::{
    resolve_light_sampler, resolve_scene_light_sampler, LightSamplerKind,
};
use pbrt_r4::parser::parse_string;
use pbrt_r4::parser::scene_builder::SceneBuilder;
use pbrt_r4::scene::Scene as CpuScene;
use pbrt_r4::util::spectrum::{safe_div, SampledWavelengths};
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
    let cpu = builder.build_scene().unwrap();
    let lambda = SampledWavelengths::sample_visible(0.5);
    let cpu_weights = cpu
        .lights
        .iter()
        .map(|light| safe_div(light.phi(&lambda), lambda.pdf_spectrum()).average() as f32)
        .collect::<Vec<_>>();
    assert_eq!(cpu_weights.len(), 4);
    assert_close(weights[0], cpu_weights[0] + cpu_weights[1]);
    assert_close(weights[1], cpu_weights[2] + cpu_weights[3]);
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
    let cpu = builder.build_scene().unwrap();
    assert_power_weights(&flat, &cpu);
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 2e-5 * expected.abs().max(1e-8),
        "actual={actual}, expected={expected}"
    );
}

fn flatten_source(source: &str) -> (FlatScene, CpuScene) {
    let mut builder = SceneBuilder::new();
    parse_string(source, &mut builder).unwrap();
    let cpu = builder.build_scene().unwrap();
    let root = builder.build_gpu_ir_node().unwrap();
    {
        let mut node = root.write().unwrap();
        tessellate_shapes(&mut node).unwrap();
        prepare_triangle_meshes(&mut node).unwrap();
    }
    (flatten_node(root).unwrap(), cpu)
}

fn assert_power_weights(flat: &FlatScene, cpu: &CpuScene) {
    let lambda = SampledWavelengths::sample_visible(0.5);
    let expected = cpu
        .lights
        .iter()
        .filter(|light| light.bounds().is_some())
        .chain(cpu.lights.iter().filter(|light| light.bounds().is_none()))
        .map(|light| safe_div(light.phi(&lambda), lambda.pdf_spectrum()).average() as f32)
        .collect::<Vec<_>>();
    let actual = flat.lights.power_weights.as_ref().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_close(*actual, expected);
    }
}

const PREFIX: &str = r#"
Integrator "path" "string lightsampler" ["power"]
Sampler "independent" "integer pixelsamples" [1]
Film "rgb" "integer xresolution" [1] "integer yresolution" [1]
Camera "perspective"
WorldBegin
"#;
const TRIANGLE: &str =
    r#"Shape "trianglemesh" "point3 P" [0 0 1 1 0 1 0 1 1] "integer indices" [0 1 2]"#;

#[test]
fn power_scene_preserves_gpu_integrator_name_handling() {
    for integrator in ["path", "volpath", "bdpt", "ao", "unknown"] {
        let source = format!("{PREFIX} LightSource \"point\" LightSource \"distant\" {TRIANGLE}")
            .replace(
                "Integrator \"path\"",
                &format!("Integrator \"{integrator}\""),
            );
        let (flat, cpu) = flatten_source(&source);
        assert_eq!(
            flat.render_settings.integrator_kind,
            if integrator == "path" {
                IntegratorKind::Path
            } else {
                IntegratorKind::VolPath
            }
        );
        assert_power_weights(&flat, &cpu);
    }
}

#[test]
fn power_scene_excludes_zero_alpha_concrete_area_from_nonarea_weights() {
    for alpha in [0, 1] {
        let source = format!(
            r#"{PREFIX}
LightSource "point" "rgb I" [3 2 1]
AreaLightSource "diffuse" "rgb L" [1 2 3]
{TRIANGLE} "float alpha" [{alpha}]
"#
        );
        let (flat, cpu) = flatten_source(&source);
        assert_eq!(flat.lights.lights.len(), 2);
        assert!(cpu
            .lights
            .iter()
            .any(|light| matches!(light.as_ref(), CpuLight::DiffuseArea(_))));
        assert_power_weights(&flat, &cpu);
    }
}

#[test]
fn power_weights_match_cpu_phi_for_all_supported_light_kinds_and_spectral_inputs() {
    let directory = tempfile::tempdir().unwrap();
    let image_path = directory.path().join("environment.png");
    ImageBuffer::<Rgb<u8>, _>::from_raw(2, 2, vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        .unwrap()
        .save(&image_path)
        .unwrap();
    let source = format!(
        r#"{PREFIX}
LightSource "infinite" "rgb L" [0.3 0.7 0.2]
LightSource "point" "rgb I" [3 2 1]
LightSource "distant" "rgb L" [0.1 0.2 0.3]
AttributeBegin
Rotate 31 1 1 0
LightSource "spot" "spectrum I" [400 1 500 2 600 0.5 700 0.1] "point from" [1 2 3] "point to" [0 0 0]
AttributeEnd
LightSource "infinite" "string filename" ["{image}"] "float scale" [2]
LightSource "infinite" "string filename" ["{image}"] "point portal" [0 0 0 1 0 0 1 1 0 0 1 0]
AreaLightSource "diffuse" "spectrum L" [400 1 500 0.5 600 2 700 0.1] "bool twosided" [true]
{TRIANGLE}
"#,
        image = image_path.display()
    );
    let (flat, cpu) = flatten_source(&source);
    assert_eq!(flat.lights.lights.len(), 3);
    assert_eq!(flat.lights.infinite_lights.len(), 4);
    assert_power_weights(&flat, &cpu);
    for (actual, light) in flat
        .lights
        .bounds
        .iter()
        .zip(cpu.lights.iter().filter(|l| l.bounds().is_some()))
    {
        assert_bounds_match_cpu(*actual, &light.bounds().unwrap());
    }
}

fn assert_bounds_match_cpu(
    actual: pbrt_r4::gpu::flat::LightBounds,
    expected: &pbrt_r4::base::light::LightBounds,
) {
    for (actual, expected) in actual.bounds.min.into_iter().zip([
        expected.bounds.min.x,
        expected.bounds.min.y,
        expected.bounds.min.z,
    ]) {
        assert!((actual - expected as f32).abs() < 2e-5);
    }
    for (actual, expected) in actual.bounds.max.into_iter().zip([
        expected.bounds.max.x,
        expected.bounds.max.y,
        expected.bounds.max.z,
    ]) {
        assert!((actual - expected as f32).abs() < 2e-5);
    }
    for (actual, expected) in
        actual
            .direction
            .into_iter()
            .zip([expected.w.x, expected.w.y, expected.w.z])
    {
        assert!((actual - expected as f32).abs() < 2e-5);
    }
    assert_close(actual.phi, expected.phi as f32);
    assert!((actual.cos_theta_o - expected.cos_theta_o as f32).abs() < 2e-5);
    assert!((actual.cos_theta_e - expected.cos_theta_e as f32).abs() < 2e-5);
    assert_eq!(actual.two_sided, expected.two_sided);
}

#[test]
fn area_bounds_match_cpu_normals_under_orientation_and_affine_transforms() {
    for transform in [
        "",
        "Translate 1 2 3",
        "Rotate 37 1 1 0",
        "Scale 3 0.5 2",
        "Scale -2 1 3",
    ] {
        for reverse in ["", "ReverseOrientation"] {
            for normals in [
                "",
                r#""normal N" [0 0 1 0 0 1 0 0 1]"#,
                r#""normal N" [1 0 0.1 1 0 0.1 0 0 -0.15]"#,
            ] {
                let source = format!(
                    r#"{PREFIX}
LightSource "point"
{transform}
{reverse}
AreaLightSource "diffuse"
{TRIANGLE} {normals}
"#
                );
                let (flat, cpu) = flatten_source(&source);
                let expected = cpu
                    .lights
                    .iter()
                    .find(|l| matches!(l.as_ref(), CpuLight::DiffuseArea(_)))
                    .unwrap()
                    .bounds()
                    .unwrap();
                assert_bounds_match_cpu(flat.lights.bounds[0], &expected);
                let center = expected.centroid();
                let p = [
                    center.x as f32 + flat.lights.bounds[0].direction[0] * 10.0,
                    center.y as f32 + flat.lights.bounds[0].direction[1] * 10.0,
                    center.z as f32 + flat.lights.bounds[0].direction[2] * 10.0,
                ];
                assert!(flat.lights.bounds[0].importance(p, [0.0; 3]).unwrap() > 0.0);
            }
        }
    }
}
