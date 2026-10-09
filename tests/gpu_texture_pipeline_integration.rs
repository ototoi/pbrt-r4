use std::process::Command;

use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::pipeline::Pipeline;
use pbrt_r4::gpu::webgpu::shader::required_limits_for_sources;
use pbrt_r4::gpu::webgpu::stage::COMPUTE_STAGES;
use pbrt_r4::gpu::webgpu::stages::canonical_wavefront_bindings;
use pbrt_r4::util::imageio::read_image::read_image;
use pbrt_r4::util::spectrum::RGBSpectrum;

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn texture_material_pipeline_compiles() {
    let _ = env_logger::builder().is_test(true).try_init();
    let canonical_bindings = canonical_wavefront_bindings();
    let stage_sources = COMPUTE_STAGES
        .iter()
        .map(|stage| stage.source)
        .collect::<Vec<_>>();
    let mut required = required_limits_for_sources(&canonical_bindings, &stage_sources).unwrap();
    required.bind_groups = required.bind_groups.max(2);
    let context = Context::new(required, 1, 1).unwrap();
    Pipeline::new(&context.device, 1, 1, 1, false).unwrap();
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn dynamic_and_constant_scale_shader_paths_match_the_cpu_reference() {
    let directory = tempfile::tempdir().unwrap();
    let dynamic_scene = directory.path().join("dynamic-scale.pbrt");
    let constant_scene = directory.path().join("constant-scale.pbrt");
    std::fs::write(&dynamic_scene, scale_scene(true)).unwrap();
    std::fs::write(&constant_scene, scale_scene(false)).unwrap();

    let dynamic_gpu = render(&dynamic_scene, directory.path(), "dynamic-gpu.exr", true);
    let constant_gpu = render(&constant_scene, directory.path(), "constant-gpu.exr", true);
    let dynamic_cpu = render(&dynamic_scene, directory.path(), "dynamic-cpu.exr", false);

    assert_eq!(dynamic_gpu.len(), constant_gpu.len());
    for (dynamic, constant) in dynamic_gpu.iter().zip(&constant_gpu) {
        for (dynamic, constant) in dynamic.to_rgb().into_iter().zip(constant.to_rgb()) {
            assert!(
                (dynamic - constant).abs() <= 1e-5,
                "dynamic and constant WGSL Scale paths differ: {dynamic} != {constant}"
            );
        }
    }

    let gpu_energy = image_energy(&dynamic_gpu);
    let cpu_energy = image_energy(&dynamic_cpu);
    assert!(gpu_energy > 0.0 && cpu_energy > 0.0);
    let relative_error = (gpu_energy - cpu_energy).abs() / cpu_energy;
    assert!(
        relative_error < 0.35,
        "GPU Scale result differs from CPU reference: gpu={gpu_energy}, cpu={cpu_energy}"
    );
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn bump_mapping_changes_the_gpu_render() {
    let directory = tempfile::tempdir().unwrap();
    let scene = directory.path().join("bump.pbrt");
    let baseline = directory.path().join("baseline.pbrt");
    std::fs::write(&scene, bump_scene(true)).unwrap();
    std::fs::write(&baseline, bump_scene(false)).unwrap();

    let bumped = render(&scene, directory.path(), "bumped.exr", true);
    let plain = render(&baseline, directory.path(), "plain.exr", true);

    let difference: f32 = bumped
        .iter()
        .zip(&plain)
        .map(|(bumped, plain)| {
            bumped
                .to_rgb()
                .into_iter()
                .zip(plain.to_rgb())
                .map(|(bumped, plain)| (bumped - plain).abs())
                .sum::<f32>()
        })
        .sum();
    assert!(difference > 0.01, "bump map did not change the GPU render");

    let constant_scene_path = directory.path().join("constant-bump.pbrt");
    std::fs::write(&constant_scene_path, constant_bump_scene()).unwrap();
    let constant = render(
        &constant_scene_path,
        directory.path(),
        "constant-bump.exr",
        true,
    );
    assert_same_image(
        &constant,
        &plain,
        "constant displacement changed the render",
    );

    for (name, scene) in [
        (
            "reversed-uv",
            bump_scene_with_options(true, true, false, false),
        ),
        (
            "reversed-orientation",
            bump_scene_with_options(true, false, true, false),
        ),
        (
            "negative-determinant",
            bump_scene_with_options(true, false, false, true),
        ),
    ] {
        let scene_path = directory.path().join(format!("{name}.pbrt"));
        std::fs::write(&scene_path, scene).unwrap();
        let pixels = render(&scene_path, directory.path(), &format!("{name}.exr"), true);
        assert!(pixels.iter().all(RGBSpectrum::is_valid));
    }
}

fn bump_scene(with_bump: bool) -> String {
    bump_scene_with_options(with_bump, false, false, false)
}

fn bump_scene_with_options(
    with_bump: bool,
    reverse_uv: bool,
    reverse_orientation: bool,
    negative_determinant: bool,
) -> String {
    let bump = if with_bump {
        r#"
Texture "height" "float" "bilerp"
    "float v00" [ 0 ] "float v01" [ 0.5 ]
    "float v10" [ 0.5 ] "float v11" [ 1 ]
Material "diffuse" "rgb reflectance" [ 0.8 0.7 0.6 ]
    "texture displacement" [ "height" ]
"#
    } else {
        "Material \"diffuse\" \"rgb reflectance\" [ 0.8 0.7 0.6 ]\n"
    };
    let uv = if reverse_uv {
        "    \"point2 uv\" [ 1 0  0 0  0 1  1 1 ]\n"
    } else {
        "    \"point2 uv\" [ 0 0  1 0  1 1  0 1 ]\n"
    };
    let orientation = if reverse_orientation {
        "ReverseOrientation\n"
    } else {
        ""
    };
    let transform = if negative_determinant {
        "Scale -1 1 1\n"
    } else {
        ""
    };
    format!(
        r#"Integrator "volpath" "integer maxdepth" [ 1 ]
Sampler "independent" "integer pixelsamples" [ 1 ]
Film "rgb" "integer xresolution" [ 8 ] "integer yresolution" [ 8 ]
LookAt 0 0 4  0 0 0  0 1 0
Camera "perspective" "float fov" [ 35 ]
WorldBegin
{bump}
LightSource "point" "point3 from" [ 0 2 4 ] "rgb I" [ 40 40 40 ]
{transform}{orientation}
Shape "trianglemesh"
    "point3 P" [ -2 -2 0  2 -2 0  2 2 0  -2 2 0 ]
{uv}    "integer indices" [ 0 1 2  0 2 3 ]
"#
    )
}

fn constant_bump_scene() -> String {
    bump_scene(false).replace(
        "Material \"diffuse\" \"rgb reflectance\" [ 0.8 0.7 0.6 ]",
        "Texture \"height\" \"float\" \"constant\" \"float value\" [ 0.4 ]\nMaterial \"diffuse\" \"rgb reflectance\" [ 0.8 0.7 0.6 ] \"texture displacement\" [ \"height\" ]",
    )
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn mix_bump_mapping_uses_only_the_selected_child() {
    let directory = tempfile::tempdir().unwrap();
    let plain_scene = directory.path().join("plain.pbrt");
    let bumped_scene = directory.path().join("bumped.pbrt");
    let mix_plain_scene = directory.path().join("mix-plain.pbrt");
    let mix_bumped_scene = directory.path().join("mix-bumped.pbrt");
    std::fs::write(&plain_scene, bump_scene(false)).unwrap();
    std::fs::write(&bumped_scene, bump_scene(true)).unwrap();
    std::fs::write(&mix_plain_scene, mix_bump_scene(0.0)).unwrap();
    std::fs::write(&mix_bumped_scene, mix_bump_scene(1.0)).unwrap();

    let plain = render(&plain_scene, directory.path(), "plain.exr", true);
    let bumped = render(&bumped_scene, directory.path(), "bumped.exr", true);
    let mix_plain = render(&mix_plain_scene, directory.path(), "mix-plain.exr", true);
    let mix_bumped = render(&mix_bumped_scene, directory.path(), "mix-bumped.exr", true);

    assert_same_image(
        &mix_plain,
        &plain,
        "Mix selected the bumped child at amount zero",
    );
    assert_same_image(
        &mix_bumped,
        &bumped,
        "Mix did not apply the bumped child at amount one",
    );
}

fn mix_bump_scene(amount: f32) -> String {
    let material_definitions = format!(
        r#"Texture "height" "float" "bilerp"
    "float v00" [ 0 ] "float v01" [ 0.5 ]
    "float v10" [ 0.5 ] "float v11" [ 1 ]
MakeNamedMaterial "bumped" "string type" [ "diffuse" ]
    "rgb reflectance" [ 0.8 0.7 0.6 ] "texture displacement" [ "height" ]
MakeNamedMaterial "plain" "string type" [ "diffuse" ] "rgb reflectance" [ 0.8 0.7 0.6 ]
MakeNamedMaterial "mixed" "string type" [ "mix" ] "float amount" [ {amount} ]
    "string namedmaterial1" [ "bumped" ] "string namedmaterial2" [ "plain" ]
NamedMaterial "mixed"
"#
    );
    base_triangle_scene(&material_definitions)
}

fn base_triangle_scene(material: &str) -> String {
    format!(
        r#"Integrator "volpath" "integer maxdepth" [ 1 ]
Sampler "independent" "integer pixelsamples" [ 1 ]
Film "rgb" "integer xresolution" [ 8 ] "integer yresolution" [ 8 ]
LookAt 0 0 4  0 0 0  0 1 0
Camera "perspective" "float fov" [ 35 ]
WorldBegin
{material}
LightSource "point" "point3 from" [ 0 2 4 ] "rgb I" [ 40 40 40 ]
Shape "trianglemesh"
    "point3 P" [ -2 -2 0  2 -2 0  2 2 0  -2 2 0 ]
    "point2 uv" [ 0 0  1 0  1 1  0 1 ]
    "integer indices" [ 0 1 2  0 2 3 ]
"#
    )
}

fn assert_same_image(actual: &[RGBSpectrum], expected: &[RGBSpectrum], message: &str) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        for (actual, expected) in actual.to_rgb().into_iter().zip(expected.to_rgb()) {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "{message}: {actual} != {expected}"
            );
        }
    }
}

fn scale_scene(dynamic: bool) -> String {
    let factor = if dynamic {
        r#"
Texture "factor" "float" "bilerp"
    "float v00" [ 0.5 ] "float v01" [ 0.5 ]
    "float v10" [ 0.5 ] "float v11" [ 0.5 ]
Texture "scaled" "spectrum" "scale"
    "rgb tex" [ 0.8 0.6 0.4 ] "texture scale" [ "factor" ]
"#
    } else {
        r#"
Texture "scaled" "spectrum" "scale"
    "rgb tex" [ 0.8 0.6 0.4 ] "float scale" [ 0.5 ]
"#
    };
    format!(
        r#"Integrator "volpath" "integer maxdepth" [ 1 ]
Sampler "independent" "integer pixelsamples" [ 1 ]
Film "rgb" "integer xresolution" [ 8 ] "integer yresolution" [ 8 ]
LookAt 0 1.8 5.5  0 0 0  0 1 0
Camera "perspective" "float fov" [ 35 ]
WorldBegin
{factor}
LightSource "point" "point3 from" [ 0 3 2 ] "rgb I" [ 40 40 40 ]
Material "diffuse" "texture reflectance" [ "scaled" ]
Shape "trianglemesh"
    "point3 P" [ -3 0 -3  3 0 -3  3 0 3  -3 0 3 ]
    "integer indices" [ 0 2 1  0 3 2 ]
"#
    )
}

fn render(
    scene: &std::path::Path,
    directory: &std::path::Path,
    filename: &str,
    gpu: bool,
) -> Vec<RGBSpectrum> {
    let output = directory.join(filename);
    let mut command = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"));
    if gpu {
        command.arg("--use-gpu");
    }
    let status = command
        .args([
            "--outfile",
            output.to_str().unwrap(),
            scene.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "render failed for {}", scene.display());
    let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
    assert_eq!([resolution.x, resolution.y], [8, 8]);
    assert!(pixels.iter().all(RGBSpectrum::is_valid));
    pixels
}

fn image_energy(pixels: &[RGBSpectrum]) -> f32 {
    pixels
        .iter()
        .map(|pixel| pixel.to_rgb().into_iter().sum::<f32>())
        .sum()
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn conductor_remapped_roughness_matches_explicit_alpha() {
    let directory = tempfile::tempdir().unwrap();
    for material in ["eta-k", "reflectance"] {
        for textured in [false, true] {
            let mut images = Vec::new();
            for (name, roughness, remap) in [
                ("remapped", 0.01, ""),
                ("alpha", 0.1, "\"bool remaproughness\" [ false ]"),
            ] {
                let texture = if textured {
                    format!(
                        r#"Texture "rough" "float" "bilerp"
"float v00" [ {roughness} ] "float v01" [ {roughness} ]
"float v10" [ {roughness} ] "float v11" [ {roughness} ]"#
                    )
                } else {
                    String::new()
                };
                let rough = if textured {
                    r#""texture roughness" [ "rough" ]"#.to_string()
                } else {
                    format!(r#""float roughness" [ {roughness} ]"#)
                };
                let optics = if material == "eta-k" {
                    r#""rgb eta" [ 1 1 1 ] "rgb k" [ 2 2 2 ]"#
                } else {
                    r#""rgb reflectance" [ 0.5 0.5 0.5 ]"#
                };
                let scene = directory
                    .path()
                    .join(format!("{material}-{textured}-{name}.pbrt"));
                std::fs::write(
                    &scene,
                    format!(
                        r#"Integrator "path" "integer maxdepth" [ 1 ]
Sampler "halton" "integer pixelsamples" [ 16 ]
Film "rgb" "integer xresolution" [ 8 ] "integer yresolution" [ 8 ]
LookAt 0 0 3  0 0 0  0 1 0
Camera "perspective" "float fov" [ 35 ]
WorldBegin
{texture}
LightSource "point" "point3 from" [ 0.3 0.3 3 ] "rgb I" [ 10 10 10 ]
Material "conductor" {optics} {rough} {remap}
Shape "trianglemesh" "point3 P" [ -2 -2 0  2 -2 0  2 2 0  -2 2 0 ]
"integer indices" [ 0 1 2  0 2 3 ] "point2 uv" [ 0 0  1 0  1 1  0 1 ]
"#
                    ),
                )
                .unwrap();
                images.push(render(
                    &scene,
                    directory.path(),
                    &format!("{material}-{textured}-{name}.exr"),
                    true,
                ));
            }
            assert!(image_energy(&images[0]) > 0.0);
            assert_same_image(
                &images[0],
                &images[1],
                "conductor roughness remapping differs from explicit alpha",
            );
        }
    }
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn negative_conductor_roughness_texture_matches_zero() {
    let directory = tempfile::tempdir().unwrap();
    for (material, optics) in [
        ("eta-k", r#""rgb eta" [ 1 1 1 ] "rgb k" [ 2 2 2 ]"#),
        ("reflectance", r#""rgb reflectance" [ 0.5 0.5 0.5 ]"#),
    ] {
        let mut images = Vec::new();
        for (name, roughness) in [("negative", -0.01), ("zero", 0.0)] {
            let scene = directory.path().join(format!("{material}-{name}.pbrt"));
            std::fs::write(
                &scene,
                format!(
                    r#"Integrator "path" "integer maxdepth" [ 2 ]
Sampler "halton" "integer pixelsamples" [ 16 ]
Film "rgb" "integer xresolution" [ 8 ] "integer yresolution" [ 8 ]
LookAt 0 0 3  0 0 0  0 1 0
Camera "perspective" "float fov" [ 35 ]
WorldBegin
Texture "rough" "float" "bilerp"
"float v00" [ {roughness} ] "float v01" [ {roughness} ]
"float v10" [ {roughness} ] "float v11" [ {roughness} ]
LightSource "infinite" "rgb L" [ 1 1 1 ]
Material "conductor" {optics} "texture roughness" [ "rough" ]
Shape "trianglemesh" "point3 P" [ -2 -2 0  2 -2 0  2 2 0  -2 2 0 ]
"integer indices" [ 0 1 2  0 2 3 ] "point2 uv" [ 0 0  1 0  1 1  0 1 ]
"#
                ),
            )
            .unwrap();
            images.push(render(
                &scene,
                directory.path(),
                &format!("{material}-{name}.exr"),
                true,
            ));
        }
        assert!(image_energy(&images[1]) > 0.0);
        assert_same_image(
            &images[0],
            &images[1],
            "negative conductor roughness differs from zero",
        );
    }
}
