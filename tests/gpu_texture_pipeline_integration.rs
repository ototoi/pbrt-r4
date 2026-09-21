use std::process::Command;

use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::pipeline::Pipeline;
use pbrt_r4::gpu::webgpu::stages::RequiredLimits;
use pbrt_r4::util::imageio::read_image::read_image;
use pbrt_r4::util::spectrum::RGBSpectrum;

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn texture_material_pipeline_compiles() {
    let _ = env_logger::builder().is_test(true).try_init();
    let required = RequiredLimits {
        storage_buffers_per_shader_stage: 30,
        uniform_buffers_per_shader_stage: 5,
        buffers_and_acceleration_structures_per_shader_stage: 36,
        bind_groups: 2,
    };
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
