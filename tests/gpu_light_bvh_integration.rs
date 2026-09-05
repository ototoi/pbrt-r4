use std::process::Command;

use pbrt_r4::util::imageio::read_image::read_image;
use pbrt_r4::util::spectrum::RGBSpectrum;

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn area_light_scene_renders_through_gpu_light_bvh() {
    let scene = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/scenes/gpu-wavefront-single-area-diffuse.pbrt");
    render_and_validate(&scene);
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn multiple_area_light_groups_render_through_gpu_light_bvh() {
    let scene = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/scenes/gpu-wavefront-multiple-area-diffuse.pbrt");
    render_and_validate(&scene);
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn multiple_area_light_groups_render_through_gpu_uniform_sampler() {
    let scene = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/scenes/gpu-wavefront-multiple-area-diffuse.pbrt");
    render_and_validate_with_sampler(&scene, Some("uniform"));
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn disk_area_light_renders_through_gpu_light_bvh() {
    let scene = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/scenes/gpu-wavefront-disk-area-diffuse.pbrt");
    render_and_validate(&scene);
}

fn render_and_validate(scene: &std::path::Path) {
    render_and_validate_with_sampler(scene, None);
}

fn render_and_validate_with_sampler(scene: &std::path::Path, light_sampler: Option<&str>) {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("light-bvh.exr");
    let render_scene = if let Some(light_sampler) = light_sampler {
        let source = std::fs::read_to_string(scene).unwrap();
        let source = source.replacen(
            "    \"integer maxdepth\" [ 2 ]",
            &format!(
                "    \"integer maxdepth\" [ 2 ]\n    \"string lightsampler\" [ \"{light_sampler}\" ]"
            ),
            1,
        );
        let path = directory.path().join("scene.pbrt");
        std::fs::write(&path, source).unwrap();
        path
    } else {
        scene.to_path_buf()
    };
    let status = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
        .args([
            "--use-gpu",
            "--quick",
            "--outfile",
            output.to_str().unwrap(),
            render_scene.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(
        status.success(),
        "GPU render failed for {} sampler={light_sampler:?}",
        scene.display()
    );

    let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
    assert_eq!(resolution.x, 128);
    assert_eq!(resolution.y, 128);
    assert!(!pixels.is_empty());
    assert!(pixels.iter().all(RGBSpectrum::is_valid));
    assert!(pixels.iter().any(|pixel| !pixel.is_black()));
}
