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
fn disk_area_light_renders_through_gpu_light_bvh() {
    let scene = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/scenes/gpu-wavefront-disk-area-diffuse.pbrt");
    render_and_validate(&scene);
}

fn render_and_validate(scene: &std::path::Path) {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("light-bvh.exr");
    let status = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
        .args([
            "--use-gpu",
            "--quick",
            "--outfile",
            output.to_str().unwrap(),
            scene.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(
        status.success(),
        "GPU render failed for {}",
        scene.display()
    );

    let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
    assert_eq!(resolution.x, 128);
    assert_eq!(resolution.y, 128);
    assert!(!pixels.is_empty());
    assert!(pixels.iter().all(RGBSpectrum::is_valid));
}
