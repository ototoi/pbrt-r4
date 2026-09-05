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
fn indirect_area_light_hit_adds_emission_at_maxdepth_two() {
    let scene = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/scenes/gpu-wavefront-multiple-area-diffuse.pbrt");
    let primary_only = render_pixels_with_maxdepth(&scene, 0);
    let one_bounce = render_pixels_with_maxdepth(&scene, 1);
    let two_bounce = render_pixels_with_maxdepth(&scene, 2);
    let energy_primary = image_energy(&primary_only);
    let energy_one_bounce = image_energy(&one_bounce);
    let energy_two_bounce = image_energy(&two_bounce);

    assert_eq!(energy_primary, 0.0);
    assert!(energy_one_bounce > 0.0);
    assert!(
        energy_one_bounce > energy_primary,
        "maxdepth=1 should include an indirect emissive hit: depth0={energy_primary}, depth1={energy_one_bounce}"
    );
    assert!(energy_two_bounce >= energy_one_bounce);
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

fn render_pixels_with_maxdepth(scene: &std::path::Path, maxdepth: u32) -> Vec<RGBSpectrum> {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("indirect.exr");
    let source = std::fs::read_to_string(scene)
        .unwrap()
        .replacen("    0 1 0\n    0 1 0", "    0 0 0\n    0 1 0", 1)
        .replacen(
            "    \"integer maxdepth\" [ 2 ]",
            &format!("    \"integer maxdepth\" [ {maxdepth} ]"),
            1,
        );
    let render_scene = directory.path().join("scene.pbrt");
    std::fs::write(&render_scene, source).unwrap();
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
    assert!(status.success(), "GPU render failed at maxdepth={maxdepth}");

    let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
    assert_eq!(resolution.x, 128);
    assert_eq!(resolution.y, 128);
    assert!(!pixels.is_empty());
    assert!(pixels.iter().all(RGBSpectrum::is_valid));
    pixels
}

fn image_energy(pixels: &[RGBSpectrum]) -> f32 {
    pixels
        .iter()
        .map(|pixel| pixel.to_rgb().into_iter().sum::<f32>())
        .sum()
}
