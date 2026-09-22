use std::path::Path;
use std::process::Command;

use image::{ImageBuffer, Rgb};
use pbrt_r4::util::imageio::read_image::read_image;
use pbrt_r4::util::spectrum::RGBSpectrum;

const FORWARD_PORTAL: &str = "-0.5 -0.5 1  -0.5 0.5 1  0.5 0.5 1  0.5 -0.5 1";
const DIRECT_PORTAL: &str = "-2 -2 2  -2 2 2  2 2 2  2 -2 2";
const LEFT_DIRECT_PORTAL: &str = "-2 -2 2  -2 2 2  0 2 2  0 -2 2";
const RIGHT_DIRECT_PORTAL: &str = "0 -2 2  0 2 2  2 2 2  2 -2 2";

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn portal_direct_pipeline_handles_positive_and_zero_distributions() {
    let directory = tempfile::tempdir().unwrap();
    let white_image = write_image(directory.path(), "white", [255, 255, 255]);
    let black_image = write_image(directory.path(), "black", [0, 0, 0]);

    let white = render_single_portal_scene(directory.path(), &white_image, "white");
    let black = render_single_portal_scene(directory.path(), &black_image, "black");

    assert!(white.iter().all(RGBSpectrum::is_valid));
    assert!(white.iter().any(|pixel| !pixel.is_black()));
    assert!(black.iter().all(RGBSpectrum::is_valid));
    assert!(black.iter().all(RGBSpectrum::is_black));
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn portal_escaped_radiance_is_clipped_to_the_projected_bounds() {
    let directory = tempfile::tempdir().unwrap();
    let image = write_image(directory.path(), "white", [255, 255, 255]);
    let source = escaped_scene(&portal_light(&image, FORWARD_PORTAL));
    let pixels = render_scene(directory.path(), "portal-escaped", &source, [16, 16]);

    assert!(pixels.iter().all(RGBSpectrum::is_valid));
    assert!(!pixels[8 * 16 + 8].is_black());
    assert!(pixels[0].is_black());
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn multiple_portal_images_contribute_independent_escaped_radiance() {
    let directory = tempfile::tempdir().unwrap();
    let red_image = write_image(directory.path(), "red", [255, 0, 0]);
    let blue_image = write_image(directory.path(), "blue", [0, 0, 255]);
    let red_light = portal_light(&red_image, FORWARD_PORTAL);
    let blue_light = portal_light(&blue_image, FORWARD_PORTAL);
    let red = render_scene(
        directory.path(),
        "red-portal",
        &escaped_scene(&red_light),
        [16, 16],
    );
    let blue = render_scene(
        directory.path(),
        "blue-portal",
        &escaped_scene(&blue_light),
        [16, 16],
    );
    let both = render_scene(
        directory.path(),
        "multiple-portals",
        &escaped_scene(&format!("{red_light}\n{blue_light}")),
        [16, 16],
    );

    let center = 8 * 16 + 8;
    assert!(!red[center].is_black());
    assert!(!blue[center].is_black());
    assert_rgb_close(both[center], red[center] + blue[center], 1e-4);
    assert!(both[0].is_black());
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn multiple_portals_preserve_direct_and_secondary_escape_mis_energy() {
    let directory = tempfile::tempdir().unwrap();
    let image = write_image(directory.path(), "white", [255, 255, 255]);
    let left_light = portal_light(&image, LEFT_DIRECT_PORTAL);
    let right_light = portal_light(&image, RIGHT_DIRECT_PORTAL);
    let left = render_scene(
        directory.path(),
        "left-bounce-portal",
        &diffuse_bounce_scene(&left_light),
        [1, 1],
    );
    let right = render_scene(
        directory.path(),
        "right-bounce-portal",
        &diffuse_bounce_scene(&right_light),
        [1, 1],
    );
    let multiple = render_scene(
        directory.path(),
        "multiple-bounce-portals",
        &diffuse_bounce_scene(&format!("{left_light}\n{right_light}")),
        [1, 1],
    );

    assert!(left[0].is_valid() && !left[0].is_black());
    assert!(right[0].is_valid() && !right[0].is_black());
    assert!(multiple[0].is_valid());
    assert_rgb_close(multiple[0], left[0] + right[0], 0.05);
}

fn write_image(directory: &Path, name: &str, rgb: [u8; 3]) -> std::path::PathBuf {
    let path = directory.join(format!("{name}.png"));
    ImageBuffer::<Rgb<u8>, _>::from_pixel(3, 3, Rgb(rgb))
        .save(&path)
        .unwrap();
    path
}

fn portal_light(image: &Path, portal: &str) -> String {
    format!(
        r#"LightSource "infinite"
    "string filename" ["{}"] "string encoding" ["linear"]
    "point3 portal" [{portal}]"#,
        image.display()
    )
}

fn escaped_scene(lights: &str) -> String {
    format!(
        r#"LookAt 0 0 0  0 0 1  0 1 0
Camera "perspective" "float fov" [90]
Film "rgb" "integer xresolution" [16] "integer yresolution" [16]
Sampler "independent" "integer pixelsamples" [1]
Integrator "path" "integer maxdepth" [1]
WorldBegin
{lights}
Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
Shape "trianglemesh"
    "point3 P" [-1 -1 -2  1 -1 -2  0 1 -2]
    "integer indices" [0 1 2]
"#
    )
}

fn diffuse_bounce_scene(lights: &str) -> String {
    format!(
        r#"LookAt 0 0 3  0 0 0  0 1 0
Camera "perspective" "float fov" [20]
Film "rgb" "integer xresolution" [1] "integer yresolution" [1]
Sampler "independent" "integer pixelsamples" [4096]
Integrator "path" "integer maxdepth" [1]
WorldBegin
{lights}
Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
Shape "trianglemesh"
    "point3 P" [-10 -10 0  10 -10 0  10 10 0  -10 10 0]
    "integer indices" [0 1 2  0 2 3]
"#
    )
}

fn render_single_portal_scene(directory: &Path, image: &Path, name: &str) -> Vec<RGBSpectrum> {
    let source = format!(
        r#"LookAt 0 0 3  0 0 0  0 1 0
Camera "perspective" "float fov" [45]
Film "rgb" "integer xresolution" [8] "integer yresolution" [8]
Sampler "independent" "integer pixelsamples" [1]
Integrator "path" "integer maxdepth" [1]
WorldBegin
{}
Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
Shape "trianglemesh"
    "point3 P" [-2 -2 0  2 -2 0  0 2 0]
    "integer indices" [0 1 2]
"#,
        portal_light(image, DIRECT_PORTAL)
    );
    render_scene(directory, &format!("portal-{name}"), &source, [8, 8])
}

fn render_scene(
    directory: &Path,
    name: &str,
    source: &str,
    expected_resolution: [i32; 2],
) -> Vec<RGBSpectrum> {
    let scene = directory.join(format!("{name}.pbrt"));
    std::fs::write(&scene, source).unwrap();
    let output = directory.join(format!("{name}.exr"));
    let status = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
        .args([
            "--use-gpu",
            "--outfile",
            output.to_str().unwrap(),
            scene.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "GPU Portal render failed for {name}");

    let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
    assert_eq!([resolution.x, resolution.y], expected_resolution);
    pixels
}

fn assert_rgb_close(actual: RGBSpectrum, expected: RGBSpectrum, relative_tolerance: f32) {
    for (actual, expected) in actual.to_rgb().into_iter().zip(expected.to_rgb()) {
        let tolerance = 1e-4 + relative_tolerance * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "RGB mismatch: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}
