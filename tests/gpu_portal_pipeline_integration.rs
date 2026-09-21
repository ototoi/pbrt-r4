use std::path::Path;
use std::process::Command;

use image::{ImageBuffer, Rgb};
use pbrt_r4::util::imageio::read_image::read_image;
use pbrt_r4::util::spectrum::RGBSpectrum;

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn portal_direct_pipeline_handles_positive_and_zero_distributions() {
    let directory = tempfile::tempdir().unwrap();
    let white_image = directory.path().join("white.png");
    let black_image = directory.path().join("black.png");
    ImageBuffer::<Rgb<u8>, _>::from_pixel(3, 3, Rgb([255, 255, 255]))
        .save(&white_image)
        .unwrap();
    ImageBuffer::<Rgb<u8>, _>::from_pixel(3, 3, Rgb([0, 0, 0]))
        .save(&black_image)
        .unwrap();

    let white = render_portal_scene(directory.path(), &white_image, "white");
    let black = render_portal_scene(directory.path(), &black_image, "black");

    assert!(white.iter().all(RGBSpectrum::is_valid));
    assert!(white.iter().any(|pixel| !pixel.is_black()));
    assert!(black.iter().all(RGBSpectrum::is_valid));
    assert!(black.iter().all(RGBSpectrum::is_black));
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn portal_escaped_radiance_is_clipped_to_the_projected_bounds() {
    let directory = tempfile::tempdir().unwrap();
    let image = directory.path().join("white.png");
    ImageBuffer::<Rgb<u8>, _>::from_pixel(3, 3, Rgb([255, 255, 255]))
        .save(&image)
        .unwrap();
    let scene = directory.path().join("portal-escaped.pbrt");
    std::fs::write(
        &scene,
        format!(
            r#"LookAt 0 0 0  0 0 1  0 1 0
Camera "perspective" "float fov" [90]
Film "rgb" "integer xresolution" [16] "integer yresolution" [16]
Sampler "independent" "integer pixelsamples" [1]
Integrator "path" "integer maxdepth" [1]
WorldBegin
LightSource "infinite"
    "string filename" ["{}"] "string encoding" ["linear"]
    "point3 portal" [-0.5 -0.5 1  -0.5 0.5 1  0.5 0.5 1  0.5 -0.5 1]
Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
Shape "trianglemesh"
    "point3 P" [-1 -1 -2  1 -1 -2  0 1 -2]
    "integer indices" [0 1 2]
"#,
            image.display()
        ),
    )
    .unwrap();
    let output = directory.path().join("portal-escaped.exr");
    let status = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
        .args([
            "--use-gpu",
            "--outfile",
            output.to_str().unwrap(),
            scene.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "GPU escaped Portal render failed");

    let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
    assert_eq!([resolution.x, resolution.y], [16, 16]);
    assert!(pixels.iter().all(RGBSpectrum::is_valid));
    assert!(!pixels[8 * 16 + 8].is_black());
    assert!(pixels[0].is_black());
}

fn render_portal_scene(directory: &Path, image: &Path, name: &str) -> Vec<RGBSpectrum> {
    let scene = directory.join(format!("portal-{name}.pbrt"));
    std::fs::write(
        &scene,
        format!(
            r#"LookAt 0 0 3  0 0 0  0 1 0
Camera "perspective" "float fov" [45]
Film "rgb" "integer xresolution" [8] "integer yresolution" [8]
Sampler "independent" "integer pixelsamples" [1]
Integrator "path" "integer maxdepth" [1]
WorldBegin
LightSource "infinite"
    "string filename" ["{}"] "string encoding" ["linear"]
    "point3 portal" [-2 -2 2  -2 2 2  2 2 2  2 -2 2]
Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
Shape "trianglemesh"
    "point3 P" [-2 -2 0  2 -2 0  0 2 0]
    "integer indices" [0 1 2]
"#,
            image.display()
        ),
    )
    .unwrap();
    let output = directory.join(format!("portal-{name}.exr"));
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
    assert_eq!([resolution.x, resolution.y], [8, 8]);
    pixels
}
