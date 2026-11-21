use std::path::Path;
use std::process::{Command, Output};

use pbrt_r4::util::imageio::read_image::read_image;

const HEADER: &str = r#"MakeNamedMedium "clear" "string type" "homogeneous"
    "rgb sigma_a" [0 0 0] "rgb sigma_s" [0 0 0]
MediumInterface "clear" "clear"
LookAt 0 0 5  0 0 0  0 1 0
Camera "perspective" "float fov" [35]
Film "rgb" "integer xresolution" [16] "integer yresolution" [16]
PixelFilter "box"
Sampler "independent" "integer pixelsamples" [4]
Integrator "volpath" "integer maxdepth" [2]
WorldBegin
Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
Shape "trianglemesh" "point3 P" [-5 -5 0  5 -5 0  5 5 0  -5 5 0] "integer indices" [0 1 2  0 2 3]
"#;

/// A light near the floor, below any camera-side planes.
const LOW_LIGHT: &str = r#"AttributeBegin
    AreaLightSource "diffuse" "rgb L" [4 4 4] "bool twosided" true
    Shape "trianglemesh" "point3 P" [-0.3 -0.3 0.5  0.3 -0.3 0.5  0.3 0.3 0.5  -0.3 0.3 0.5] "integer indices" [0 1 2  0 2 3]
AttributeEnd
"#;

/// A light off to the side, behind any light-side planes.
const SIDE_LIGHT: &str = r#"AttributeBegin
    AreaLightSource "diffuse" "rgb L" [8 8 8] "bool twosided" true
    Shape "trianglemesh" "point3 P" [3 -1 0.2  3 1 0.2  3 1 1.2  3 -1 1.2] "integer indices" [0 1 2  0 2 3]
AttributeEnd
"#;

/// Interface-only planes; `plane(i)` returns the four corners of plane `i`.
fn interface_planes(count: usize, plane: impl Fn(usize) -> [[f32; 3]; 4]) -> String {
    let mut points = Vec::new();
    let mut indices = Vec::new();
    for i in 0..count {
        let base = 4 * i;
        points.extend(plane(i).into_iter().flatten().map(|v| v.to_string()));
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3].map(|v| v.to_string()));
    }
    format!(
        "AttributeBegin\n    Material \"\"\n    MediumInterface \"clear\" \"clear\"\n    Shape \"trianglemesh\" \"point3 P\" [{}] \"integer indices\" [{}]\nAttributeEnd\n",
        points.join(" "),
        indices.join(" "),
    )
}

/// Horizontal planes between the camera (z = 5) and the low light.
fn camera_side_planes(count: usize) -> String {
    interface_planes(count, |i| {
        let z = 1.0 + 3.0 * (i as f32 + 0.5) / count as f32;
        [
            [-5.0, -5.0, z],
            [5.0, -5.0, z],
            [5.0, 5.0, z],
            [-5.0, 5.0, z],
        ]
    })
}

/// Vertical planes between the visible floor and the side light.
fn light_side_planes(count: usize) -> String {
    interface_planes(count, |i| {
        let x = 1.8 + 0.9 * (i as f32 + 0.5) / count as f32;
        [
            [x, -5.0, -1.0],
            [x, 5.0, -1.0],
            [x, 5.0, 2.0],
            [x, -5.0, 2.0],
        ]
    })
}

fn run(directory: &Path, name: &str, world: &str) -> Output {
    let scene = directory.join(format!("{name}.pbrt"));
    std::fs::write(&scene, format!("{HEADER}{world}")).unwrap();
    Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
        .args(["--gpu", "--quiet", "--outfile"])
        .arg(directory.join(format!("{name}.exr")))
        .arg(&scene)
        .output()
        .unwrap()
}

fn render(directory: &Path, name: &str, world: &str) -> Vec<[f64; 3]> {
    let result = run(directory, name, world);
    assert!(
        result.status.success(),
        "{name} failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output = directory.join(format!("{name}.exr"));
    let (pixels, _) = read_image(output.to_str().unwrap()).unwrap();
    pixels
        .iter()
        .map(|pixel| pixel.to_rgb().map(f64::from))
        .collect()
}

/// Zero-coefficient media draw no random numbers, so crossing interfaces
/// changes the image only by the ray-origin offset at each crossing.
fn assert_planes_do_not_change_image(light: &str, planes: &str) {
    let directory = tempfile::tempdir().unwrap();
    let plain = render(directory.path(), "plain", light);
    let crossed = render(directory.path(), "crossed", &format!("{light}{planes}"));
    assert!(plain.iter().flatten().any(|v| *v > 0.0));
    for (index, (plain, crossed)) in plain.iter().zip(&crossed).enumerate() {
        for (plain, crossed) in plain.iter().zip(crossed) {
            assert!(
                (plain - crossed).abs() <= 1.0e-3 * plain.abs().max(1.0e-3),
                "pixel {index} is {crossed} across the planes, {plain} without them"
            );
        }
    }
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn camera_rays_cross_many_interfaces_without_changing_the_image() {
    assert_planes_do_not_change_image(LOW_LIGHT, &camera_side_planes(40));
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn shadow_rays_cross_many_interfaces_without_changing_the_image() {
    assert_planes_do_not_change_image(SIDE_LIGHT, &light_side_planes(40));
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn crossing_more_interfaces_than_the_limit_fails_the_render() {
    let directory = tempfile::tempdir().unwrap();
    let world = format!("{LOW_LIGHT}{}", camera_side_planes(1100));
    let result = run(directory.path(), "too-many", &world);
    assert!(
        !result.status.success(),
        "crossing 1100 interfaces must fail the render"
    );
    assert!(String::from_utf8_lossy(&result.stderr).contains("rendering reported an error"));
}
