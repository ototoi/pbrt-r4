use std::path::Path;
use std::process::Command;

use pbrt_r4::util::imageio::read_image::read_image;

const HEADER: &str = r#"MakeNamedMedium "fog" "string type" "homogeneous"
    "rgb sigma_a" [0.05 0.1 0.2] "rgb sigma_s" [0 0 0]
MediumInterface "fog" "fog"
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

/// Three interface-only planes above a light near the floor: camera rays
/// cross all of them, shadow rays none.
const CAMERA_SIDE: &str = r#"AttributeBegin
    AreaLightSource "diffuse" "rgb L" [4 4 4] "bool twosided" true
    Shape "trianglemesh" "point3 P" [-0.3 -0.3 0.5  0.3 -0.3 0.5  0.3 0.3 0.5  -0.3 0.3 0.5] "integer indices" [0 1 2  0 2 3]
AttributeEnd
AttributeBegin
    Material ""
    MediumInterface "fog" "fog"
    Shape "trianglemesh" "point3 P" [-5 -5 1  5 -5 1  5 5 1  -5 5 1] "integer indices" [0 1 2  0 2 3]
    Shape "trianglemesh" "point3 P" [-5 -5 2  5 -5 2  5 5 2  -5 5 2] "integer indices" [0 1 2  0 2 3]
    Shape "trianglemesh" "point3 P" [-5 -5 3  5 -5 3  5 5 3  -5 5 3] "integer indices" [0 1 2  0 2 3]
AttributeEnd
"#;

/// Three interface-only planes between the visible floor and a light off to
/// the side: shadow rays (and bounce rays toward the light) cross them.
const LIGHT_SIDE: &str = r#"AttributeBegin
    AreaLightSource "diffuse" "rgb L" [8 8 8] "bool twosided" true
    Shape "trianglemesh" "point3 P" [3 -1 0.2  3 1 0.2  3 1 1.2  3 -1 1.2] "integer indices" [0 1 2  0 2 3]
AttributeEnd
AttributeBegin
    Material ""
    MediumInterface "fog" "fog"
    Shape "trianglemesh" "point3 P" [1.8 -5 -1  1.8 5 -1  1.8 5 2  1.8 -5 2] "integer indices" [0 1 2  0 2 3]
    Shape "trianglemesh" "point3 P" [2.2 -5 -1  2.2 5 -1  2.2 5 2  2.2 -5 2] "integer indices" [0 1 2  0 2 3]
    Shape "trianglemesh" "point3 P" [2.6 -5 -1  2.6 5 -1  2.6 5 2  2.6 -5 2] "integer indices" [0 1 2  0 2 3]
AttributeEnd
"#;

struct Render {
    pixels: Vec<[f32; 3]>,
    bytes: Vec<u8>,
    retried: bool,
}

fn render(directory: &Path, name: &str, world: &str, rounds: u32) -> Render {
    let scene = directory.join(format!("{name}.pbrt"));
    let output = directory.join(format!("{name}-{rounds}.exr"));
    std::fs::write(&scene, format!("{HEADER}{world}")).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
        .env("PBRT_R4_GPU_CONTINUATION_ROUNDS", rounds.to_string())
        .args([
            "--gpu",
            "--quiet",
            "--logtostderr",
            "--minloglevel",
            "0",
            "--outfile",
        ])
        .arg(&output)
        .arg(&scene)
        .output()
        .unwrap();
    let log = String::from_utf8_lossy(&result.stderr);
    assert!(
        result.status.success(),
        "{name} with {rounds} rounds failed:\n{log}"
    );
    let (pixels, _) = read_image(output.to_str().unwrap()).unwrap();
    Render {
        pixels: pixels
            .iter()
            .map(|pixel| pixel.to_rgb().map(|v| v as f32))
            .collect(),
        bytes: std::fs::read(&output).unwrap(),
        retried: log.contains("GPU render: retrying"),
    }
}

fn check_scene(name: &str, world: &str) {
    let directory = tempfile::tempdir().unwrap();
    let waiting = render(directory.path(), name, world, 0);
    let enough = render(directory.path(), name, world, 8);
    let short = render(directory.path(), name, world, 1);

    assert!(!waiting.retried && !enough.retried);
    assert!(
        enough.bytes == waiting.bytes,
        "{name}: enough GPU rounds must match the waiting loop exactly"
    );
    assert!(
        short.retried,
        "{name}: one round must leave continuations for a retry"
    );
    for (index, (short, waiting)) in short.pixels.iter().zip(&waiting.pixels).enumerate() {
        for (short, waiting) in short.iter().zip(waiting) {
            assert!(
                (short - waiting).abs() <= 1.0e-5 * waiting.abs().max(1.0),
                "{name}: pixel {index} is {short} after a retry, {waiting} with the waiting loop"
            );
        }
    }
    assert!(waiting.pixels.iter().flatten().any(|v| *v > 0.0));
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn camera_rays_past_the_gpu_rounds_are_retried() {
    check_scene("camera-side", CAMERA_SIDE);
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn shadow_rays_past_the_gpu_rounds_are_retried() {
    check_scene("light-side", LIGHT_SIDE);
}
