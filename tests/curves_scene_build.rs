use pbrt_r4::util::imageio::read_image::read_image;
use std::fs;
use std::path::Path;
use std::process::Command;

fn write_scene(path: &Path, output: &Path) {
    let scene = format!(
        r#"Integrator "diagnostic" "string mode" "normal" "string filename" "{}"
Film "rgb" "integer xresolution" 16 "integer yresolution" 16
Sampler "independent" "integer pixelsamples" 1
LookAt 0 0 -3  0 0 0  0 1 0
Camera "perspective" "float fov" 40
WorldBegin
Material "diffuse" "rgb reflectance" [ 0.5 0.5 0.5 ]
Shape "curve" "float width" 0.25
    "point3 P" [ -0.8 0 0  -0.3 0 0  0.3 0 0  0.8 0 0 ]
Shape "curve" "float width" 0.25
    "point3 P" [ -0.8 0.4 0  -0.3 0.4 0  0.3 0.4 0  0.8 0.4 0 ]
"#,
        output.display()
    );
    fs::write(path, scene).unwrap();
}

fn render(scene: &Path, mode: &str) {
    let output = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
        .args([
            "--quiet",
            "--nthreads",
            "1",
            "--scene-build",
            mode,
            "--scene-build-jobs",
            "2",
        ])
        .arg(scene)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{mode} scene build failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn serial_and_parallel_curve_scene_builds_match() {
    let dir = tempfile::tempdir().unwrap();
    let serial_scene = dir.path().join("serial.pbrt");
    let parallel_scene = dir.path().join("parallel.pbrt");
    let serial_output = dir.path().join("serial.exr");
    let parallel_output = dir.path().join("parallel.exr");
    write_scene(&serial_scene, &serial_output);
    write_scene(&parallel_scene, &parallel_output);

    render(&serial_scene, "serial");
    render(&parallel_scene, "parallel");

    let (serial_pixels, serial_resolution) = read_image(serial_output.to_str().unwrap()).unwrap();
    let (parallel_pixels, parallel_resolution) =
        read_image(parallel_output.to_str().unwrap()).unwrap();
    assert_eq!(serial_resolution, parallel_resolution);
    assert_eq!(serial_pixels.len(), parallel_pixels.len());
    for (serial, parallel) in serial_pixels.iter().zip(&parallel_pixels) {
        assert_eq!(serial, parallel);
    }
}

#[test]
fn curves_shape_input_is_rejected_without_panicking() {
    let dir = tempfile::tempdir().unwrap();
    let scene = dir.path().join("reserved-curves.pbrt");
    fs::write(
        &scene,
        r#"WorldBegin
Shape "curves"
Shape "curve"
    "point3 P" [ 0 0 0  0 0 1  1 0 1  1 0 2 ]
WorldEnd
"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
        .args(["--quiet", "--scene-build", "serial"])
        .arg(scene)
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Shape \"curves\" is reserved"),
        "unexpected error: {stderr}"
    );
}
