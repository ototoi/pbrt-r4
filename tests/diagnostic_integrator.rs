use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn write_scene(name: &str, integrator: &str, mode: &str, output: &str) -> PathBuf {
    let scene_path = std::env::temp_dir().join(format!(
        "pbrt-r4-diagnostic-{name}-{}.pbrt",
        std::process::id()
    ));
    let scene = format!(
        "Integrator \"{integrator}\" \"string mode\" \"{mode}\" \"string filename\" \"{output}\"\n\
         Film \"rgb\" \"integer xresolution\" 8 \"integer yresolution\" 8\n\
         Sampler \"independent\" \"integer pixelsamples\" 1\n\
         Camera \"perspective\" \"float fov\" 45\n\
         WorldBegin\n\
         Material \"diffuse\" \"rgb reflectance\" [0.5 0.5 0.5]\n\
         Shape \"sphere\" \"float radius\" 1\n"
    );
    fs::write(&scene_path, scene).unwrap();
    scene_path
}

fn render(scene_path: &PathBuf) {
    let status = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
        .args(["--quiet", "--nthreads", "1"])
        .arg(scene_path)
        .status()
        .unwrap();
    assert!(status.success(), "diagnostic render failed: {status}");
}

#[test]
fn normal_mode_writes_rgb_exr() {
    let output = std::env::temp_dir().join(format!(
        "pbrt-r4-diagnostic-normal-{}.exr",
        std::process::id()
    ));
    let scene = write_scene("normal", "diagnostic", "normal", output.to_str().unwrap());
    render(&scene);

    let bytes = fs::read(&output).unwrap();
    assert!(bytes.windows(1).any(|channel| channel == b"R"));
    assert!(bytes.windows(1).any(|channel| channel == b"G"));
    assert!(bytes.windows(1).any(|channel| channel == b"B"));

    let _ = fs::remove_file(scene);
    let _ = fs::remove_file(output);
}

#[test]
fn depth_name_remains_a_compatibility_alias() {
    let output = std::env::temp_dir().join(format!(
        "pbrt-r4-diagnostic-depth-{}.exr",
        std::process::id()
    ));
    let scene = write_scene("depth", "depth", "t_hit", output.to_str().unwrap());
    render(&scene);

    let bytes = fs::read(&output).unwrap();
    assert!(!bytes.is_empty());

    let _ = fs::remove_file(scene);
    let _ = fs::remove_file(output);
}
