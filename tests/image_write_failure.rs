use std::fs;
use std::process::Command;

fn scene(film: &str) -> String {
    format!(
        r#"Film "{film}" "integer xresolution" [4] "integer yresolution" [4]
Sampler "independent" "integer pixelsamples" [1]
Integrator "path" "integer maxdepth" [1]
LookAt 0 0 4 0 0 0 0 1 0
Camera "perspective"
WorldBegin
Material "diffuse"
Shape "sphere"
WorldEnd
"#
    )
}

fn render(scene_path: &str, output: &str) -> std::process::ExitStatus {
    Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
        .args(["--outfile", output, scene_path])
        .output()
        .expect("run pbrt-r4")
        .status
}

#[test]
fn successful_write_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    let scene_path = dir.path().join("scene.pbrt");
    fs::write(&scene_path, scene("rgb")).unwrap();
    let output = dir.path().join("image.exr");
    assert!(render(scene_path.to_str().unwrap(), output.to_str().unwrap()).success());
    assert!(output.exists());
}

#[test]
fn failed_writes_exit_nonzero() {
    let dir = tempfile::tempdir().unwrap();
    let scene_path = dir.path().join("scene.pbrt");
    let blocker = dir.path().join("regular-file");
    fs::write(&blocker, "fixture").unwrap();

    for film in ["rgb", "gbuffer", "spectral"] {
        fs::write(&scene_path, scene(film)).unwrap();

        let blocked = blocker.join("image.exr");
        let status = render(scene_path.to_str().unwrap(), blocked.to_str().unwrap());
        assert!(!status.success(), "{}: parent is a regular file", film);

        let missing = dir.path().join("nonexistent").join("image.exr");
        let status = render(scene_path.to_str().unwrap(), missing.to_str().unwrap());
        assert!(!status.success(), "{}: parent does not exist", film);
    }
}

#[test]
fn gbuffer_and_spectral_succeed_on_valid_output() {
    let dir = tempfile::tempdir().unwrap();
    let scene_path = dir.path().join("scene.pbrt");
    for film in ["gbuffer", "spectral"] {
        fs::write(&scene_path, scene(film)).unwrap();
        let output = dir.path().join(format!("{}.exr", film));
        assert!(
            render(scene_path.to_str().unwrap(), output.to_str().unwrap()).success(),
            "{}",
            film
        );
        assert!(output.exists(), "{}", film);
    }
}
