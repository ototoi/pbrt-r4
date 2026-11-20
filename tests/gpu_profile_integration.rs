#![cfg(feature = "webgpu")]

use std::path::{Path, PathBuf};

fn smoke_scene() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/scenes/gpu-spot-distant-smoke.pbrt")
}

fn render_smoke(profile: bool, out: &Path) -> std::process::Output {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_pbrt-r4"));
    command.arg("--gpu");
    if profile {
        command.arg("--gpu-profile");
    }
    command
        .arg("--outfile")
        .arg(out)
        .arg(smoke_scene())
        .output()
        .unwrap()
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn gpu_profile_prints_phase_and_stage_tables() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("p.exr");
    let output = render_smoke(true, &out);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    for phase in ["parse", "flatten", "pipeline create", "render loop"] {
        assert!(stderr.contains(phase), "missing {phase}: {stderr}");
    }
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn gpu_profile_does_not_change_the_image() {
    let dir = tempfile::tempdir().unwrap();
    let profiled = dir.path().join("profiled.exr");
    let plain = dir.path().join("plain.exr");
    assert!(render_smoke(true, &profiled).status.success());
    assert!(render_smoke(false, &plain).status.success());
    let (a, res_a) = pbrt_r4::util::imageio::read_image(&profiled.to_string_lossy()).unwrap();
    let (b, res_b) = pbrt_r4::util::imageio::read_image(&plain.to_string_lossy()).unwrap();
    assert_eq!(res_a, res_b);
    assert_eq!(a, b);
}
