use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};

use pbrt_r4::util::base::Point2i;
use pbrt_r4::util::imageio::read_raw_image_exr_with_channels;
use tempfile::tempdir;

fn pspec() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pspec"))
}

#[test]
fn grid_writes_float_exr_and_radial_text() {
    let directory = tempdir().unwrap();
    let outbase = directory.path().join("grid");
    let output = pspec()
        .args([
            "grid",
            "--npoints",
            "4",
            "--nsets",
            "1",
            "--resolution",
            "5",
            "--outbase",
        ])
        .arg(&outbase)
        .output()
        .unwrap();

    assert!(output.status.success(), "stderr: {:?}", output.stderr);
    assert!(outbase.with_extension("exr").is_file());
    let (raw, channels) = read_raw_image_exr_with_channels(&outbase.with_extension("exr")).unwrap();
    assert_eq!(raw.resolution, Point2i::new(5, 5));
    assert_eq!(raw.channels, 1);
    assert_eq!(channels, vec!["power"]);
    let text = fs::read_to_string(outbase.with_extension("txt")).unwrap();
    assert_eq!(text.lines().count(), 1);
    assert!(text.starts_with("1 "));
}

#[test]
fn stdin_dat_reads_multiple_sets_until_nsets() {
    let directory = tempdir().unwrap();
    let outbase = directory.path().join("stdin");
    let mut child = pspec()
        .args([
            "stdin.dat",
            "--npoints",
            "2",
            "--nsets",
            "2",
            "--resolution",
            "5",
            "--outbase",
        ])
        .arg(&outbase)
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"0 0 0.5 0.5 # 0.25 0.25 0.75 0.75\n")
        .unwrap();

    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "stderr: {:?}", output.stderr);
    assert!(outbase.with_extension("exr").is_file());
    assert!(outbase.with_extension("txt").is_file());
}

#[test]
fn stdin_binary_reads_little_endian_points() {
    let directory = tempdir().unwrap();
    let outbase = directory.path().join("binary");
    let mut child = pspec()
        .args([
            "stdin.binary",
            "--npoints",
            "2",
            "--nsets",
            "1",
            "--resolution",
            "5",
            "--outbase",
        ])
        .arg(&outbase)
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = Vec::new();
    for value in [0.0_f32, 0.25, 0.5, 0.75] {
        input.extend_from_slice(&value.to_le_bytes());
    }
    child.stdin.take().unwrap().write_all(&input).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "stderr: {:?}", output.stderr);
}

#[test]
fn cwd_pts_uses_whitespace_tokens_and_stops_at_requested_sets() {
    let directory = tempdir().unwrap();
    fs::write(directory.path().join("pts-a"), "0 0 0.5 0.5\n").unwrap();
    fs::write(directory.path().join("pts-b"), "not a point\n").unwrap();
    let outbase = directory.path().join("cwd");
    let output = pspec()
        .args([
            "cwd.pts",
            "--npoints",
            "2",
            "--nsets",
            "1",
            "--resolution",
            "5",
            "--outbase",
        ])
        .arg(&outbase)
        .current_dir(directory.path())
        .output()
        .unwrap();
    assert!(output.status.success(), "stderr: {:?}", output.stderr);
}

#[test]
fn invalid_nsets_is_rejected() {
    let output = pspec().args(["grid", "--nsets", "0"]).output().unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--nsets must be greater than zero"));
}

#[test]
fn all_generated_sampler_variants_run() {
    let directory = tempdir().unwrap();
    for sampler in [
        "grid",
        "halton",
        "halton.owen",
        "halton.permutedigits",
        "independent",
        "lhs",
        "pmj02bn",
        "sobol",
        "sobol.fastowen",
        "sobol.owen",
        "sobol.permutedigits",
        "sobol.z",
        "stratified",
    ] {
        let outbase = directory.path().join(sampler.replace('.', "-"));
        let output = pspec()
            .args([
                sampler,
                "--npoints",
                "4",
                "--nsets",
                "1",
                "--resolution",
                "5",
                "--outbase",
            ])
            .arg(outbase)
            .output()
            .unwrap();
        assert!(output.status.success(), "{sampler}: {:?}", output.stderr);
    }
}
