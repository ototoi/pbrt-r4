use std::process::Command;

fn nanovdb2pbrt() -> Command {
    Command::new(env!("CARGO_BIN_EXE_nanovdb2pbrt"))
}

#[test]
fn help_exits_successfully() {
    let output = nanovdb2pbrt().arg("--help").output().unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("nanovdb2pbrt"));
}

#[test]
fn missing_filename_is_an_argument_error() {
    let output = nanovdb2pbrt().output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("must specify a nanovdb filename"));
    assert!(String::from_utf8_lossy(&output.stderr).contains("usage: nanovdb2pbrt"));
}

#[test]
fn missing_file_is_reported_without_dense_expansion() {
    let output = nanovdb2pbrt()
        .args([
            "--downsample=2",
            "--grid",
            "not-present",
            "does-not-exist.nvdb",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("does-not-exist.nvdb"));
    assert!(output.stdout.is_empty());
}

#[test]
fn negative_downsample_value_is_parsed_before_file_open() {
    let output = nanovdb2pbrt()
        .args(["--downsample", "-1", "does-not-exist.nvdb"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("does-not-exist.nvdb"));
    assert!(!stderr.contains("missing value for --downsample"));
}

#[test]
#[ignore = "requires a NanoVDB fixture specified by PBRT_NANOVDB_TEST_FILE"]
fn configured_fixture_is_converted() {
    let filename = std::env::var("PBRT_NANOVDB_TEST_FILE")
        .expect("PBRT_NANOVDB_TEST_FILE must name a test .nvdb fixture");
    let output = nanovdb2pbrt().arg(filename).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"integer nx\""));
    assert!(stdout.contains("\"point3 p0\""));
    assert!(stdout.contains("\"float density\""));
}
