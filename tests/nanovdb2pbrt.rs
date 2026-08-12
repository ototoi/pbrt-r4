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
