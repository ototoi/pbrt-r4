use std::process::Command;

#[test]
fn gpu_and_wavefront_options_match_pbrt_v4_spelling() {
    let output = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
        .arg("--help")
        .output()
        .expect("failed to run pbrt-r4 --help");

    assert!(output.status.success());

    let help = String::from_utf8(output.stdout).expect("help output was not UTF-8");
    assert!(
        help.lines()
            .any(|line| line.trim_start().starts_with("--gpu ")),
        "missing --gpu option in help:\n{help}"
    );
    assert!(
        help.lines()
            .any(|line| line.trim_start().starts_with("--wavefront ")),
        "missing --wavefront option in help:\n{help}"
    );
    assert!(
        !help.contains("--use-gpu"),
        "unexpected --use-gpu option in help:\n{help}"
    );
    assert!(
        !help.contains("--use-wavefront"),
        "unexpected --use-wavefront option in help:\n{help}"
    );
}
