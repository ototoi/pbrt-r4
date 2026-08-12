use image::{ImageBuffer, Luma, Rgb};
use std::process::Command;
use tempfile::tempdir;

fn imgtool() -> Command {
    Command::new(env!("CARGO_BIN_EXE_imgtool"))
}

#[test]
fn no_arguments_prints_help_to_stderr() {
    let output = imgtool().output().unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("usage: imgtool"));
}

#[test]
fn help_reports_unknown_command_as_failure() {
    let output = imgtool().args(["help", "missing"]).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("not known"));
}

#[test]
fn cat_prints_pixel_values() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("pixels.png");
    let image = ImageBuffer::<Rgb<u8>, _>::from_raw(2, 1, vec![0, 128, 255, 255, 64, 32]).unwrap();
    image.save(&path).unwrap();

    let output = imgtool()
        .args(["cat", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("(0, 0): 0.000000,0.501961,1.000000"));
    assert!(stdout.contains("(1, 0): 1.000000,0.250980,0.125490"));
}

#[test]
fn cat_list_prints_single_channel_rows() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("luma.png");
    let image = ImageBuffer::<Luma<u8>, _>::from_raw(2, 1, vec![0, 255]).unwrap();
    image.save(&path).unwrap();

    let output = imgtool()
        .args(["cat", "--list", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "{0.000000, 1.000000}\n"
    );
}

#[test]
fn info_prints_resolution_and_channel_statistics() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("pixels.png");
    let image = ImageBuffer::<Rgb<u8>, _>::from_raw(2, 1, vec![0, 128, 255, 255, 64, 32]).unwrap();
    image.save(&path).unwrap();

    let output = imgtool()
        .args(["info", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("resolution (2, 1)"));
    assert!(stdout.contains("color space: sRGB"));
    assert!(stdout.contains("pixel format: U256"));
    assert!(stdout.contains("R:"));
    assert!(stdout.contains("min     0.000000 max     1.000000"));
}

#[test]
fn convert_applies_crop_scale_and_flip() {
    let directory = tempdir().unwrap();
    let input = directory.path().join("input.png");
    let output_path = directory.path().join("output.exr");
    let image = ImageBuffer::<Luma<u8>, _>::from_raw(2, 2, vec![0, 64, 128, 255]).unwrap();
    image.save(&input).unwrap();

    let output = imgtool()
        .args([
            "convert",
            "--crop",
            "0,2,0,2",
            "--flipy",
            "--scale",
            "2",
            "--outfile",
            output_path.to_str().unwrap(),
            input.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let listed = imgtool()
        .args(["cat", output_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(listed.status.success());
    let stdout = String::from_utf8_lossy(&listed.stdout);
    assert!(stdout.contains("(0, 0): 1.003922"));
}

#[test]
fn makesky_writes_aces_exr() {
    let directory = tempdir().unwrap();
    let output_path = directory.path().join("sky.exr");
    let output = imgtool()
        .args([
            "makesky",
            "--resolution",
            "4",
            "--outfile",
            output_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_path.exists());

    let info = imgtool()
        .args(["info", output_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(info.status.success());
    let stdout = String::from_utf8_lossy(&info.stdout);
    assert!(stdout.contains("resolution (4, 4)"));
    assert!(stdout.contains("R:"));
}

#[test]
fn average_and_diff_use_per_channel_values() {
    let directory = tempdir().unwrap();
    for (name, value) in [("avg-0.png", 0_u8), ("avg-1.png", 255_u8)] {
        let image = ImageBuffer::<Luma<u8>, _>::from_raw(1, 1, vec![value]).unwrap();
        image.save(directory.path().join(name)).unwrap();
    }
    let average_path = directory.path().join("average.exr");
    let average = imgtool()
        .args([
            "average",
            "--outfile",
            average_path.to_str().unwrap(),
            directory.path().join("avg-").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        average.status.success(),
        "{}",
        String::from_utf8_lossy(&average.stderr)
    );

    let listed = imgtool()
        .args(["cat", average_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&listed.stdout).contains("0.500000"));

    let diff = imgtool()
        .args([
            "diff",
            "--reference",
            directory.path().join("avg-0.png").to_str().unwrap(),
            directory.path().join("avg-1.png").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!diff.status.success());
    assert!(String::from_utf8_lossy(&diff.stdout).contains("Y: 1.000000"));
}

#[test]
fn error_averages_matching_images_and_writes_error_image() {
    let directory = tempdir().unwrap();
    for (name, value) in [("render-0.png", 0_u8), ("render-1.png", 255_u8)] {
        let image = ImageBuffer::<Luma<u8>, _>::from_raw(1, 1, vec![value]).unwrap();
        image.save(directory.path().join(name)).unwrap();
    }
    let error_path = directory.path().join("error.exr");
    let output = imgtool()
        .args([
            "error",
            "--reference",
            directory.path().join("render-0.png").to_str().unwrap(),
            "--errorfile",
            error_path.to_str().unwrap(),
            directory.path().join("render-").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("MSE estimate"));
    assert!(error_path.exists());
}

#[test]
fn scalenormalmap_scales_xy_and_reconstructs_z() {
    let directory = tempdir().unwrap();
    let input_path = directory.path().join("normal.png");
    let output_path = directory.path().join("scaled.exr");
    let image = ImageBuffer::<Rgb<u8>, _>::from_raw(1, 1, vec![128, 128, 255]).unwrap();
    image.save(&input_path).unwrap();

    let output = imgtool()
        .args([
            "scalenormalmap",
            "--scale",
            "2",
            "--outfile",
            output_path.to_str().unwrap(),
            input_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let listed = imgtool()
        .args(["cat", output_path.to_str().unwrap()])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&listed.stdout);
    assert!(stdout.contains("0.503922,0.503922,0.999969"));
}
