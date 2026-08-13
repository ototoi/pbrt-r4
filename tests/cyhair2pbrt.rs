use std::io::Write;
use std::process::Command;

use tempfile::NamedTempFile;

const FLAG_SEGMENTS: u32 = 1 << 0;
const FLAG_POINTS: u32 = 1 << 1;

fn write_hair(strands: &[(&[u16], &[[f32; 3]])]) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    let num_strands = strands.len() as u32;
    let total_points = strands
        .iter()
        .map(|(_, points)| points.len())
        .sum::<usize>() as u32;
    let mut header = [0u8; 128];
    header[..4].copy_from_slice(b"HAIR");
    header[4..8].copy_from_slice(&num_strands.to_le_bytes());
    header[8..12].copy_from_slice(&total_points.to_le_bytes());
    header[12..16].copy_from_slice(&(FLAG_SEGMENTS | FLAG_POINTS).to_le_bytes());
    file.write_all(&header).unwrap();
    for (segments, _) in strands {
        assert_eq!(segments.len(), 1);
        file.write_all(&segments[0].to_le_bytes()).unwrap();
    }
    for (_, points) in strands {
        for point in *points {
            for value in point {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
        }
    }
    file
}

fn run(input: &std::path::Path, extra: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_cyhair2pbrt"))
        .arg(input)
        .arg("-")
        .args(extra)
        .output()
        .unwrap()
}

#[test]
fn converts_segments_and_applies_zup_to_yup() {
    let file = write_hair(&[(
        &[4],
        &[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ],
    )]);
    let output = run(file.path(), &[]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("The number of strands = 2. user_thickness = 1.000000"));
    assert!(stdout.contains(
        "0.000000 0.000000 0.000000 0.333333 1.000000 0.666667 0.666667 2.000000 1.333333 1.000000 3.000000 2.000000"
    ));
    assert!(stdout.contains("width0\" [ 1.000000 ] \"float width1\" [ 1.000000 ]"));
    assert!(output.stderr.starts_with(b"Converted 2 strands."));
}

#[test]
fn max_strands_zero_matches_v4_all_strands_behavior() {
    let file = write_hair(&[(
        &[3],
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
    )]);
    let output = run(file.path(), &["0"]);
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("The number of strands = 1."));
    assert!(stdout.contains("Shape \"curve\""));
}

#[test]
fn handles_segment_count_boundaries_and_strand_offsets() {
    let file = write_hair(&[
        (&[2], &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        (
            &[3],
            &[
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [13.0, 0.0, 0.0],
            ],
        ),
        (
            &[4],
            &[
                [20.0, 0.0, 0.0],
                [21.0, 0.0, 0.0],
                [22.0, 0.0, 0.0],
                [23.0, 0.0, 0.0],
                [24.0, 0.0, 0.0],
            ],
        ),
    ]);
    let output = run(file.path(), &[]);
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("The number of strands = 3."));
    assert!(stdout.contains("10.000000"));
    assert!(stdout.contains("20.000000"));
}

#[test]
fn rejects_invalid_arguments_and_truncated_input() {
    let no_args = Command::new(env!("CARGO_BIN_EXE_cyhair2pbrt"))
        .output()
        .unwrap();
    assert!(!no_args.status.success());
    assert!(String::from_utf8_lossy(&no_args.stderr).contains("usage: cyhair2pbrt"));

    let file = write_hair(&[(
        &[3],
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
    )]);
    let invalid_max = run(file.path(), &["-2"]);
    assert!(!invalid_max.status.success());
    assert!(String::from_utf8_lossy(&invalid_max.stderr).contains("max strands"));

    let mut truncated = NamedTempFile::new().unwrap();
    truncated.write_all(b"HAIR").unwrap();
    let invalid_file = run(truncated.path(), &[]);
    assert!(!invalid_file.status.success());
    assert!(String::from_utf8_lossy(&invalid_file.stderr).contains("128-byte header"));
}

#[test]
fn rejects_default_segment_point_count_mismatch() {
    let mut file = NamedTempFile::new().unwrap();
    let mut header = [0u8; 128];
    header[..4].copy_from_slice(b"HAIR");
    header[4..8].copy_from_slice(&1u32.to_le_bytes());
    header[8..12].copy_from_slice(&4u32.to_le_bytes());
    header[12..16].copy_from_slice(&FLAG_POINTS.to_le_bytes());
    header[16..20].copy_from_slice(&4u32.to_le_bytes());
    file.write_all(&header).unwrap();
    for i in 0..4 {
        file.write_all(&(i as f32).to_le_bytes()).unwrap();
        file.write_all(&0.0f32.to_le_bytes()).unwrap();
        file.write_all(&0.0f32.to_le_bytes()).unwrap();
    }
    let output = run(file.path(), &[]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("point count mismatch"));
}
