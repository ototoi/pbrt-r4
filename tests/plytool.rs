use std::io::Write;
use std::process::Command;

use tempfile::{NamedTempFile, TempDir};

#[test]
fn reports_mesh_counts_and_bounds() {
    let mut file = NamedTempFile::with_suffix(".ply").unwrap();
    writeln!(
        file,
        "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n9 9 9\n3 0 1 2"
    )
    .unwrap();

    let output = run_info(file.path());
    assert!(output.contains("Triangles: 1"));
    assert!(output.contains("Quads: 0"));
    assert!(output.contains("Vertex positions: 4"));
    assert!(output.contains("Vertex normals: 0"));
    assert!(output.contains("Vertex uvs: 0"));
    assert!(output.contains("Face indices: 0"));
    assert!(output.contains("Notice: vertex 3 is not used."));
    assert!(output.contains("Bounding box: [ 0 0 0 ] - [ 9 9 9 ]"));
}

#[test]
fn reports_bounds_for_negative_coordinates() {
    let mut file = NamedTempFile::with_suffix(".ply").unwrap();
    writeln!(
        file,
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n-3 -4 -5\n-1 -2 -3\n-2 -6 -4\n3 0 1 2"
    )
    .unwrap();

    let output = run_info(file.path());
    assert!(output.contains("Bounding box: [ -3 -6 -5 ] - [ -1 -2 -3 ]"));
}

fn run_info(path: &std::path::Path) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_plytool"))
        .arg("info")
        .arg(path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "plytool info failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn binary_matches_cli_error_and_empty_info_behavior() {
    let binary = env!("CARGO_BIN_EXE_plytool");

    let empty_info = Command::new(binary).arg("info").output().unwrap();
    assert!(empty_info.status.success());
    assert!(empty_info.stdout.is_empty());

    let unknown = Command::new(binary).arg("unknown").output().unwrap();
    assert!(!unknown.status.success());
    assert!(String::from_utf8_lossy(&unknown.stderr).contains("unknown: command unknown"));
    assert!(String::from_utf8_lossy(&unknown.stdout).contains("usage: plytool <command>"));
}

#[test]
fn cat_prints_mesh_elements() {
    let file = source_ply();
    let output = Command::new(env!("CARGO_BIN_EXE_plytool"))
        .args(["cat", file.path().to_str().unwrap()])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Triangle: 0 1 2"));
    assert!(stdout.contains("Vertex position 0:"));
}

#[test]
fn split_writes_remapped_triangle_files() {
    let file = source_ply_with_two_triangles();
    let directory = TempDir::new().unwrap();
    let outbase = directory.path().join("part");
    let output = Command::new(env!("CARGO_BIN_EXE_plytool"))
        .args([
            "split",
            "--maxfaces",
            "1",
            "--outbase",
            outbase.to_str().unwrap(),
            file.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(directory.path().join("part-000.ply").exists());
    assert!(directory.path().join("part-001.ply").exists());
}

#[test]
fn displace_reads_image_and_writes_ply() {
    let file = source_ply_with_uvs();
    let directory = TempDir::new().unwrap();
    let image_path = directory.path().join("displacement.png");
    let output_path = directory.path().join("displaced.ply");
    let image = image::GrayImage::from_pixel(1, 1, image::Luma([128]));
    image.save(&image_path).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_plytool"))
        .args([
            "displace",
            "--edge-length",
            "10",
            "--image",
            image_path.to_str().unwrap(),
            "--outfile",
            output_path.to_str().unwrap(),
            file.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output_path.exists());
}

fn source_ply() -> NamedTempFile {
    source_ply_with_contents(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n",
    )
}

fn source_ply_with_two_triangles() -> NamedTempFile {
    source_ply_with_contents(
        "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n3 0 1 2\n3 1 3 2\n",
    )
}

fn source_ply_with_uvs() -> NamedTempFile {
    source_ply_with_contents(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty float u\nproperty float v\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0 0 0\n1 0 0 1 0\n0 1 0 0 1\n3 0 1 2\n",
    )
}

fn source_ply_with_contents(contents: &str) -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(".ply").unwrap();
    file.write_all(contents.as_bytes()).unwrap();
    file
}
