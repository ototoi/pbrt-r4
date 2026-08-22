use flate2::write::GzEncoder;
use flate2::Compression;
use pbrt_r4::parser::read_file::read_file_with_include_sources;
use pbrt_r4::parser::{parse_file, parse_string, DebugTarget, PrintTarget, SceneBuilder};
use std::cell::RefCell;
use std::fs;
use std::io::{Result as IoResult, Write};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct SharedWriter(Arc<Mutex<Vec<u8>>>);

impl Write for SharedWriter {
    fn write(&mut self, bytes: &[u8]) -> IoResult<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> IoResult<()> {
        Ok(())
    }
}

fn operation_names(target: &DebugTarget) -> Vec<String> {
    target
        .operations
        .borrow()
        .iter()
        .map(|operation| operation.name.clone())
        .collect()
}

#[test]
fn include_sources_preserve_directive_order_and_work_directories() {
    let directory = tempfile::tempdir().expect("temporary directory should be created");
    let root = directory.path().join("root.pbrt");
    let child = directory.path().join("child.pbrt");
    let grandchild = directory.path().join("grandchild.pbrt");
    fs::write(&grandchild, "Rotate 90 0 1 0\n").expect("grandchild scene should be written");
    fs::write(&child, "Include \"grandchild.pbrt\"\nTranslate 1 2 3\n")
        .expect("child scene should be written");
    fs::write(&root, "Identity\nInclude \"child.pbrt\"\nScale 2 2 2\n")
        .expect("root scene should be written");

    let sources = read_file_with_include_sources(root.to_str().unwrap())
        .expect("include expansion should succeed");
    assert!(sources.len() >= 10);
    assert!(sources.iter().all(|source| !source.is_empty()));

    let mut target = DebugTarget::new();
    parse_file(root.to_str().unwrap(), &mut target).expect("scene should parse");
    let mut identity_target = DebugTarget::new();
    parse_string("Identity", &mut identity_target).expect("identity should parse");
    let identity_name = operation_names(&identity_target)
        .into_iter()
        .next()
        .expect("identity should be recorded");
    assert_eq!(
        operation_names(&target),
        [
            "WorkDirBegin",
            identity_name.as_str(),
            "WorkDirBegin",
            "WorkDirBegin",
            "Rotate",
            "WorkDirEnd",
            "Translate",
            "WorkDirEnd",
            "Scale",
            "WorkDirEnd",
        ]
    );
}

#[test]
fn gzipped_include_is_parsed_in_order() {
    let directory = tempfile::tempdir().expect("temporary directory should be created");
    let root = directory.path().join("root.pbrt.gz");
    let child = directory.path().join("child.pbrt");
    fs::write(&child, "Translate 1 2 3\n").expect("child scene should be written");

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(b"Identity\nInclude \"child.pbrt\"\nScale 2 2 2\n")
        .expect("gzipped scene should be written");
    fs::write(&root, encoder.finish().expect("gzip should finish"))
        .expect("compressed root scene should be written");

    let mut target = DebugTarget::new();
    parse_file(root.to_str().unwrap(), &mut target).expect("gzipped scene should parse");
    let names = operation_names(&target);
    assert!(names.contains(&"Translate".to_string()));
    assert!(names.contains(&"Scale".to_string()));
}

#[test]
fn include_cycle_is_reported_as_an_error() {
    let directory = tempfile::tempdir().expect("temporary directory should be created");
    let first = directory.path().join("first.pbrt");
    let second = directory.path().join("second.pbrt");
    fs::write(&first, "Include \"second.pbrt\"\n").expect("first scene should be written");
    fs::write(&second, "Include \"first.pbrt\"\n").expect("second scene should be written");

    let error = read_file_with_include_sources(first.to_str().unwrap())
        .expect_err("include cycle should be rejected");
    assert!(error.to_string().contains("Include cycle detected"));
}

#[test]
fn include_parse_errors_report_the_current_source_chunk() {
    let directory = tempfile::tempdir().expect("temporary directory should be created");
    let root = directory.path().join("root.pbrt");
    let child = directory.path().join("child.pbrt");
    fs::write(&child, "NotAParserOperation\n").expect("child scene should be written");
    fs::write(&root, "Identity\nInclude \"child.pbrt\"\n").expect("root scene should be written");

    let mut target = DebugTarget::new();
    let error = parse_file(root.to_str().unwrap(), &mut target)
        .expect_err("invalid child operation should be rejected");
    assert!(error.msg.contains("line 1"));
    assert!(error.msg.contains("operation `NotAParserOperation`"));
}

#[test]
fn include_result_reaches_scene_builder() {
    let directory = tempfile::tempdir().expect("temporary directory should be created");
    let root = directory.path().join("root.pbrt");
    let child = directory.path().join("child.pbrt");
    fs::write(
        &child,
        "Material \"diffuse\" \"rgb reflectance\" [.5 .5 .5]\n\
         Shape \"sphere\" \"float radius\" [1]\n",
    )
    .expect("child scene should be written");
    fs::write(&root, "WorldBegin\nInclude \"child.pbrt\"\nWorldEnd\n")
        .expect("root scene should be written");

    let mut builder = SceneBuilder::new();
    parse_file(root.to_str().unwrap(), &mut builder).expect("scene should parse");
    assert_eq!(builder.materials.len(), 1);
    assert_eq!(builder.shapes.len(), 1);
}

#[test]
fn include_result_reaches_print_target() {
    let directory = tempfile::tempdir().expect("temporary directory should be created");
    let root = directory.path().join("root.pbrt");
    let child = directory.path().join("child.pbrt");
    fs::write(&child, "Translate 1 2 3\n").expect("child scene should be written");
    fs::write(&root, "Identity\nInclude \"child.pbrt\"\n").expect("root scene should be written");

    let bytes = Arc::new(Mutex::new(Vec::new()));
    let writer = Arc::new(RefCell::new(SharedWriter(bytes.clone())));
    let mut target = PrintTarget::new(writer);
    parse_file(root.to_str().unwrap(), &mut target).expect("scene should parse");
    let output = String::from_utf8(bytes.lock().unwrap().clone()).expect("output should be UTF-8");
    assert!(output.contains("Identity"));
    assert!(output.contains("Translate 1 2 3"));
}
