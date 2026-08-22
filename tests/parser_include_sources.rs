use pbrt_r4::parser::read_file::read_file_with_include_sources;
use pbrt_r4::parser::{parse_file, DebugTarget};
use std::fs;

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
    fs::write(&child, "Translate 1 2 3\n").expect("child scene should be written");
    fs::write(&root, "Identity\nInclude \"child.pbrt\"\nScale 2 2 2\n")
        .expect("root scene should be written");

    let sources = read_file_with_include_sources(root.to_str().unwrap())
        .expect("include expansion should succeed");
    assert!(sources.len() >= 7);
    assert!(sources.iter().all(|source| !source.is_empty()));

    let mut target = DebugTarget::new();
    parse_file(root.to_str().unwrap(), &mut target).expect("scene should parse");
    assert_eq!(
        operation_names(&target),
        [
            "WorkDirBegin",
            "Identitiy",
            "WorkDirBegin",
            "Translate",
            "WorkDirEnd",
            "Scale",
            "WorkDirEnd",
        ]
    );
}
