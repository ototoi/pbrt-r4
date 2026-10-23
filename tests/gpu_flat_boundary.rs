use std::fs;
use std::path::{Path, PathBuf};

fn flat_data_modules(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if path.file_name().unwrap() != "flatten" {
                flat_data_modules(&path, out);
            }
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            out.push(path);
        }
    }
}

fn module_depth(relative: &Path) -> usize {
    let segments = relative.components().count();
    if relative.file_name().unwrap() == "mod.rs" {
        segments - 1
    } else {
        segments
    }
}

fn strip_line_comments(source: &str) -> String {
    source
        .lines()
        .map(|line| line.split("//").next().unwrap())
        .collect::<Vec<_>>()
        .join("\n")
}

fn path_segments_after(source: &str, index: usize) -> Vec<&str> {
    source[index..]
        .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':' || c == '{'))
        .next()
        .unwrap()
        .split("::")
        .collect()
}

fn violations(source: &str, depth: usize) -> Vec<String> {
    let mut found = Vec::new();
    for (index, _) in source.match_indices("crate::") {
        let segments = path_segments_after(source, index);
        let allowed = matches!(segments.get(1), Some(&"util"))
            || (segments.get(1) == Some(&"gpu") && segments.get(2) == Some(&"flat"));
        if !allowed {
            found.push(segments.join("::"));
        }
    }
    for (index, _) in source.match_indices("super::") {
        if source[..index].ends_with("super::") {
            continue;
        }
        let segments = path_segments_after(source, index);
        let supers = segments.iter().take_while(|s| **s == "super").count();
        let next = segments.get(supers).copied().unwrap_or_default();
        if supers > depth || (supers == depth && next == "flatten") {
            found.push(segments.join("::"));
        }
    }
    for (index, _) in source.match_indices("flat::flatten") {
        found.push(path_segments_after(source, index).join("::"));
    }
    found
}

#[test]
fn flat_data_modules_depend_only_on_flat_and_util() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/flat");
    let mut files = Vec::new();
    flat_data_modules(&root, &mut files);
    files.retain(|path| path != &root.join("mod.rs"));
    files.sort();
    assert!(files.len() > 20);
    let mut failures = Vec::new();
    for path in &files {
        let relative = path.strip_prefix(&root).unwrap();
        let source = strip_line_comments(&fs::read_to_string(path).unwrap());
        for violation in violations(&source, module_depth(relative)) {
            failures.push(format!("{}: {violation}", relative.display()));
        }
    }
    assert!(
        failures.is_empty(),
        "Flat IR data modules may only use crate::gpu::flat and crate::util:\n{}",
        failures.join("\n")
    );
}

#[test]
fn boundary_scan_rejects_outside_dependencies() {
    assert_eq!(
        violations("use crate::paramdict::ParameterDictionary;", 1),
        ["crate::paramdict::ParameterDictionary"]
    );
    assert_eq!(
        violations("use crate::gpu::node::Texture;", 1),
        ["crate::gpu::node::Texture"]
    );
    assert_eq!(violations("use crate::{util, gpu};", 1), ["crate::{util"]);
    assert_eq!(
        violations("use super::super::node::Texture;", 1),
        ["super::super::node::Texture"]
    );
    assert_eq!(
        violations("use super::flatten::flatten_filter;", 1),
        ["super::flatten::flatten_filter"]
    );
    assert_eq!(
        violations("use super::super::flatten::X;", 2),
        ["super::super::flatten::X"]
    );
    assert_eq!(
        violations("use crate::gpu::flat::flatten::X;", 1),
        ["flat::flatten::X"]
    );
    assert!(violations("use crate::util::error::PbrtError;", 1).is_empty());
    assert!(violations("use crate::gpu::flat::INVALID_INDEX;", 2).is_empty());
    assert!(violations("use super::super::texture::Mipmap;", 2).is_empty());
    assert!(violations("values.iter().flatten()", 1).is_empty());
}
