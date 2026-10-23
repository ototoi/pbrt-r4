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

fn split_use_statements(source: &str) -> (Vec<&str>, String) {
    let mut trees = Vec::new();
    let mut rest = String::new();
    let mut cursor = 0;
    while let Some(offset) = source[cursor..].find("use ") {
        let start = cursor + offset;
        let at_word_start = source[..start]
            .chars()
            .last()
            .is_none_or(|c| !(c.is_alphanumeric() || c == '_'));
        if !at_word_start {
            rest.push_str(&source[cursor..start + 4]);
            cursor = start + 4;
            continue;
        }
        let end = source[start..]
            .find(';')
            .map_or(source.len(), |offset| start + offset);
        rest.push_str(&source[cursor..start]);
        trees.push(&source[start + 4..end]);
        cursor = (end + 1).min(source.len());
    }
    rest.push_str(&source[cursor..]);
    (trees, rest)
}

fn split_top_level_commas(items: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    for (index, c) in items.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&items[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    parts.push(&items[start..]);
    parts
}

fn expand_use_tree(tree: &str) -> Vec<String> {
    let tree = tree.split_whitespace().collect::<Vec<_>>().join(" ");
    let Some(open) = tree.find('{') else {
        return vec![tree.split(" as ").next().unwrap().replace(' ', "")];
    };
    let prefix = tree[..open].replace(' ', "");
    let close = tree.rfind('}').unwrap();
    split_top_level_commas(&tree[open + 1..close])
        .into_iter()
        .filter(|item| !item.trim().is_empty())
        .flat_map(expand_use_tree)
        .map(|path| format!("{prefix}{path}"))
        .collect()
}

fn leaves_boundary(path: &str, depth: usize) -> bool {
    let path = path.strip_suffix("::self").unwrap_or(path);
    let segments = path.split("::").collect::<Vec<_>>();
    match segments[0] {
        "crate" => {
            let allowed = segments.get(1) == Some(&"util")
                || (segments.get(1) == Some(&"gpu")
                    && segments.get(2) == Some(&"flat")
                    && segments.get(3) != Some(&"flatten"));
            !allowed
        }
        "super" => {
            let supers = segments.iter().take_while(|s| **s == "super").count();
            let next = segments.get(supers).copied().unwrap_or_default();
            supers > depth || (supers == depth && next == "flatten")
        }
        _ => false,
    }
}

fn inline_path(source: &str, index: usize) -> &str {
    source[index..]
        .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':'))
        .next()
        .unwrap()
}

fn violations(source: &str, depth: usize) -> Vec<String> {
    let (trees, rest) = split_use_statements(source);
    let mut found = trees
        .into_iter()
        .flat_map(expand_use_tree)
        .filter(|path| leaves_boundary(path, depth))
        .collect::<Vec<_>>();
    for prefix in ["crate::", "super::"] {
        for (index, _) in rest.match_indices(prefix) {
            if rest[..index].ends_with("super::") {
                continue;
            }
            let path = inline_path(&rest, index);
            if leaves_boundary(path, depth) {
                found.push(path.to_string());
            }
        }
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
    let cases: &[(&str, usize, &[&str])] = &[
        (
            "use crate::paramdict::ParameterDictionary;",
            1,
            &["crate::paramdict::ParameterDictionary"],
        ),
        ("use crate::gpu::node::Texture;", 1, &["crate::gpu::node::Texture"]),
        ("use crate::{util, gpu};", 1, &["crate::gpu"]),
        (
            "use crate::gpu::{\n    flat::{self, INVALID_INDEX},\n    node::Texture as NodeTexture,\n};",
            1,
            &["crate::gpu::node::Texture"],
        ),
        (
            "use super::super::node::Texture;",
            1,
            &["super::super::node::Texture"],
        ),
        (
            "use super::{super::node::Texture};",
            1,
            &["super::super::node::Texture"],
        ),
        (
            "use super::flatten::flatten_filter;",
            1,
            &["super::flatten::flatten_filter"],
        ),
        (
            "use super::{flatten::flatten_filter};",
            1,
            &["super::flatten::flatten_filter"],
        ),
        (
            "use super::super::flatten::X;",
            2,
            &["super::super::flatten::X"],
        ),
        (
            "use crate::gpu::flat::flatten::X;",
            1,
            &["crate::gpu::flat::flatten::X"],
        ),
        (
            "use crate::gpu::flat::{flatten::X};",
            1,
            &["crate::gpu::flat::flatten::X"],
        ),
        (
            "let p = crate::paramdict::ParameterDictionary::default();",
            1,
            &["crate::paramdict::ParameterDictionary::default"],
        ),
        ("use crate::util::error::PbrtError;", 1, &[]),
        ("use crate::gpu::flat::{self, texture::Mipmap};", 1, &[]),
        ("use crate::gpu::flat::INVALID_INDEX;", 2, &[]),
        ("use super::super::texture::Mipmap;", 2, &[]),
        ("pub use library::{TextureLibrary, TextureRoot};", 1, &[]),
        ("values.iter().flatten()", 1, &[]),
    ];
    for (source, depth, expected) in cases {
        assert_eq!(violations(source, *depth), *expected, "{source}");
    }
}
