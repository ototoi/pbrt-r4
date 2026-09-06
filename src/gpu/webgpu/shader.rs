const RESOURCES_SHADER: &str = include_str!("shaders/resources.wgsl");
const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const TRIANGLE_SAMPLING_SHADER: &str = include_str!("shaders/triangle_sampling.wgsl");
const LAYERED_SHADER: &str = include_str!("shaders/layered.wgsl");

use std::collections::{HashMap, HashSet};

use super::shader_composer::{compose, ShaderModuleSpec};

pub fn create_module(device: &wgpu::Device, label: &str, stage_source: &str) -> wgpu::ShaderModule {
    let descriptor = wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(compose_source(stage_source))),
    };
    device.create_shader_module(descriptor)
}

#[doc(hidden)]
pub fn compose_source(stage_source: &str) -> String {
    compose_source_with_layered(stage_source, stage_source.contains("layered_"))
}

pub fn compose_source_with_layered(stage_source: &str, include_layered: bool) -> String {
    let common_source = prune_common_source(
        COMMON_SHADER,
        &[stage_source, TRIANGLE_SAMPLING_SHADER, LAYERED_SHADER],
    );
    let resource_source = select_resources(
        RESOURCES_SHADER,
        &format!("{common_source}\n{stage_source}\n{TRIANGLE_SAMPLING_SHADER}\n{LAYERED_SHADER}"),
    );
    // The module graph is built entirely from the literals above. A missing
    // dependency or cycle is therefore a source invariant, not a runtime scene
    // condition; keep the failure explicit while avoiding a generic expect.
    match compose(
        vec![
            ShaderModuleSpec {
                id: "resources".to_string(),
                source: resource_source,
                dependencies: Vec::new(),
            },
            ShaderModuleSpec {
                id: "common".to_string(),
                source: common_source,
                dependencies: vec!["resources".to_string()],
            },
            ShaderModuleSpec {
                id: "triangle_sampling".to_string(),
                source: TRIANGLE_SAMPLING_SHADER.to_string(),
                dependencies: vec!["common".to_string()],
            },
            ShaderModuleSpec {
                id: "layered".to_string(),
                source: LAYERED_SHADER.to_string(),
                dependencies: vec!["common".to_string()],
            },
            ShaderModuleSpec {
                id: "stage".to_string(),
                source: stage_source.to_string(),
                dependencies: if include_layered {
                    vec!["triangle_sampling".to_string(), "layered".to_string()]
                } else {
                    vec!["triangle_sampling".to_string()]
                },
            },
        ],
        "stage",
    ) {
        Ok(composed) => composed.source,
        Err(error) => unreachable!("built-in WebGPU shader module graph is invalid: {error}"),
    }
}

fn prune_common_source(source: &str, roots: &[&str]) -> String {
    let (prefix, functions) = split_functions(source);
    let by_name = functions
        .iter()
        .map(|function| (function.name.as_str(), function))
        .collect::<HashMap<_, _>>();
    let mut selected = HashSet::new();
    let mut pending = function_calls(&roots.join("\n"));
    while let Some(name) = pending.pop() {
        if !selected.insert(name.clone()) {
            continue;
        }
        if let Some(function) = by_name.get(name.as_str()) {
            pending.extend(function_calls(&function.source));
        }
    }
    let mut result = prefix;
    for function in functions {
        if selected.contains(&function.name) {
            result.push_str(&function.source);
            result.push('\n');
        }
    }
    result
}

fn select_resources(source: &str, references: &str) -> String {
    let prefix = source
        .find("@group(0) @binding(")
        .map(|index| &source[..index])
        .unwrap_or("");
    let declarations = source
        .split("@group(0) @binding(")
        .skip(1)
        .filter_map(|block| {
            let block = format!("@group(0) @binding({block}");
            let end = block.find(';')? + 1;
            let declaration = &block[..end];
            let var = if let Some(start) = declaration.find("var<") {
                let name_start = declaration[start..]
                    .find('>')
                    .map(|offset| start + offset + 1)?;
                declaration[name_start..].split_whitespace().next()
            } else {
                let name_start = declaration.find("var ").map(|offset| offset + 4)?;
                declaration[name_start..].split_whitespace().next()
            }?
            .trim_end_matches([':', ';']);
            contains_identifier(references, var).then_some(format!("{declaration}\n"))
        })
        .collect::<String>();
    format!("{prefix}{declarations}")
}

fn contains_identifier(source: &str, identifier: &str) -> bool {
    source.match_indices(identifier).any(|(start, _)| {
        let end = start + identifier.len();
        let boundary = |byte: Option<u8>| {
            byte.is_none_or(|value| !value.is_ascii_alphanumeric() && value != b'_')
        };
        boundary(source.as_bytes().get(start.wrapping_sub(1)).copied())
            && boundary(source.as_bytes().get(end).copied())
    })
}

#[derive(Debug)]
struct FunctionSource {
    name: String,
    source: String,
}

fn split_functions(source: &str) -> (String, Vec<FunctionSource>) {
    let mut prefix = String::new();
    let mut functions = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = source[cursor..].find("\nfn ") {
        let start = cursor + relative + 1;
        prefix.push_str(&source[cursor..start]);
        let name_start = start + 3;
        let name_end = source[name_start..]
            .find('(')
            .map(|offset| name_start + offset)
            .unwrap_or(name_start);
        let name = source[name_start..name_end].trim().to_string();
        let body_start = source[name_end..]
            .find('{')
            .map(|offset| name_end + offset)
            .unwrap_or(source.len());
        let mut depth = 0usize;
        let mut end = body_start;
        for (offset, byte) in source.as_bytes()[body_start..].iter().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = body_start + offset + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        functions.push(FunctionSource {
            name,
            source: source[start..end].to_string(),
        });
        cursor = end;
    }
    prefix.push_str(&source[cursor..]);
    (prefix, functions)
}

fn function_calls(source: &str) -> Vec<String> {
    let bytes = source.as_bytes();
    let mut calls = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index].is_ascii_alphabetic() || bytes[index] == b'_' {
            let start = index;
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            let name = &source[start..index];
            let mut lookahead = index;
            while lookahead < bytes.len() && bytes[lookahead].is_ascii_whitespace() {
                lookahead += 1;
            }
            if lookahead < bytes.len() && bytes[lookahead] == b'(' {
                calls.push(name.to_string());
            }
        } else {
            index += 1;
        }
    }
    calls
}
