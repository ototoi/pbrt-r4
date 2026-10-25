use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};

use super::shader_composer::{compose, ShaderModuleSpec};
use super::stages::{BindingSpec, RequiredLimits};
use crate::util::error::PbrtError;

const RESOURCES_SHADER: &str = include_str!("shaders/resources.wgsl");
const TYPES_SHADER: &str = include_str!("shaders/types.wgsl");
const SPECTRUM_SHADER: &str = include_str!("shaders/spectrum.wgsl");
const TRIANGLE_SAMPLING_SHADER: &str = include_str!("shaders/triangle_sampling.wgsl");
const MEASURED_SHADER: &str = include_str!("shaders/measured.wgsl");
const SAMPLER_SHADER: &str = include_str!("shaders/sampler.wgsl");
const PORTAL_SHADER: &str = include_str!("shaders/portal_image_infinite.wgsl");

const TEXTURE_NOISE_BRANCH_BEGIN: &str = "// TEXTURE_NOISE_BRANCH_BEGIN";
const TEXTURE_NOISE_BRANCH_END: &str = "// TEXTURE_NOISE_BRANCH_END";
const NANOVDB_BRANCH_BEGIN: &str = "// NANOVDB_BRANCH_BEGIN";
const NANOVDB_BRANCH_END: &str = "// NANOVDB_BRANCH_END";

/// Shared WGSL function library, one file per pbrt-v4 source area.
///
/// Stages only receive the functions they transitively call, so the file
/// boundaries are for readers and do not affect the composed modules.
pub const COMMON_LIBRARY: &[(&str, &str)] = &[
    ("filters", include_str!("shaders/lib/filters.wgsl")),
    ("float", include_str!("shaders/lib/float.wgsl")),
    ("vecmath", include_str!("shaders/lib/vecmath.wgsl")),
    ("hash_u32", include_str!("shaders/lib/hash_u32.wgsl")),
    ("hash", include_str!("shaders/lib/hash.wgsl")),
    ("color", include_str!("shaders/lib/color.wgsl")),
    ("path_state", include_str!("shaders/lib/path_state.wgsl")),
    ("work_queues", include_str!("shaders/lib/work_queues.wgsl")),
    ("interaction", include_str!("shaders/lib/interaction.wgsl")),
    ("sparse_grid", include_str!("shaders/lib/sparse_grid.wgsl")),
    ("medium", include_str!("shaders/lib/medium.wgsl")),
    ("blackbody", include_str!("shaders/lib/blackbody.wgsl")),
    ("bssrdf", include_str!("shaders/lib/bssrdf.wgsl")),
    (
        "bssrdf_scattering",
        include_str!("shaders/lib/bssrdf_scattering.wgsl"),
    ),
    ("alpha_mask", include_str!("shaders/lib/alpha_mask.wgsl")),
    ("materials", include_str!("shaders/lib/materials.wgsl")),
    ("textures", include_str!("shaders/lib/textures.wgsl")),
    ("scattering", include_str!("shaders/lib/scattering.wgsl")),
    (
        "bxdfs/dielectric",
        include_str!("shaders/lib/bxdfs/dielectric.wgsl"),
    ),
    (
        "bxdfs/conductor",
        include_str!("shaders/lib/bxdfs/conductor.wgsl"),
    ),
    (
        "bxdfs/layered",
        include_str!("shaders/lib/bxdfs/layered.wgsl"),
    ),
    ("bxdfs/hair", include_str!("shaders/lib/bxdfs/hair.wgsl")),
    ("lights", include_str!("shaders/lib/lights.wgsl")),
    (
        "image_infinite_sampling",
        include_str!("shaders/lib/image_infinite_sampling.wgsl"),
    ),
    (
        "light_samplers",
        include_str!("shaders/lib/light_samplers.wgsl"),
    ),
    (
        "light_sampling",
        include_str!("shaders/lib/light_sampling.wgsl"),
    ),
];

pub fn create_module(device: &wgpu::Device, label: &str, stage_source: &str) -> wgpu::ShaderModule {
    let descriptor = wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(compose_source(stage_source))),
    };
    device.create_shader_module(descriptor)
}

#[doc(hidden)]
pub fn compose_source(stage_source: &str) -> String {
    compose_source_with_noise(stage_source, true)
}

pub fn compose_source_with_noise(stage_source: &str, noise_enabled: bool) -> String {
    compose_source_with_features(stage_source, noise_enabled, true)
}

pub fn compose_source_with_features(
    stage_source: &str,
    noise_enabled: bool,
    nanovdb_enabled: bool,
) -> String {
    let stage_source = if nanovdb_enabled {
        stage_source.to_string()
    } else {
        remove_marked_section(stage_source, NANOVDB_BRANCH_BEGIN, NANOVDB_BRANCH_END)
            .replace("medium.kind > 4u", "medium.kind > 3u")
    };
    let roots = vec![stage_source.as_str(), TRIANGLE_SAMPLING_SHADER];
    let common_source = prune_common_source(&common_input(noise_enabled), &roots);
    let references = format!("{common_source}\n{stage_source}\n{TRIANGLE_SAMPLING_SHADER}");
    let resource_source = select_resources(RESOURCES_SHADER, &references);
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
                id: "stage".to_string(),
                source: stage_source.to_string(),
                dependencies: vec!["triangle_sampling".to_string()],
            },
        ],
        "stage",
    ) {
        Ok(composed) => composed.source,
        Err(error) => unreachable!("built-in WebGPU shader module graph is invalid: {error}"),
    }
}

/// Apply scene capacities without changing the shader's binding declarations.
pub fn specialize_source(
    source: String,
    texture_image_count: u32,
    texture_sampler_count: u32,
    texture_program_capacity: u32,
) -> String {
    source
        .replace(
            "const TEXTURE_PROGRAM_CAPACITY: u32 = 256u;",
            &format!("const TEXTURE_PROGRAM_CAPACITY: u32 = {texture_program_capacity}u;"),
        )
        .replace(
            "binding_array<texture_2d<f32>>",
            &format!("binding_array<texture_2d<f32>, {texture_image_count}u>"),
        )
        .replace(
            "binding_array<sampler>",
            &format!("binding_array<sampler, {texture_sampler_count}u>"),
        )
}

/// Use scalar storage bindings when geometry fits in one page. Buffer binding
/// arrays are not available on every native backend, while ordinary storage
/// buffers are.
pub fn specialize_geometry_page_bindings(source: String, geometry_page_count: u32) -> String {
    if geometry_page_count != 1 {
        return source
            .replace(
                "binding_array<VertexPage>",
                &format!("binding_array<VertexPage, {geometry_page_count}u>"),
            )
            .replace(
                "binding_array<IndexPage>",
                &format!("binding_array<IndexPage, {geometry_page_count}u>"),
            );
    }
    source
        .replace(
            "var<storage, read> vertices: binding_array<VertexPage>;",
            "var<storage, read> vertices: VertexPage;",
        )
        .replace(
            "var<storage, read> indices: binding_array<IndexPage>;",
            "var<storage, read> indices: IndexPage;",
        )
        .replace("vertices[geometry.page].values", "vertices.values")
        .replace("indices[geometry.page].values", "indices.values")
}

/// Concatenates [`COMMON_LIBRARY`]; without noise support the texture noise
/// branch is removed so stages do not require the noise tables.
pub fn common_library_source(noise_enabled: bool) -> String {
    COMMON_LIBRARY
        .iter()
        .map(|(name, source)| {
            if !noise_enabled && *name == "textures" {
                remove_marked_section(source, TEXTURE_NOISE_BRANCH_BEGIN, TEXTURE_NOISE_BRANCH_END)
            } else {
                source.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn common_input(noise_enabled: bool) -> String {
    let library = common_library_source(noise_enabled);
    format!(
        "{TYPES_SHADER}\n{library}\n{SPECTRUM_SHADER}\n{MEASURED_SHADER}\n{SAMPLER_SHADER}\n{PORTAL_SHADER}"
    )
}

fn remove_marked_section(source: &str, begin: &str, end: &str) -> String {
    let mut result = source.to_string();
    loop {
        let Some(begin_offset) = result.find(begin) else {
            return result;
        };
        let Some(relative_end) = result[begin_offset..].find(end) else {
            return result;
        };
        let end_offset = begin_offset + relative_end + end.len();
        result.replace_range(begin_offset..end_offset, "");
    }
}

/// Returns the `(group, binding)` pairs declared by a composed WGSL module.
///
/// The composed source is the ABI for an individual compute stage, so this
/// parser intentionally operates on the generated declarations rather than
/// duplicating a stage-to-resource table in Rust.
pub fn resource_bindings(source: &str) -> Vec<(u32, u32)> {
    let mut bindings = BTreeSet::new();
    let marker = "@group(";
    let mut cursor = 0;
    while let Some(relative) = source[cursor..].find(marker) {
        let group_start = cursor + relative + marker.len();
        let Some(group_end) = source[group_start..].find(')') else {
            break;
        };
        let after_group = group_start + group_end + 1;
        let binding_marker = " @binding(";
        let Some(binding_offset) = source[after_group..].find(binding_marker) else {
            cursor = after_group;
            continue;
        };
        let binding_start = after_group + binding_offset + binding_marker.len();
        let Some(binding_end) = source[binding_start..].find(')') else {
            break;
        };
        if let (Ok(group), Ok(binding)) = (
            source[group_start..group_start + group_end]
                .trim()
                .parse::<u32>(),
            source[binding_start..binding_start + binding_end]
                .trim()
                .parse::<u32>(),
        ) {
            bindings.insert((group, binding));
        }
        cursor = binding_start + binding_end + 1;
    }
    bindings.into_iter().collect()
}

pub fn resource_binding_numbers(source: &str) -> Vec<u32> {
    resource_bindings(source)
        .into_iter()
        .filter_map(|(group, binding)| (group == 0).then_some(binding))
        .collect()
}

/// Resolve the composed shader ABI against the layout registry, rejecting
/// unregistered resources and duplicate registry entries before device creation.
pub fn resolve_resource_bindings(
    canonical_bindings: &[BindingSpec],
    stage_name: &str,
    source: &str,
) -> Result<Vec<BindingSpec>, PbrtError> {
    RequiredLimits::from_bindings(canonical_bindings)
        .map_err(|error| PbrtError::error(&format!("Shader {stage_name}: {error}")))?;
    resource_bindings(source)
        .into_iter()
        .map(|(group, binding)| {
            canonical_bindings
                .iter()
                .find(|candidate| candidate.group == group && candidate.binding == binding)
                .copied()
                .ok_or_else(|| {
                    PbrtError::error(&format!(
                        "Shader {stage_name} uses unregistered group {group} binding {binding}."
                    ))
                })
        })
        .collect()
}

pub fn required_limits_for_sources(
    canonical_bindings: &[BindingSpec],
    stage_sources: &[&str],
) -> Result<RequiredLimits, PbrtError> {
    required_limits_for_sources_with_nanovdb(canonical_bindings, stage_sources, true)
}

pub fn required_limits_for_sources_with_nanovdb(
    canonical_bindings: &[BindingSpec],
    stage_sources: &[&str],
    nanovdb_enabled: bool,
) -> Result<RequiredLimits, PbrtError> {
    let mut required = RequiredLimits::default();
    for stage_source in stage_sources {
        let source = compose_source_with_features(stage_source, true, nanovdb_enabled);
        let bindings = resolve_resource_bindings(canonical_bindings, "anonymous stage", &source)?;
        required = required.maximum(RequiredLimits::from_bindings(&bindings)?);
    }
    Ok(required)
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
    let first_declaration = source.find("@group(").unwrap_or(source.len());
    let prefix = &source[..first_declaration];
    let declarations = source
        .match_indices("@group(")
        .filter_map(|(start, _)| {
            let end = source[start..].find(';')? + start + 1;
            let declaration = &source[start..end];
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
    while cursor < source.len() {
        let start = if source[cursor..].starts_with("fn ") {
            cursor
        } else if let Some(relative) = source[cursor..].find("\nfn ") {
            cursor + relative + 1
        } else {
            break;
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    // prune_common_source resolves calls by name, so a second definition in
    // another library file would silently shadow the first one.
    #[test]
    fn shared_function_names_are_unique() {
        let source = format!("{}\n{TRIANGLE_SAMPLING_SHADER}", common_input(true));
        let mut seen = HashSet::new();
        let duplicates = split_functions(&source)
            .1
            .into_iter()
            .filter(|function| !seen.insert(function.name.clone()))
            .map(|function| function.name)
            .collect::<Vec<_>>();
        assert!(
            duplicates.is_empty(),
            "duplicate WGSL functions: {duplicates:?}"
        );
    }

    #[test]
    fn noise_branch_is_removed_only_from_textures() {
        let with_noise = common_library_source(true);
        let without_noise = common_library_source(false);
        assert!(with_noise.contains(TEXTURE_NOISE_BRANCH_BEGIN));
        assert!(!without_noise.contains(TEXTURE_NOISE_BRANCH_BEGIN));
        assert!(!without_noise.contains(TEXTURE_NOISE_BRANCH_END));
    }

    #[test]
    fn geometry_pages_use_scalar_bindings_for_one_page() {
        let source = "var<storage, read> vertices: binding_array<VertexPage>;\nvar<storage, read> indices: binding_array<IndexPage>;\nlet v = vertices[geometry.page].values[i];\nlet x = arrayLength(&indices[geometry.page].values);".to_string();
        let scalar = specialize_geometry_page_bindings(source.clone(), 1);
        assert!(scalar.contains("var<storage, read> vertices: VertexPage;"));
        assert!(scalar.contains("var<storage, read> indices: IndexPage;"));
        assert!(scalar.contains("vertices.values[i]"));
        assert!(scalar.contains("arrayLength(&indices.values)"));
        assert!(!scalar.contains("binding_array<VertexPage"));

        let paged = specialize_geometry_page_bindings(source, 3);
        assert!(paged.contains("binding_array<VertexPage, 3u>"));
        assert!(paged.contains("binding_array<IndexPage, 3u>"));
    }
}
