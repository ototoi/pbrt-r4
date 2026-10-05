//! Resolve file parameters before they are stored in a `SceneEntity`.
//!
//! - `spectrum`-typed string params → resolve and load SPD files.
//! - File-valued `string` params with recognized names → replace with
//!   absolute paths.

use crate::paramdict::ParameterDictionary;
use crate::util::spectrum::composite::Spectrum;
use crate::util::spectrum::source::spectrum_from_file;

use std::path::Path;

const STRING_FILE_PATH_KEYS: [&str; 6] = [
    "filename",
    "emissionfilename",
    "mapname",
    "bsdffile",
    "lensfile",
    "normalmap",
];

pub fn resolve_file_paths(
    params: &ParameterDictionary,
    work_dirs: &[String],
) -> ParameterDictionary {
    let n_params = params.clone();
    let keys = params.get_keys();
    for key in &keys {
        let (parameter_type, name) = split_type_and_key(key);
        let is_spectrum_file = parameter_type == "spectrum";
        let is_named_string_path = parameter_type == "string"
            && STRING_FILE_PATH_KEYS
                .iter()
                .any(|path_key| *path_key == name);
        if !is_spectrum_file && !is_named_string_path {
            continue;
        }
        if let Some(names) = params.get_strings_ref(key) {
            if let Some(mut resolved_names) = n_params.get_strings_mut(key) {
                for (index, name) in names.iter().enumerate() {
                    if let Some(path) = resolve_filepath(name, work_dirs) {
                        resolved_names[index] = path;
                    }
                }
            }
        }
    }

    n_params
}

pub fn make_absolute_path(
    params: &ParameterDictionary,
    work_dirs: &[String],
) -> ParameterDictionary {
    let mut n_params = resolve_file_paths(params, work_dirs);
    for key in params
        .get_keys()
        .iter()
        .filter(|key| param_type(key) == "spectrum")
    {
        let names = n_params
            .get_strings_ref(key)
            .map(|names| names.to_vec())
            .unwrap_or_default();
        for name in names {
            if let Some(Spectrum::PiecewiseLinear(pls)) = spectrum_from_file(&name) {
                n_params.add_sampled_spectrum_no_key(&name, &pls.lambda, &pls.values);
            }
        }
    }
    n_params
}

fn resolve_filepath(name: &str, work_dirs: &[String]) -> Option<String> {
    let path = Path::new(name);
    if path.is_absolute() || work_dirs.is_empty() {
        return Some(path.to_string_lossy().to_string());
    }
    for d in work_dirs.iter().rev() {
        let dir = Path::new(d);
        let full = dir.join(name);
        if full.exists() {
            return Some(full.to_string_lossy().to_string());
        }
    }
    None
}

fn split_type_and_key(s: &str) -> (&str, &str) {
    let parts: Vec<&str> = s.split_ascii_whitespace().collect();
    match parts.len() {
        2 => (parts[0], parts[1]),
        1 => ("", parts[0]),
        _ => ("", s),
    }
}

fn param_type(s: &str) -> &str {
    split_type_and_key(s).0
}
