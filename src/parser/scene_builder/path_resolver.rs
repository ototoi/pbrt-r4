//! Resolve relative paths inside a `ParameterDictionary` against the
//! current `work_dirs`. Equivalent logic to SceneBuilder's
//! `make_absolute_path`, exposed as a free function so SceneBuilder
//! can reuse it without depending on SceneBuilder.
//!
//! - `spectrum`-typed string params → read the SPD file and add a
//!   sampled spectrum.
//! - `filename` / `emissionfilename` / `mapname` / `bsdffile` / `lensfile` / `normalmap` string params →
//!   replace with an absolute path.

use crate::paramdict::ParameterDictionary;
use crate::util::spectrum::composite::Spectrum;
use crate::util::spectrum::source::spectrum_from_file;

use std::path::Path;

pub fn make_absolute_path(
    params: &ParameterDictionary,
    work_dirs: &[String],
) -> ParameterDictionary {
    let mut n_params = params.clone();
    let keys = params.get_keys();

    // 1) "spectrum"-typed string params → load the SPD file and turn it
    //    into a sampled spectrum on the dictionary.
    {
        let target_keys: Vec<_> = keys
            .iter()
            .filter(|k| param_type(k) == "spectrum")
            .collect();
        for key in target_keys {
            if let Some(names) = params.get_strings_ref(key) {
                for name in names.iter() {
                    if let Some(path) = resolve_filepath(name, work_dirs) {
                        if let Some(Spectrum::PiecewiseLinear(pls)) = spectrum_from_file(&path) {
                            n_params.add_sampled_spectrum_no_key(name, &pls.lambda, &pls.values);
                        }
                    }
                }
            }
        }
    }

    // 2) filename / emissionfilename / mapname / bsdffile / lensfile / normalmap → replace with an
    //    absolute path.
    {
        let file_path_keys = [
            "filename",
            "emissionfilename",
            "mapname",
            "bsdffile",
            "lensfile",
            "normalmap",
        ];
        let target_keys: Vec<_> = keys
            .iter()
            .filter(|k| {
                let bare = param_key(k);
                file_path_keys.iter().any(|f| *f == bare)
            })
            .collect();
        let mut replaces = Vec::new();
        for key in target_keys {
            if let Some(names) = params.get_strings_ref(key) {
                for name in names.iter() {
                    if let Some(path) = resolve_filepath(name, work_dirs) {
                        replaces.push((key.clone(), path));
                    }
                }
            }
        }
        for (key, value) in replaces.iter() {
            n_params.replace_one_string(key, value);
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

fn param_key(s: &str) -> &str {
    split_type_and_key(s).1
}
