use pbrt_r4::paramdict::ParameterDictionary;
use std::fs;
use std::process::Command;

#[test]
fn single_getter_records_multiple_values() {
    let mut params = ParameterDictionary::new();
    params.add_ints("xresolution", &[1, 2]);
    assert_eq!(params.get_one_int("xresolution", 0), 1);
    let errors = params.take_cardinality_errors();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("xresolution"));
    assert!(params.take_cardinality_errors().is_empty());
}

#[test]
fn single_getter_records_empty_array_without_panicking() {
    let mut params = ParameterDictionary::new();
    params.add_ints("empty", &[]);
    assert_eq!(params.get_one_int("empty", 7), 7);
    let errors = params.take_cardinality_errors();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("empty"));
}

#[test]
fn valid_single_and_array_values_are_accepted() {
    let mut params = ParameterDictionary::new();
    params.add_float("scale", 1.0);
    params.add_ints("indices", &[0, 1, 2]);
    assert_eq!(params.get_one_float("scale", 0.0), 1.0);
    assert_eq!(params.get_ints("indices"), vec![0, 1, 2]);
    assert!(params.take_cardinality_errors().is_empty());
}

#[test]
fn report_unused_distinguishes_typed_keys_and_shadowing() {
    let mut params = ParameterDictionary::new();
    params.add_string("key", "value");
    params.add_int("key", 3);
    assert_eq!(params.get_strings("key"), vec!["value".to_string()]);
    assert_eq!(params.get_ints("key"), vec![3]);
    assert!(params.report_unused().is_ok());

    let mut typo = ParameterDictionary::new();
    typo.add_float("roughnes", 0.2);
    let error = typo.report_unused().unwrap_err();
    assert!(error.to_string().contains("roughnes"));

    let mut shadowed = ParameterDictionary::new();
    shadowed.add_float("value", 0.1);
    shadowed.add_float("value", 0.2);
    assert_eq!(shadowed.get_one_float("value", 0.0), 0.1);
    assert!(shadowed.report_unused().is_ok());
}

#[test]
fn may_be_unused_and_clone_keep_state_isolated() {
    let mut params = ParameterDictionary::new();
    params.add_float("inherited", 0.5);
    params.mark_may_be_unused("inherited");
    assert!(params.report_unused().is_ok());

    let clone = params.clone();
    assert!(clone.report_unused().is_ok());

    let other = ParameterDictionary::new();
    assert!(other.report_unused().is_ok());

    let mut unused = ParameterDictionary::new();
    unused.add_float("typo", 0.5);
    assert!(unused.report_unused().is_err());
    assert!(
        params.report_unused().is_ok(),
        "usage leaked across dictionaries"
    );
}

fn render(scene: &str, output: &str) -> std::process::ExitStatus {
    let dir = tempfile::tempdir().unwrap();
    let scene_path = dir.path().join("scene.pbrt");
    fs::write(&scene_path, scene).unwrap();
    Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
        .args(["--outfile", output, scene_path.to_str().unwrap()])
        .output()
        .unwrap()
        .status
}

fn base(material_line: &str, film_line: &str) -> String {
    format!(
        r#"{film_line}
Sampler "independent" "integer pixelsamples" [1]
Integrator "path" "integer maxdepth" [1]
LookAt 0 0 4 0 0 0 0 1 0
Camera "perspective"
WorldBegin
{material_line}
Shape "sphere"
WorldEnd
"#
    )
}

#[test]
fn cli_rejects_multi_value_film_and_typo_material() {
    let dir = tempfile::tempdir().unwrap();
    let multi = base(
        "Material \"diffuse\"",
        "Film \"rgb\" \"integer xresolution\" [1 2] \"integer yresolution\" [1]",
    );
    let output = dir.path().join("multi.exr");
    assert!(!render(&multi, output.to_str().unwrap()).success());

    let typo = base(
        "Material \"diffuse\" \"float roughnes\" [0.2]",
        "Film \"rgb\" \"integer xresolution\" [4] \"integer yresolution\" [4]",
    );
    let output = dir.path().join("typo.exr");
    assert!(!render(&typo, output.to_str().unwrap()).success());

    let valid = base(
        "Material \"diffuse\"",
        "Film \"rgb\" \"integer xresolution\" [4] \"integer yresolution\" [4]",
    );
    let output = dir.path().join("valid.exr");
    assert!(render(&valid, output.to_str().unwrap()).success());

    // A directive-owned value that shadows an inherited Attribute by name
    // must not inherit the mayBeUnused exemption from the Attribute value.
    let overridden = base(
        "Attribute \"material\" \"float roughness\" [0.5]\nMaterial \"diffuse\" \"string roughness\" [\"typo\"]",
        "Film \"rgb\" \"integer xresolution\" [4] \"integer yresolution\" [4]",
    );
    let output = dir.path().join("overridden-typo.exr");
    assert!(!render(&overridden, output.to_str().unwrap()).success());
}

#[test]
fn merge_missing_reports_only_inserted_keys() {
    let mut directive = ParameterDictionary::new();
    directive.add_rgb("roughness", &[0.3, 0.3, 0.3]);
    let mut inherited = ParameterDictionary::new();
    inherited.add_float("roughness", 0.2);
    inherited.add_float("unused_inherited", 0.1);

    let inserted = directive.merge_missing(&inherited);
    assert_eq!(inserted, vec!["unused_inherited".to_string()]);
    for key in inserted {
        directive.mark_may_be_unused(&key);
    }
    // The directive-owned "rgb roughness" shadows the inherited value and
    // must not inherit its mayBeUnused exemption.
    assert!(directive.report_unused().is_err());
}

#[test]
fn surviving_inherited_parameter_keeps_may_be_unused() {
    let mut directive = ParameterDictionary::new();
    let mut inherited = ParameterDictionary::new();
    inherited.add_float("roughness", 0.2);
    let inserted = directive.merge_missing(&inherited);
    assert_eq!(inserted, vec!["roughness".to_string()]);
    for key in inserted {
        directive.mark_may_be_unused(&key);
    }
    assert!(directive.report_unused().is_ok());
}

fn render_with_output(scene: &str, output: &str) -> std::process::Output {
    let dir = tempfile::tempdir().unwrap();
    let scene_path = dir.path().join("scene.pbrt");
    fs::write(&scene_path, scene).unwrap();
    Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
        .args(["--outfile", output, scene_path.to_str().unwrap()])
        .output()
        .unwrap()
}

fn attribute_scene(attribute: &str, material: &str) -> String {
    base(
        &format!("Attribute \"material\" {attribute}\n{material}"),
        "Film \"rgb\" \"integer xresolution\" [4] \"integer yresolution\" [4]",
    )
}

#[test]
fn cli_reports_directive_param_shadowing_unused_attribute() {
    let dir = tempfile::tempdir().unwrap();
    for (attribute, material) in [
        (
            "\"float roughness\" [0.2]",
            "Material \"diffuse\" \"rgb roughness\" [0.3 0.3 0.3]",
        ),
        (
            "\"float roughness\" [0.2]",
            "Material \"diffuse\" \"float roughness\" [0.3]",
        ),
    ] {
        let scene = attribute_scene(attribute, material);
        let output = dir.path().join("shadowed.exr");
        let result = render_with_output(&scene, output.to_str().unwrap());
        assert!(!result.status.success(), "scene:\n{}", scene);
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr.contains("\"roughness\": unused parameter."),
            "stderr: {}",
            stderr
        );
    }
}

#[test]
fn cli_allows_unused_inherited_attribute_without_override() {
    let dir = tempfile::tempdir().unwrap();
    let scene = attribute_scene("\"float roughness\" [0.2]", "Material \"diffuse\"");
    let output = dir.path().join("inherited.exr");
    let result = render_with_output(&scene, output.to_str().unwrap());
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists());
}
