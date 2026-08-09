use pbrt_r4::parser::{
    parse_file_upgraded, parse_string, parse_string_upgraded, PrintTarget, SceneBuilder,
};
use std::cell::RefCell;
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

fn upgrade_output(scene: &str) -> String {
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let writer = Arc::new(RefCell::new(SharedWriter(bytes.clone())));
    let mut target = PrintTarget::new(writer);
    parse_string_upgraded(scene, &mut target).expect("scene should upgrade");
    let output = bytes.lock().unwrap().clone();
    String::from_utf8(output).unwrap()
}

fn upgrade_file_output(path: &str) -> String {
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let writer = Arc::new(RefCell::new(SharedWriter(bytes.clone())));
    let mut target = PrintTarget::new(writer);
    parse_file_upgraded(path, &mut target).expect("file should upgrade");
    let output = bytes.lock().unwrap().clone();
    String::from_utf8(output).unwrap()
}

#[test]
fn upgrades_legacy_film_luminance_parameter() {
    let output = upgrade_output(r#"Film "rgb" "float maxsampleluminance" 7"#);
    assert!(output.contains("maxcomponentvalue"));
    assert!(output.contains("7"));
    assert!(!output.contains("maxsampleluminance"));
}

#[test]
fn upgrades_matte_kd_to_diffuse_reflectance() {
    let output = upgrade_output(r#"Material "matte" "rgb Kd" [ .2 .3 .4 ]"#);
    assert!(output.contains("Material \"diffuse\""));
    assert!(output.contains("reflectance"));
    assert!(output.contains("0.2"));
    assert!(output.contains("0.3"));
    assert!(output.contains("0.4"));
    assert!(!output.contains(" Kd"));
}

#[test]
fn removes_matte_sigma_during_upgrade() {
    let output = upgrade_output(r#"Material "matte" "float sigma" .2"#);
    assert!(!output.contains("sigma"));
}

#[test]
fn rejects_conflicting_film_parameter_names() {
    let mut target = PrintTarget::new_stdout(false);
    let result = parse_string_upgraded(
        r#"Film "rgb" "float maxsampleluminance" 7 "float maxcomponentvalue" 9"#,
        &mut target,
    );
    assert!(result.is_err());
}

#[test]
fn upgrades_zero_specular_plastic_to_diffuse() {
    let output = upgrade_output(
        r#"Material "plastic" "rgb Kd" [ .2 .3 .4 ] "rgb Ks" [ 0 0 0 ] "float eta" 1.5"#,
    );
    assert!(output.contains("Material \"diffuse\""));
    assert!(output.contains("reflectance"));
    assert!(!output.contains(" Ks"));
    assert!(!output.contains(" eta"));
}

#[test]
fn rejects_nonopaque_uber_material() {
    let mut target = PrintTarget::new_stdout(false);
    let result = parse_string_upgraded(r#"Material "uber" "rgb opacity" [ .8 1 1 ]"#, &mut target);
    assert!(result.is_err());
}

#[test]
fn upgrades_glass_index_to_dielectric_eta() {
    let output = upgrade_output(r#"Material "glass" "float index" 1.5"#);
    assert!(output.contains("Material \"dielectric\""));
    assert!(output.contains("eta"));
    assert!(!output.contains(" index"));
}

#[test]
fn upgrades_translucent_kd_to_transmittance() {
    let output = upgrade_output(r#"Material "translucent" "rgb Kd" [ .2 .3 .4 ]"#);
    assert!(output.contains("Material \"diffusetransmission\""));
    assert!(output.contains("transmittance"));
    assert!(!output.contains(" Kd"));
}

#[test]
fn upgrades_material_bumpmap_to_displacement() {
    let output =
        upgrade_output(r#"Material "matte" "texture bumpmap" "bump" "rgb Kd" [ .2 .3 .4 ]"#);
    assert!(output.contains("displacement"));
    assert!(!output.contains("bumpmap"));
}

#[test]
fn upgrades_mirror_to_silver_conductor() {
    let output = upgrade_output(r#"Material "mirror""#);
    assert!(output.contains("Material \"conductor\""));
    assert!(output.contains("metal-Ag-eta"));
    assert!(output.contains("metal-Ag-k"));
    assert!(output.contains("roughness"));
}

#[test]
fn upgrades_mix_amount_and_material_order() {
    let output = upgrade_output(
        r#"Material "mix" "rgb amount" [ .2 .2 .2 ] "string namedmaterial1" "left" "string namedmaterial2" "right""#,
    );
    assert!(output.contains("\"amount\""));
    assert!(output.contains("string materials"));
    assert!(output.contains("right"));
    assert!(output.contains("left"));
    assert!(!output.contains("namedmaterial1"));
}

#[test]
fn rejects_nonconstant_infinite_light_with_mapname() {
    let mut target = PrintTarget::new_stdout(false);
    let result = parse_string_upgraded(
        r#"LightSource "infinite" "rgb L" [ 1 0 0 ] "string mapname" "env.exr""#,
        &mut target,
    );
    assert!(result.is_err());
}

#[test]
fn rejects_legacy_fourier_material() {
    let mut target = PrintTarget::new_stdout(false);
    let result = parse_string_upgraded(r#"Material "fourier""#, &mut target);
    assert!(result.is_err());
}

#[test]
fn upgrade_output_is_idempotent_for_supported_materials() {
    let first = upgrade_output(
        r#"Film "rgb" "float maxsampleluminance" 7
Material "plastic" "rgb Kd" [ .2 .3 .4 ] "rgb Ks" [ 0 0 0 ]"#,
    );
    let second = upgrade_output(&first);
    assert_eq!(first, second);
}

#[test]
fn upgrades_nonmaterial_directive_names_and_parameters() {
    let output = upgrade_output(
        r#"PixelFilter "gaussian" "float xwidth" 2 "float ywidth" 4 "float alpha" .5
Film "rgb" "float scale" 2
Sampler "random"
Integrator "directlighting"
    Camera "environment"
TransformBegin
TransformEnd
Texture "t" "color" "imagemap" "bool trilinear" true"#,
    );
    assert!(output.contains("xradius"));
    assert!(output.contains("yradius"));
    assert!(output.contains("sigma"));
    assert!(output.contains("iso"));
    assert!(output.contains("Sampler \"independent\""));
    assert!(output.contains("Integrator \"path\""));
    assert!(output.contains("maxdepth"));
    assert!(output.contains("Camera \"spherical\""));
    assert!(output.contains("AttributeBegin"));
    assert!(output.contains("AttributeEnd"));
    assert!(output.contains("Texture \"t\" \"spectrum\" \"imagemap\""));
    assert!(output.contains("filter"));
}

#[test]
fn upgrades_spectrum_scale_texture_rgb_component() {
    let output = upgrade_output(
        r#"Texture "scaled" "spectrum" "scale" "rgb tex1" [ 2 2 2 ] "texture tex2" "base""#,
    );
    assert!(output.contains("\"scale\""));
    assert!(output.contains("texture tex"));
    assert!(!output.contains("tex1"));
    assert!(!output.contains("tex2"));
}

#[test]
fn rejects_nonconstant_rgb_scale_texture_component() {
    let mut target = PrintTarget::new_stdout(false);
    let result = parse_string_upgraded(
        r#"Texture "scaled" "spectrum" "scale" "rgb tex1" [ 2 1 2 ] "texture tex2" "base""#,
        &mut target,
    );
    assert!(result.is_err());
}

#[test]
fn upgrades_trianglemesh_uv_parameter_type() {
    let output = upgrade_output(
        r#"Shape "trianglemesh" "float uv" [ 0 0 1 0 1 1 ] "point P" [ 0 0 0 1 0 0 1 1 0 ]"#,
    );
    assert!(output.contains("point2 uv"));
    assert!(!output.contains("float uv"));
}

#[test]
fn removes_redundant_single_triangle_indices() {
    let output = upgrade_output(
        r#"Shape "trianglemesh" "integer indices" [ 0 1 2 ] "point P" [ 0 0 0 1 0 0 1 1 0 ]"#,
    );
    assert!(!output.contains("indices"));
}

#[test]
fn upgrades_light_blackbody_scale_and_mapname() {
    let output = upgrade_output(
        r#"LightSource "infinite" "blackbody L" [ 5000 2 ] "rgb scale" [ 1 1 1 ] "string mapname" "env.exr""#,
    );
    assert!(output.contains("blackbody L"));
    assert!(output.contains("5000"));
    assert!(output.contains("scale"));
    assert!(output.contains("2"));
    assert!(output.contains("filename"));
    assert!(!output.contains("mapname"));
}

#[test]
fn renames_duplicate_texture_definitions() {
    let output = upgrade_output(
        r#"Texture "t" "float" "constant" "float value" 1
Texture "t" "float" "constant" "float value" 2"#,
    );
    assert!(output.contains("Texture \"t\""));
    assert!(output.contains("Texture \"t-renamed-0\""));
}

#[test]
fn renames_duplicate_object_definitions_and_instances() {
    let output = upgrade_output(
        r#"ObjectBegin "o"
ObjectEnd
ObjectBegin "o"
ObjectEnd
ObjectInstance "o""#,
    );
    assert!(output.contains("ObjectBegin \"o\""));
    assert!(output.contains("ObjectBegin \"o-renamed-0\""));
    assert!(output.contains("ObjectInstance \"o-renamed-0\""));
}

#[test]
fn upgrades_directives_inside_include_files() {
    let output = upgrade_file_output("tests/scenes/parser-upgrade-include-main.pbrt");
    assert!(output.contains("maxcomponentvalue"));
    assert!(output.contains("Material \"diffuse\""));
    assert!(output.contains("reflectance"));
}

#[test]
fn upgraded_scene_can_be_reingested_by_scene_builder() {
    let upgraded = upgrade_output(
        r#"WorldBegin
Material "matte" "rgb Kd" [ .2 .3 .4 ]
WorldEnd"#,
    );
    let mut builder = SceneBuilder::new();
    parse_string(&upgraded, &mut builder).expect("upgraded scene should parse normally");
    assert_eq!(builder.materials.len(), 1);
    assert!(builder.materials[0]
        .base
        .params
        .has_parameter("reflectance"));
}
