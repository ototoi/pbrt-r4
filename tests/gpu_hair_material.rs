#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::flat::{
    flatten_node, AttributeKind, MaterialKind as FlatMaterialKind, Scene as FlatScene,
};
use pbrt_r4::gpu::node::{prepare_triangle_meshes, tessellate_shapes};
use pbrt_r4::gpu::webgpu::material::{MaterialKind, MaterialTable};
use pbrt_r4::parser::parse_string;
use pbrt_r4::parser::scene_builder::SceneBuilder;

/// Builds a curve + Material "hair" scene through the full GPU IR pipeline
/// (Node IR build, curve dicing, mesh preparation, flattening).
fn hair_flat_scene(material_decl: &str) -> FlatScene {
    let source = format!(
        r#"
Integrator "path"
Sampler "independent" "integer pixelsamples" [1]
Film "rgb" "integer xresolution" [1] "integer yresolution" [1]
Camera "perspective"
WorldBegin
Material "hair" {material_decl}
Shape "curve" "string type" "cylinder" "float width" [0.2]
    "point3 P" [0 0 0 0.3 0 0 0.7 0 0 1 0 0]
WorldEnd
"#
    );
    let mut builder = SceneBuilder::new();
    parse_string(&source, &mut builder).expect("scene should parse");
    let root = builder.build_gpu_ir_node().expect("GPU IR should build");
    {
        let mut node = root.write().expect("node lock");
        tessellate_shapes(&mut node).expect("curves should dice");
        prepare_triangle_meshes(&mut node).expect("meshes should prepare");
    }
    flatten_node(root).expect("flat scene should build")
}

fn hair_material_node(scene: &FlatScene) -> usize {
    scene
        .materials
        .nodes
        .iter()
        .position(|node| node.kind == FlatMaterialKind::Hair)
        .expect("scene should have a hair material node")
}

const HAIR_SCHEMA: [AttributeKind; 9] = [
    AttributeKind::Scalar,
    AttributeKind::Scalar,
    AttributeKind::Scalar,
    AttributeKind::Scalar,
    AttributeKind::Spectrum,
    AttributeKind::Spectrum,
    AttributeKind::Scalar,
    AttributeKind::Scalar,
    AttributeKind::Scalar,
];

fn scalar(scene: &FlatScene, index: usize) -> f32 {
    scene.attributes.scalars[index]
}

fn spectrum_is_zero(scene: &FlatScene, index: usize) -> bool {
    scene.attributes.spectra[index]
        .samples
        .iter()
        .all(|&value| value == 0.0)
}

#[test]
fn hair_defaults_to_sigma_a_of_13_eumelanin_and_uses_the_fixed_schema() {
    let scene = hair_flat_scene("");
    let node = &scene.materials.nodes[hair_material_node(&scene)];
    for (attribute, kind) in node.attributes.iter().zip(HAIR_SCHEMA) {
        assert_eq!(attribute.kind, kind);
    }
    // v4 HairMaterial::Create defaults with nothing given: direct sigma_a
    // mode, beta_m / beta_n 0.3, alpha 2, eta 1.55.
    assert_eq!(scalar(&scene, node.attributes[8].index as usize), 0.0);
    assert!((scalar(&scene, node.attributes[0].index as usize) - 0.3).abs() < 1e-6);
    assert!((scalar(&scene, node.attributes[1].index as usize) - 0.3).abs() < 1e-6);
    assert!((scalar(&scene, node.attributes[2].index as usize) - 2.0).abs() < 1e-6);
    assert!((scalar(&scene, node.attributes[3].index as usize) - 1.55).abs() < 1e-6);
    // The default sigma_a spectrum: RGBUnbounded of 1.3 eumelanin.
    let sigma_a = &scene.attributes.spectra[node.attributes[4].index as usize];
    assert!(sigma_a.samples.iter().any(|&value| value > 0.0));
}

#[test]
fn hair_sigma_a_mode_wins_over_reflectance_and_pigments() {
    let scene = hair_flat_scene(
        "\"spectrum sigma_a\" [0.5 0.5 0.5] \"rgb reflectance\" [0.3 0.2 0.1] \"float eumelanin\" [1.7] \"float pheomelanin\" [0.3]",
    );
    let node = &scene.materials.nodes[hair_material_node(&scene)];
    assert_eq!(scalar(&scene, node.attributes[8].index as usize), 0.0);
    // Lower-priority slots are still present so the schema stays 9 wide and
    // every provided texture stays consumed.
    assert_eq!(node.attributes.len(), 9);
    assert_eq!(node.attributes[5].kind, AttributeKind::Spectrum);
    assert_eq!(node.attributes[6].kind, AttributeKind::Scalar);
}

#[test]
fn hair_reflectance_mode_without_sigma_a() {
    let scene = hair_flat_scene("\"rgb reflectance\" [0.55 0.4 0.2]");
    let node = &scene.materials.nodes[hair_material_node(&scene)];
    assert_eq!(scalar(&scene, node.attributes[8].index as usize), 1.0);
    assert!(spectrum_is_zero(&scene, node.attributes[4].index as usize));
}

#[test]
fn hair_pigment_mode_takes_eumelanin_pheomelanin_scalars_or_textures() {
    let scene = hair_flat_scene("\"float eumelanin\" [2.1] \"float pheomelanin\" [0.4]");
    let node = &scene.materials.nodes[hair_material_node(&scene)];
    assert_eq!(scalar(&scene, node.attributes[8].index as usize), 2.0);
    assert!((scalar(&scene, node.attributes[6].index as usize) - 2.1).abs() < 1e-6);
    assert!((scalar(&scene, node.attributes[7].index as usize) - 0.4).abs() < 1e-6);

    // A texture parameter lands as a Texture attribute in the same slot. Use
    // a real (tiny) image so flatten can build the texture graph.
    let directory = tempfile::tempdir().unwrap();
    let image_path = directory.path().join("density.png");
    image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
        2,
        2,
        vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255],
    )
    .unwrap()
    .save(&image_path)
    .unwrap();
    let source = {
        let path_display = image_path.display();
        format!(
            r#"
Integrator "path"
Sampler "independent" "integer pixelsamples" [1]
Film "rgb" "integer xresolution" [1] "integer yresolution" [1]
Camera "perspective"
WorldBegin
Texture "density" "float" "imagemap" "string filename" "{path_display}"
Material "hair" "texture eumelanin" ["density"]
Shape "curve" "string type" "cylinder" "float width" [0.2]
    "point3 P" [0 0 0 0.3 0 0 0.7 0 0 1 0 0]
WorldEnd
"#
        )
    };
    let mut builder = SceneBuilder::new();
    parse_string(&source, &mut builder).expect("scene should parse");
    let root = builder.build_gpu_ir_node().expect("GPU IR should build");
    {
        let mut node = root.write().expect("node lock");
        tessellate_shapes(&mut node).expect("curves should dice");
        prepare_triangle_meshes(&mut node).expect("meshes should prepare");
    }
    let flat = flatten_node(root).expect("flat scene should build");
    let node = &flat.materials.nodes[hair_material_node(&flat)];
    assert_eq!(node.attributes[6].kind, AttributeKind::Texture);
    assert_eq!(scalar(&flat, node.attributes[8].index as usize), 2.0);
}

#[test]
fn hair_uses_a_fresh_webgpu_material_tag() {
    let scene = hair_flat_scene("");
    let table = MaterialTable::from_flat(&scene).expect("material table should build");
    let index = hair_material_node(&scene);
    assert_eq!(table.nodes[index].kind, MaterialKind::Hair.tag());
    assert_eq!(MaterialKind::Hair.tag(), 16);
}

#[test]
fn hair_on_diced_curves_reference_the_curves_geometry() {
    let scene = hair_flat_scene("");
    assert!(scene
        .geometry
        .geometries
        .iter()
        .any(|geometry| { geometry.source_shape == pbrt_r4::gpu::flat::SourceShape::Curves }));
}

#[test]
fn hair_constant_sigma_a_keeps_unbounded_magnitudes() {
    // HairMaterial::create resolves sigma_a with SpectrumType::Unbounded;
    // an Albedo-typed flattening would clamp "rgb sigma_a" [2 3 4] to
    // magnitudes of at most one and darken the absorption.
    let scene = hair_flat_scene("\"rgb sigma_a\" [2 3 4]");
    let node = &scene.materials.nodes[hair_material_node(&scene)];
    assert_eq!(scalar(&scene, node.attributes[8].index as usize), 0.0);
    let sigma_a = &scene.attributes.spectra[node.attributes[4].index as usize];
    let max_sample = sigma_a
        .samples
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_sample > 1.0,
        "constant sigma_a must stay unbounded, got max sample {max_sample}"
    );
}

#[test]
fn hair_reflectance_wins_over_a_textured_legacy_color() {
    // HairMaterial::create resolves the "reflectance" name first by
    // presence, whether it is a constant or a texture; the legacy "color"
    // only applies when "reflectance" is absent. A constant reflectance
    // must therefore beat a textured "color" for the reflectance slot.
    let directory = tempfile::tempdir().unwrap();
    let image_path = directory.path().join("tint.png");
    image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
        2,
        2,
        vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255],
    )
    .unwrap()
    .save(&image_path)
    .unwrap();
    let source = {
        let path_display = image_path.display();
        format!(
            r#"
Integrator "path"
Sampler "independent" "integer pixelsamples" [1]
Film "rgb" "integer xresolution" [1] "integer yresolution" [1]
Camera "perspective"
WorldBegin
Texture "tint" "spectrum" "imagemap" "string filename" "{path_display}"
Material "hair" "rgb reflectance" [0.4 0.5 0.6] "texture color" ["tint"]
Shape "curve" "string type" "cylinder" "float width" [0.2]
    "point3 P" [0 0 0 0.3 0 0 0.7 0 0 1 0 0]
WorldEnd
"#
        )
    };
    let mut builder = SceneBuilder::new();
    parse_string(&source, &mut builder).expect("scene should parse");
    let root = builder.build_gpu_ir_node().expect("GPU IR should build");
    {
        let mut node = root.write().expect("node lock");
        tessellate_shapes(&mut node).expect("curves should dice");
        prepare_triangle_meshes(&mut node).expect("meshes should prepare");
    }
    let flat = flatten_node(root).expect("flat scene should build");
    let node = &flat.materials.nodes[hair_material_node(&flat)];
    assert_eq!(
        scalar(&flat, node.attributes[8].index as usize),
        1.0,
        "reflectance mode"
    );
    assert_eq!(
        node.attributes[5].kind,
        AttributeKind::Spectrum,
        "constant reflectance must win over the textured legacy color"
    );
}
