#![cfg(feature = "webgpu")]

use std::sync::{Arc, RwLock};

use pbrt_r4::gpu::flat::{self, AlphaMask, INVALID_INDEX};
use pbrt_r4::gpu::node::{
    node_ref_to_json_string, prepare_triangle_meshes, tessellate_shapes, AlphaMask as NodeAlpha,
    AlphaMaskComponent, Component, Node, NodeRef,
};
use pbrt_r4::gpu::webgpu::abi::{ALPHA_MASK_CONSTANT, ALPHA_MASK_OPAQUE, ALPHA_MASK_TEXTURE};
use pbrt_r4::gpu::webgpu::material::MaterialTable;
use pbrt_r4::gpu::webgpu::scene::instance::convert_instances;
use pbrt_r4::parser::parse_string;
use pbrt_r4::parser::scene_builder::SceneBuilder;

const TRIANGLE: &str = r#"Shape "trianglemesh"
    "point3 P" [-1 -1 0  1 -1 0  0 1 0] "integer indices" [0 1 2]"#;

fn node_scene(body: &str) -> NodeRef {
    let source = format!(
        r#"
Camera "perspective"
Film "rgb" "integer xresolution" [8] "integer yresolution" [8]
Sampler "independent" "integer pixelsamples" [1]
Integrator "path"
WorldBegin
{body}
"#
    );
    let mut builder = SceneBuilder::new();
    parse_string(&source, &mut builder).unwrap();
    builder.build_gpu_ir_node().unwrap()
}

fn flatten(root: NodeRef) -> flat::Scene {
    {
        let mut node = root.write().unwrap();
        tessellate_shapes(&mut node).unwrap();
        prepare_triangle_meshes(&mut node).unwrap();
    }
    flat::flatten_node(root).unwrap()
}

#[test]
fn alpha_is_shape_coverage_and_does_not_duplicate_a_shared_bsdf() {
    let root = node_scene(&format!(
        r#"
MakeNamedMaterial "shared" "string type" ["diffuse"]
NamedMaterial "shared"
{TRIANGLE}
{TRIANGLE} "float alpha" [0.5]
{TRIANGLE} "float alpha" [1]
"#
    ));
    let json = node_ref_to_json_string(&root).unwrap();
    assert!(json.contains("AlphaMask"));
    assert!(!json.contains("alphamask"));
    let scene = flatten(root);
    assert_eq!(scene.instances.len(), 3);
    assert_eq!(scene.instances[0].alpha, AlphaMask::Opaque);
    assert_eq!(scene.instances[1].alpha, AlphaMask::Constant(0.5));
    assert_eq!(scene.instances[2].alpha, AlphaMask::Opaque);
    assert_eq!(scene.materials.roots.len(), 1);
    assert_eq!(scene.materials.nodes.len(), 1);
    assert_eq!(scene.materials.nodes[0].kind, flat::MaterialKind::Diffuse);
    assert!(scene
        .instances
        .iter()
        .all(|instance| instance.material_root == 0));
}

#[test]
fn alpha_survives_without_a_surface_material() {
    let scene = flatten(node_scene(&format!(
        r#"
Material "interface"
{TRIANGLE} "float alpha" [0.5]
"#
    )));
    let instance = &scene.instances[0];
    assert_eq!(instance.material_root, INVALID_INDEX);
    assert_eq!(instance.alpha, AlphaMask::Constant(0.5));
    assert!(scene.materials.nodes.is_empty());
    let gpu = convert_instances(&scene, scene.geometry.geometries.len()).unwrap();
    assert_eq!(gpu[0].alpha_kind, ALPHA_MASK_CONSTANT);
    assert_eq!(f32::from_bits(gpu[0].alpha_value), 0.5);
}

#[test]
fn float_alpha_texture_is_separate_from_material_attributes_and_keeps_its_root() {
    let scene = flatten(node_scene(&format!(
        r#"
Texture "mask" "float" "constant" "float value" [0.5]
Material "diffuse"
{TRIANGLE} "texture alpha" ["mask"]
"#
    )));
    let AlphaMask::Texture(root) = scene.instances[0].alpha else {
        panic!("expected texture alpha")
    };
    assert!(matches!(
        scene.texture_library.roots[root as usize],
        flat::texture::TextureRoot::Float { .. }
    ));
    assert_eq!(scene.materials.nodes.len(), 1);
    assert_eq!(scene.materials.nodes[0].attributes.len(), 1);
    assert_eq!(scene.materials.nodes[0].attributes[0].name, "reflectance");
    let gpu = convert_instances(&scene, scene.geometry.geometries.len()).unwrap();
    assert_eq!(
        (gpu[0].alpha_kind, gpu[0].alpha_value),
        (ALPHA_MASK_TEXTURE, root)
    );
}

#[test]
fn constant_zero_emitter_exception_reads_independent_shape_alpha() {
    for (definition, alpha, zero) in [
        ("", "\"float alpha\" [0]", true),
        (
            "Texture \"mask\" \"float\" \"constant\" \"float value\" [0]",
            "\"texture alpha\" [\"mask\"]",
            true,
        ),
        // The constant texture default is one; this used to be an inline regression test.
        (
            "Texture \"mask\" \"float\" \"constant\"",
            "\"texture alpha\" [\"mask\"]",
            false,
        ),
        ("", "\"float alpha\" [0.5]", false),
    ] {
        let scene = flatten(node_scene(&format!(
            r#"
{definition}
AreaLightSource "diffuse" "rgb L" [1 1 1]
Material "diffuse"
{TRIANGLE} {alpha}
"#
        )));
        let flags = scene.lights.sampling_models[0].flags;
        assert_eq!(flat::area_light_is_zero_alpha_sample_only(flags), zero);
        assert_eq!(scene.materials.nodes.len(), 1);
        assert_eq!(scene.materials.nodes[0].kind, flat::MaterialKind::Diffuse);
    }
}

#[test]
fn instance_alpha_conversion_rejects_invalid_values_and_texture_roots() {
    let mut scene = flatten(node_scene(TRIANGLE));
    for alpha in [
        AlphaMask::Constant(f32::NAN),
        AlphaMask::Constant(f32::INFINITY),
        AlphaMask::Texture(INVALID_INDEX),
    ] {
        scene.instances[0].alpha = alpha;
        assert!(convert_instances(&scene, 1)
            .unwrap_err()
            .to_string()
            .contains("alpha"));
    }
    for value in [-0.5, -0.0, 0.0, 0.5, 1.0, 1.5] {
        scene.instances[0].alpha = AlphaMask::Constant(value);
        let gpu = convert_instances(&scene, 1).unwrap();
        assert_eq!(gpu[0].alpha_value, value.to_bits());
    }
    scene.instances[0].alpha = AlphaMask::Opaque;
    assert_eq!(
        convert_instances(&scene, 1).unwrap()[0].alpha_kind,
        ALPHA_MASK_OPAQUE
    );
}

#[test]
fn flatten_rejects_duplicate_alpha_components() {
    // Invalid Node IR is rejected independently of scene-file parsing.
    let root = node_scene(TRIANGLE);
    let mut invalid = Node::new("invalid-alpha");
    invalid.add_component(Component::AlphaMask(AlphaMaskComponent {
        alpha: NodeAlpha::Constant(f32::NAN),
    }));
    invalid.add_component(Component::AlphaMask(AlphaMaskComponent {
        alpha: NodeAlpha::Opaque,
    }));
    root.write()
        .unwrap()
        .add_child(Arc::new(RwLock::new(invalid)));
    let error = flat::flatten_node(root).unwrap_err().to_string();
    assert!(error.contains("at most one AlphaMask"));
}

fn find_shape(root: &NodeRef) -> Option<NodeRef> {
    let node = root.read().unwrap();
    if node
        .components
        .iter()
        .any(|component| matches!(component, Component::Shape(_)))
    {
        return Some(Arc::clone(root));
    }
    node.children.iter().find_map(find_shape)
}

#[test]
fn flatten_rejects_non_finite_shape_alpha() {
    let root = node_scene(TRIANGLE);
    let shape = find_shape(&root).unwrap();
    for component in &mut shape.write().unwrap().components {
        if let Component::AlphaMask(component) = component {
            component.alpha = NodeAlpha::Constant(f32::NAN);
        }
    }
    assert!(flat::flatten_node(root)
        .unwrap_err()
        .to_string()
        .contains("alpha must be finite"));
}

#[test]
fn transformed_instances_share_geometry_and_bsdf_but_preserve_alpha() {
    let root = node_scene(&format!(
        r#"
ObjectBegin "masked"
Material "diffuse"
{TRIANGLE} "float alpha" [0.5]
ObjectEnd
AttributeBegin
Translate 2 0 0
ObjectInstance "masked"
AttributeEnd
AttributeBegin
Rotate 30 0 1 0
Scale 2 3 4
ObjectInstance "masked"
AttributeEnd
"#
    ));
    let scene = flatten(root);
    assert_eq!(scene.instances.len(), 2);
    assert_eq!(scene.instances[0].geometry, scene.instances[1].geometry);
    assert_eq!(
        scene.instances[0].material_root,
        scene.instances[1].material_root
    );
    assert!(scene
        .instances
        .iter()
        .all(|instance| instance.alpha == AlphaMask::Constant(0.5)));
    assert_ne!(scene.instances[0].transform, scene.instances[1].transform);
}

#[test]
fn material_root_validation_stays_on_the_material_path() {
    let mut scene = flatten(node_scene(TRIANGLE));
    scene.materials.roots[0].node_offset = INVALID_INDEX;
    let error = MaterialTable::from_flat(&scene)
        .err()
        .expect("invalid material root must be rejected")
        .to_string();
    assert!(error.contains("Flat material root node is missing."));
}
