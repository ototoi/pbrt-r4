use std::collections::HashMap;
use std::fs;

use pbrt_r4::gpu::flat::{evaluate_dense_spectrum, flatten_node, MediumData};
use pbrt_r4::gpu::node::{Component, TextureComponent};
use pbrt_r4::parser::{parse_file, SceneBuilder};

#[test]
fn gpu_texture_paths_resolve_against_their_own_included_scene() {
    let directory = tempfile::tempdir().expect("temporary directory should be created");
    let first_dir = directory.path().join("first");
    let second_dir = directory.path().join("second");
    std::fs::create_dir(&first_dir).expect("first include directory should be created");
    std::fs::create_dir(&second_dir).expect("second include directory should be created");

    let first_image = first_dir.join("shared.png");
    let second_image = second_dir.join("shared.png");
    std::fs::write(&first_image, "first").expect("first image should be written");
    std::fs::write(&second_image, "second").expect("second image should be written");
    std::fs::write(
        first_dir.join("first.pbrt"),
        "Texture \"firstTexture\" \"spectrum\" \"imagemap\" \"string filename\" [\"shared.png\"]\n",
    )
    .expect("first include should be written");
    std::fs::write(
        second_dir.join("second.pbrt"),
        "Texture \"secondTexture\" \"spectrum\" \"imagemap\" \"string filename\" [\"shared.png\"]\n",
    )
    .expect("second include should be written");
    let root_path = directory.path().join("root.pbrt");
    std::fs::write(
        &root_path,
        "WorldBegin\nInclude \"first/first.pbrt\"\nInclude \"second/second.pbrt\"\nWorldEnd\n",
    )
    .expect("root scene should be written");

    let mut builder = SceneBuilder::new();
    parse_file(root_path.to_str().unwrap(), &mut builder).expect("scene should parse");
    let root = builder
        .build_gpu_ir_node()
        .expect("GPU Node IR should build without decoding the images");
    let root = root.read().unwrap();
    let scene = root
        .components
        .iter()
        .find_map(|component| match component {
            Component::Scene(component) => Some(&component.scene),
            _ => None,
        })
        .expect("root should contain the scene component");
    let image_paths: HashMap<_, _> = scene
        .texture_nodes
        .iter()
        .filter_map(|node| {
            node.components
                .iter()
                .find_map(|component| match component {
                    TextureComponent::Texture(texture) => {
                        texture.image_path().map(|path| (node.name.clone(), path))
                    }
                    TextureComponent::Mapping(_) => None,
                })
        })
        .collect();

    assert_eq!(image_paths.get("firstTexture"), Some(&first_image));
    assert_eq!(image_paths.get("secondTexture"), Some(&second_image));
}

#[test]
fn included_uniformgrid_receives_parent_coefficients_in_flat_ir() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(
        directory.path().join("density.pbrt"),
        "MakeNamedMedium \"smoke\" \"string type\" \"uniformgrid\" \"float density\" [0.25 1] \"integer nx\" [2] \"integer ny\" [1] \"integer nz\" [1]\n",
    ).unwrap();
    let root = directory.path().join("root.pbrt");
    fs::write(
        &root,
        "Camera \"perspective\"\nFilm \"rgb\"\nWorldBegin\nInclude \"density.pbrt\"\n\"spectrum sigma_a\" [300 20 830 20]\n\"spectrum sigma_s\" [300 160 830 160]\nWorldEnd\n",
    ).unwrap();
    let mut builder = SceneBuilder::new();
    parse_file(root.to_str().unwrap(), &mut builder).unwrap();
    let scene = flatten_node(builder.build_gpu_ir_node().unwrap()).unwrap();
    assert_eq!(scene.media.len(), 1);
    let medium = &scene.media[0];
    assert_eq!(medium.name, "smoke");
    assert_eq!(
        evaluate_dense_spectrum(&scene.attributes.spectra, medium.sigma_a, 500.0).unwrap(),
        20.0
    );
    assert_eq!(
        evaluate_dense_spectrum(&scene.attributes.spectra, medium.sigma_s, 500.0).unwrap(),
        160.0
    );
    let MediumData::UniformGrid(grid) = &medium.data else {
        panic!("expected uniformgrid payload");
    };
    assert_eq!(grid.resolution, [2, 1, 1]);
    assert_eq!(grid.density, [0.25, 1.0]);
}
