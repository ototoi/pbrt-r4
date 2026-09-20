use std::path::{Path, PathBuf};

use pbrt_r4::gpu::flat::{MeasuredBsdfLibrary, INVALID_MEASURED_OFFSET};
use pbrt_r4::gpu::node::{Component, Node};
use pbrt_r4::parser::{parse_file, SceneBuilder};
use tempfile::tempdir;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("bsdfs")
        .join(name)
}

#[test]
fn measured_bsdf_library_packs_tables_and_deduplicates_canonical_paths() {
    let path = fixture("paper_white_spec.bsdf");
    let mut library = MeasuredBsdfLibrary::default();
    let first = library.intern(&path).unwrap();
    let second = library
        .intern(&path.parent().unwrap().join("./paper_white_spec.bsdf"))
        .unwrap();
    assert_eq!(first, second);

    let resources = library.finish().unwrap();
    assert_eq!(resources.bsdfs.len(), 1);
    assert_eq!(resources.tables.len(), 5);
    assert_eq!(resources.atlas_pages.len(), 1);
    assert!(resources.atlas_pages[0]
        .texels
        .iter()
        .any(|value| *value != 0.0));

    let record = &resources.bsdfs[0];
    let vndf = &resources.tables[record.vndf as usize];
    assert_eq!(vndf.parameter_count, 2);
    assert_ne!(vndf.marginal_cdf_offset, INVALID_MEASURED_OFFSET);
    assert_ne!(vndf.conditional_cdf_offset, INVALID_MEASURED_OFFSET);
    let spectra = &resources.tables[record.spectra as usize];
    assert_eq!(spectra.parameter_count, 3);
    assert_eq!(spectra.marginal_cdf_offset, INVALID_MEASURED_OFFSET);
}

#[test]
fn gpu_material_path_resolution_uses_the_scene_directory() {
    let directory = tempdir().unwrap();
    let data_directory = directory.path().join("data");
    std::fs::create_dir(&data_directory).unwrap();
    let bsdf = data_directory.join("material.bsdf");
    std::fs::copy(fixture("paper_white_spec.bsdf"), &bsdf).unwrap();
    let scene_path = directory.path().join("scene.pbrt");
    std::fs::write(
        &scene_path,
        r#"
Film "rgb" "integer xresolution" [1] "integer yresolution" [1]
Camera "perspective"
Sampler "independent" "integer pixelsamples" [1]
Integrator "path"
WorldBegin
Material "measured" "string filename" ["data/material.bsdf"]
Shape "trianglemesh" "integer indices" [0 1 2]
    "point3 P" [0 0 0 1 0 0 0 1 0]
"#,
    )
    .unwrap();

    let mut builder = SceneBuilder::new();
    parse_file(scene_path.to_str().unwrap(), &mut builder).unwrap();
    let root = builder.build_gpu_ir_node().unwrap();
    let root = root.read().unwrap();
    let filename = find_measured_filename(&root).expect("measured material should be present");
    assert_eq!(
        Path::new(&filename).canonicalize().unwrap(),
        bsdf.canonicalize().unwrap()
    );
}

fn find_measured_filename(node: &Node) -> Option<String> {
    for component in &node.components {
        if let Component::Material(material) = component {
            if material.material.kind == "measured" {
                return Some(material.material.params.get_one_string("filename", ""));
            }
        }
    }
    node.children.iter().find_map(|child| {
        let child = child.read().unwrap();
        find_measured_filename(&child)
    })
}
