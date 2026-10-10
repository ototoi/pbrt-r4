use pbrt_r4::parser::{parse_file, DebugTarget, SceneBuilder};
use pbrt_r4::util::base::Float;
use pbrt_r4::util::spectrum::Spectrum;
use std::fs;

#[test]
fn included_medium_receives_parameters_after_include() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("root.pbrt");
    fs::write(
        directory.path().join("medium.pbrt"),
        "MakeNamedMedium \"smoke\" \"string type\" \"uniformgrid\" \"float density\" [1]\n",
    )
    .unwrap();
    fs::write(
        &root,
        "WorldBegin\nInclude \"medium.pbrt\"\n\"spectrum sigma_a\" [300 20 830 20]\n\"spectrum sigma_s\" [300 160 830 160]\nShape \"sphere\"\n",
    )
    .unwrap();
    let mut builder = SceneBuilder::new();
    parse_file(root.to_str().unwrap(), &mut builder).unwrap();
    let params = &builder.media["smoke"].base.params;
    assert_eq!(
        params
            .get_one_spectrum("sigma_a", &Spectrum::zero())
            .sample_at(500.0),
        20.0
    );
    assert_eq!(
        params
            .get_one_spectrum("sigma_s", &Spectrum::zero())
            .sample_at(500.0),
        160.0
    );
    assert_eq!(builder.shapes.len(), 1);
}

#[test]
fn nested_include_appends_parameters_from_each_ancestor() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("leaf.pbrt"), "Shape \"sphere\"\n").unwrap();
    fs::write(
        directory.path().join("child.pbrt"),
        "Include \"leaf.pbrt\"\n\"float radius\" [2]\n# EOF without newline",
    )
    .unwrap();
    let root = directory.path().join("root.pbrt");
    fs::write(&root, "WorldBegin\nInclude \"child.pbrt\"\n\"float zmin\" [-1]\nShape \"sphere\" \"float radius\" [3]\n").unwrap();
    let mut builder = SceneBuilder::new();
    parse_file(root.to_str().unwrap(), &mut builder).unwrap();
    assert_eq!(builder.shapes.len(), 2);
    assert_eq!(
        builder.shapes[0].base.params.get_one_float("radius", 0.0),
        2.0
    );
    assert_eq!(
        builder.shapes[0].base.params.get_one_float("zmin", 0.0),
        -1.0
    );
    assert_eq!(
        builder.shapes[1].base.params.get_one_float("radius", 0.0),
        3.0
    );
}

#[test]
fn parameter_array_can_continue_after_included_file_eof() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(
        directory.path().join("mesh.pbrt"),
        "Shape \"trianglemesh\" \"point3 P\" [0 0 0 1 0",
    )
    .unwrap();
    let root = directory.path().join("root.pbrt");
    fs::write(&root, "WorldBegin\nInclude \"mesh.pbrt\"\n0 0 1 0]\n\"integer indices\" [0 1 2]\nShape \"sphere\"\n").unwrap();
    let mut builder = SceneBuilder::new();
    parse_file(root.to_str().unwrap(), &mut builder).unwrap();
    assert_eq!(
        builder.shapes[0].base.params.get_points("P"),
        vec![0.0 as Float, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    );
    assert_eq!(
        builder.shapes[0].base.params.get_ints("indices"),
        vec![0, 1, 2]
    );
    assert_eq!(builder.shapes.len(), 2);
}

#[test]
fn continued_shape_retains_its_directory_before_next_parent_shape() {
    let directory = tempfile::tempdir().unwrap();
    let child_dir = directory.path().join("child");
    fs::create_dir(&child_dir).unwrap();
    fs::write(child_dir.join("shared.ply"), "child").unwrap();
    fs::write(directory.path().join("shared.ply"), "parent").unwrap();
    fs::write(
        child_dir.join("mesh.pbrt"),
        "Shape \"plymesh\" \"string filename\" \"shared.ply\"\n",
    )
    .unwrap();
    let root = directory.path().join("root.pbrt");
    fs::write(&root, "WorldBegin\nInclude \"child/mesh.pbrt\"\n\"bool reverse\" [true]\nShape \"plymesh\" \"string filename\" \"shared.ply\"\n").unwrap();
    let mut builder = SceneBuilder::new();
    parse_file(root.to_str().unwrap(), &mut builder).unwrap();
    assert_eq!(
        builder.shapes[0].base.params.get_one_string("filename", ""),
        child_dir.join("shared.ply").to_string_lossy()
    );
    assert_eq!(
        builder.shapes[1].base.params.get_one_string("filename", ""),
        directory.path().join("shared.ply").to_string_lossy()
    );
    assert!(builder.shapes[0].base.params.get_one_bool("reverse", false));
}

#[test]
fn empty_include_does_not_hide_invalid_orphan_parameters() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("empty.pbrt"), "# no directives\n").unwrap();
    let root = directory.path().join("root.pbrt");
    fs::write(&root, "Include \"empty.pbrt\"\n\"float radius\" [1]\n").unwrap();
    let mut target = DebugTarget::new();
    let error = parse_file(root.to_str().unwrap(), &mut target).unwrap_err();
    assert!(error.msg.contains(root.to_str().unwrap()));
    assert!(error.msg.contains("line 2"));
}

#[test]
fn unterminated_string_cannot_continue_into_parent_file() {
    let directory = tempfile::tempdir().unwrap();
    let child = directory.path().join("child.pbrt");
    fs::write(&child, "Shape \"sphere\" \"string note\" \"unterminated").unwrap();
    let root = directory.path().join("root.pbrt");
    fs::write(&root, "Include \"child.pbrt\"\nclosing\"\n").unwrap();
    let mut target = DebugTarget::new();
    let error = parse_file(root.to_str().unwrap(), &mut target).unwrap_err();
    assert!(error.msg.contains(child.to_str().unwrap()));
    assert!(error.msg.contains("operation `Shape`"));
}

#[test]
fn empty_include_preserves_the_next_directive_and_balances_directories() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("empty.pbrt"), "# empty").unwrap();
    let root = directory.path().join("root.pbrt");
    fs::write(&root, "Include \"empty.pbrt\"\nShape \"sphere\"\n").unwrap();
    let mut target = DebugTarget::new();
    parse_file(root.to_str().unwrap(), &mut target).unwrap();
    assert_eq!(target.indent.get(), 0);
    let names: Vec<_> = target
        .operations
        .borrow()
        .iter()
        .map(|op| op.name.clone())
        .collect();
    assert_eq!(
        names,
        [
            "WorkDirBegin",
            "WorkDirBegin",
            "WorkDirEnd",
            "Shape",
            "WorkDirEnd"
        ]
    );
}

#[test]
fn later_parent_error_does_not_prevent_evaluation_of_included_directive() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("child.pbrt"), "Shape \"sphere\"\n").unwrap();
    let root = directory.path().join("root.pbrt");
    fs::write(
        &root,
        "WorldBegin\nInclude \"child.pbrt\"\n\"float radius\" [2]\nShape \"unterminated",
    )
    .unwrap();
    let mut builder = SceneBuilder::new();
    let error = parse_file(root.to_str().unwrap(), &mut builder).unwrap_err();
    assert_eq!(builder.shapes.len(), 1);
    assert_eq!(
        builder.shapes[0].base.params.get_one_float("radius", 0.0),
        2.0
    );
    assert!(error.msg.contains(root.to_str().unwrap()));
    assert!(error.msg.contains("line 4"));
    assert!(builder.work_dirs.is_empty());
}

#[test]
fn sequential_includes_preserve_a_large_parent_directive() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("child.pbrt"), "Shape \"sphere\"\n").unwrap();
    let root = directory.path().join("root.pbrt");
    let mut source = String::from("WorldBegin\n");
    for _ in 0..256 {
        source.push_str("Include \"child.pbrt\"\n\"bool reverse\" [TRUE]\n# Shape \"ignored\"\n");
    }
    source.push_str("Shape \"trianglemesh\" \"point3 P\" [");
    source.push_str(&"0 0 0 ".repeat(100_000));
    source.push_str("] \"integer indices\" [0 1 2]\n");
    fs::write(&root, source).unwrap();
    let mut builder = SceneBuilder::new();
    parse_file(root.to_str().unwrap(), &mut builder).unwrap();
    assert_eq!(builder.shapes.len(), 257);
    for shape in &builder.shapes[..256] {
        assert!(shape.base.params.get_one_bool("reverse", false));
    }
    assert_eq!(
        builder.shapes[256].base.params.get_points("P").len(),
        300_000
    );
}
