#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::compiler::{GpuCompileError, GpuSceneBuildError};
use pbrt_r4::gpu::ir::GpuGeometry;
use pbrt_r4::parser::{parse_string, SceneBuilder};

#[test]
fn scene_builder_compiles_a_default_trianglemesh() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            WorldBegin
            Shape "trianglemesh"
                "point3 P" [0 0 0 1 0 0 0 1 0]
                "integer indices" [0 1 2]
            WorldEnd
        "#,
        &mut builder,
    )
    .unwrap();

    let compiled = builder.build_gpu_ir().unwrap();
    let view = compiled.view();
    assert_eq!(view.transforms.len(), 1);
    assert_eq!(view.primitives.len(), 1);
    assert_eq!(view.render.pixel_bounds.max, [1280, 720]);
    assert_eq!(view.render.sample_count, 16);
    assert!(!view.primitives[0].reverse_orientation);
    assert!(matches!(view.geometry[0], GpuGeometry::TriangleMesh(_)));
}

#[test]
fn scene_builder_preserves_render_settings_and_orientation() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Film "rgb" "integer xresolution" [32] "integer yresolution" [16]
            Sampler "independent" "integer pixelsamples" [7]
            ReverseOrientation
            WorldBegin
            Shape "trianglemesh"
                "point3 P" [0 0 0 1 0 0 0 1 0]
                "integer indices" [0 1 2]
            WorldEnd
        "#,
        &mut builder,
    )
    .unwrap();

    let compiled = builder.build_gpu_ir().unwrap();
    let view = compiled.view();
    assert_eq!(view.render.pixel_bounds.max, [32, 16]);
    assert_eq!(view.render.sample_count, 7);
    assert!(view.primitives[0].reverse_orientation);
}

#[test]
fn scene_builder_rejects_instance_shapes_without_fallback() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            WorldBegin
            ObjectBegin "mesh"
            Shape "trianglemesh"
                "point3 P" [0 0 0 1 0 0 0 1 0]
                "integer indices" [0 1 2]
            ObjectEnd
            ObjectInstance "mesh"
            WorldEnd
        "#,
        &mut builder,
    )
    .unwrap();

    assert!(matches!(
        builder.build_gpu_ir(),
        Err(GpuSceneBuildError::Compile(
            GpuCompileError::UnsupportedSceneFeature {
                feature: "instances",
                ..
            },
        ))
    ));
}

#[test]
fn scene_builder_rejects_non_triangle_shapes_without_fallback() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            WorldBegin
            Shape "sphere" "float radius" [1]
            WorldEnd
        "#,
        &mut builder,
    )
    .unwrap();

    assert!(matches!(
        builder.build_gpu_ir(),
        Err(GpuSceneBuildError::Compile(
            GpuCompileError::UnsupportedShape { name, .. },
        )) if name == "sphere"
    ));
}
