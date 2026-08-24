#![cfg(any(feature = "cuda", feature = "webgpu"))]

use pbrt_r4::gpu::compiler::{GpuResourceKind, GpuSourceEntry};
use pbrt_r4::gpu::ir::{GpuGeometry, GpuImageFilter, GpuMaterial, GpuSpectrumResource};
use pbrt_r4::parser::{parse_string, SceneBuilder};

#[test]
fn scene_builder_compiles_a_default_trianglemesh() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
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
    assert_eq!(view.transforms.len(), 2);
    assert_eq!(view.primitives.len(), 1);
    assert_eq!(view.render.film.pixel_bounds.max, [1280, 720]);
    assert_eq!(view.render.sampler.samples_per_pixel, 4);
    assert!(!view.primitives[0].reverse_orientation);
    assert!(matches!(view.geometry[0], GpuGeometry::TriangleMesh(_)));
    let requirements = compiled.requirements();
    assert_eq!(compiled.source_map().locations.len(), 1);
    assert!(compiled.source_map().resources.contains(&GpuSourceEntry {
        kind: GpuResourceKind::Geometry,
        index: 0,
        source: pbrt_r4::gpu::ir::SourceId(0),
    }));
    assert!(compiled.source_map().resources.contains(&GpuSourceEntry {
        kind: GpuResourceKind::Primitive,
        index: 0,
        source: pbrt_r4::gpu::ir::SourceId(0),
    }));
    assert!(compiled.source_map().resources.windows(2).all(|entries| (
        entries[0].kind,
        entries[0].index
    ) < (
        entries[1].kind,
        entries[1].index
    )));
    assert_eq!(requirements.resource_counts.primitives, 1);
    assert_eq!(requirements.maxima.vertices_per_geometry, 3);
    assert!(requirements.features.iter().any(|required| {
        required.feature == pbrt_r4::gpu::ir::GpuFeature::TriangleMesh
            && required.sources.as_ref() == [pbrt_r4::gpu::ir::SourceId(0)]
    }));
    assert!(requirements
        .features
        .windows(2)
        .all(|features| features[0].feature <= features[1].feature));
}

#[test]
fn scene_builder_rejects_path_integrator_without_volpath_fallback() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "path"
            WorldBegin
            Shape "trianglemesh"
                "point3 P" [0 0 0 1 0 0 0 1 0]
                "integer indices" [0 1 2]
            WorldEnd
        "#,
        &mut builder,
    )
    .unwrap();

    let error = builder.build_gpu_ir().unwrap_err();
    assert!(matches!(
        error,
        pbrt_r4::gpu::compiler::GpuSceneBuildError::Compile(
            pbrt_r4::gpu::compiler::GpuCompileError::UnsupportedSceneFeature {
                feature: "non-volpath integrator",
                ..
            }
        )
    ));
}

#[test]
fn scene_builder_preserves_render_settings_and_orientation() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Film "rgb" "integer xresolution" [32] "integer yresolution" [16]
            Sampler "independent" "integer pixelsamples" [7]
            PixelFilter "box"
            Integrator "volpath"
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
    assert_eq!(view.render.film.pixel_bounds.max, [32, 16]);
    assert_eq!(view.render.sampler.samples_per_pixel, 7);
    assert!(view.primitives[0].reverse_orientation);
}

#[test]
fn scene_builder_compiles_constant_diffuse_material() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
            Material "diffuse" "rgb reflectance" [0.2 0.4 0.6]
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
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
    assert_eq!(view.primitives[0].material.map(|id| id.0), Some(1));
    assert!(matches!(
        view.materials[1],
        GpuMaterial::Diffuse(material) if material.reflectance.0 == 1
    ));
    assert!(matches!(
        view.spectra[1],
        GpuSpectrumResource::RgbAlbedo { coefficients }
            if coefficients == [0.2, 0.4, 0.6]
    ));
}

#[test]
fn scene_builder_compiles_instances_without_flattening() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
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

    let compiled = builder.build_gpu_ir().unwrap();
    let view = compiled.view();
    assert_eq!(view.instance_definitions.len(), 1);
    assert_eq!(view.instances.len(), 1);
    assert_eq!(view.world_primitives.len(), 0);
    assert_eq!(view.world_instances.len(), 1);
    assert_eq!(view.instance_definitions[0].primitives.len(), 1);
}

#[test]
fn scene_builder_compiles_point_light() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
            WorldBegin
            LightSource "point" "rgb I" [1 1 1]
            WorldEnd
        "#,
        &mut builder,
    )
    .unwrap();

    let compiled = builder.build_gpu_ir().unwrap();
    assert_eq!(compiled.view().lights.len(), 1);
}

#[test]
fn scene_builder_binds_diffuse_area_light_to_primitive() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
            WorldBegin
            AreaLightSource "diffuse" "rgb L" [2 2 2] "float scale" [3]
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
    assert_eq!(view.lights.len(), 1);
    assert!(matches!(
        view.lights[0],
        pbrt_r4::gpu::ir::GpuLight::DiffuseArea(light)
            if light.scale == 3.0 && light.two_sided == false
    ));
    assert!(matches!(
        view.primitives[0].area_light,
        pbrt_r4::gpu::ir::GpuAreaLightBinding::Uniform(light) if light.0 == 0
    ));
}

#[test]
fn scene_builder_rejects_unsupported_area_light_inputs() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
            WorldBegin
            AreaLightSource "diffuse" "string filename" ["emission.exr"]
            Shape "trianglemesh"
                "point3 P" [0 0 0 1 0 0 0 1 0]
                "integer indices" [0 1 2]
            WorldEnd
        "#,
        &mut builder,
    )
    .unwrap();

    let error = builder.build_gpu_ir().unwrap_err();
    assert!(matches!(
        error,
        pbrt_r4::gpu::compiler::GpuSceneBuildError::Compile(
            pbrt_r4::gpu::compiler::GpuCompileError::UnsupportedSceneFeature {
                feature: "area light image emission",
                ..
            },
        )
    ));
}

#[test]
fn scene_builder_preserves_imagemap_filter_selection() {
    let image_file = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
    image::RgbImage::from_raw(2, 2, vec![128; 12])
        .unwrap()
        .save(image_file.path())
        .unwrap();

    for (filter, expected) in [
        ("point", GpuImageFilter::Point),
        ("bilinear", GpuImageFilter::Bilinear),
        ("trilinear", GpuImageFilter::Trilinear),
        (
            "ewa",
            GpuImageFilter::Ewa {
                max_anisotropy: 4.0,
            },
        ),
    ] {
        let mut builder = SceneBuilder::new();
        let scene = format!(
            r#"
                Texture "albedo" "spectrum" "imagemap"
                    "string filename" ["{}"]
                    "string filter" ["{}"]
                    "float maxanisotropy" [4]
                Film "rgb"
                PixelFilter "box"
                Sampler "independent"
                Integrator "volpath"
                Material "diffuse" "texture reflectance" ["albedo"]
                WorldBegin
                Shape "trianglemesh"
                    "point3 P" [0 0 0 1 0 0 0 1 0]
                    "integer indices" [0 1 2]
                WorldEnd
            "#,
            image_file.path().display(),
            filter
        );
        parse_string(&scene, &mut builder).unwrap();

        let compiled = builder.build_gpu_ir().unwrap();
        let texture = compiled.view().spectrum_textures[1];
        assert!(matches!(
            texture,
            pbrt_r4::gpu::ir::GpuSpectrumTexture::Image { filter, .. } if filter == expected
        ));
        assert!(matches!(
            &compiled.view().images[0].storage,
            pbrt_r4::gpu::ir::GpuTexelStorage::U8(_)
        ));
        assert_eq!(
            compiled.view().images[0].color_encoding,
            pbrt_r4::gpu::ir::GpuColorEncoding::Srgb
        );
    }
}

#[test]
fn scene_builder_uses_luma_channel_for_two_channel_float_imagemap() {
    let image_file = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
    image::GrayAlphaImage::from_raw(1, 1, vec![51, 255])
        .unwrap()
        .save(image_file.path())
        .unwrap();

    let mut builder = SceneBuilder::new();
    let scene = format!(
        r#"
            Texture "alpha" "float" "imagemap" "string filename" ["{}"]
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
            WorldBegin
            Shape "trianglemesh"
                "point3 P" [0 0 0 1 0 0 0 1 0]
                "integer indices" [0 1 2]
                "texture alpha" ["alpha"]
            WorldEnd
        "#,
        image_file.path().display()
    );
    parse_string(&scene, &mut builder).unwrap();

    let compiled = builder.build_gpu_ir().unwrap();
    let view = compiled.view();
    assert!(matches!(
        view.float_textures[0],
        pbrt_r4::gpu::ir::GpuFloatTexture::Image {
            channel: pbrt_r4::gpu::ir::GpuFloatImageChannel::Channel0,
            ..
        }
    ));
    assert_eq!(view.primitives[0].alpha.map(|id| id.0), Some(0));
    assert_eq!(
        view.images[0].channels,
        pbrt_r4::gpu::ir::GpuImageChannels::Rg
    );
}

#[test]
fn scene_builder_compiles_uniform_infinite_light() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
            WorldBegin
            LightSource "infinite" "rgb L" [0.1 0.2 0.3] "float scale" [2]
            WorldEnd
        "#,
        &mut builder,
    )
    .unwrap();

    let compiled = builder.build_gpu_ir().unwrap();
    assert_eq!(compiled.view().lights.len(), 1);
    assert!(matches!(
        compiled.view().lights[0],
        pbrt_r4::gpu::ir::GpuLight::UniformInfinite(light)
            if light.radiance.0 == 1 && light.scale == 2.0
    ));
}

#[test]
fn scene_builder_compiles_constant_spectrum_texture() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Texture "albedo" "spectrum" "constant" "rgb value" [0.2 0.3 0.4]
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
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
    assert_eq!(compiled.view().spectrum_textures.len(), 2);
}

#[test]
fn scene_builder_preserves_material_displacement_reference() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Texture "disp" "float" "constant" "float value" [0.25]
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
            Material "diffuse" "texture displacement" ["disp"]
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
    assert!(matches!(
        compiled.view().materials[1],
        GpuMaterial::Diffuse(material) if material.displacement.is_some()
    ));
}

#[test]
fn scene_builder_compiles_quadric_shapes_without_fallback() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
            WorldBegin
            Shape "sphere" "float radius" [1]
            WorldEnd
        "#,
        &mut builder,
    )
    .unwrap();

    let compiled = builder.build_gpu_ir().unwrap();
    assert!(matches!(
        compiled.view().geometry[0],
        GpuGeometry::Quadric(_)
    ));
}

#[test]
fn scene_builder_compiles_bilinear_mesh() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
            WorldBegin
            Shape "bilinearmesh"
                "point3 P" [0 0 0 1 0 0 1 1 0 0 1 0]
                "integer indices" [0 1 2 3]
            WorldEnd
        "#,
        &mut builder,
    )
    .unwrap();

    let compiled = builder.build_gpu_ir().unwrap();
    assert!(matches!(
        compiled.view().geometry[0],
        GpuGeometry::BilinearPatchMesh(_)
    ));
}

#[test]
fn scene_builder_compiles_curve_mesh() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
            WorldBegin
            Shape "curve" "string type" ["flat"]
                "point3 P" [0 0 0 0 1 0 1 1 0 1 0 0]
                "float width" [0.1 0.05]
            WorldEnd
        "#,
        &mut builder,
    )
    .unwrap();

    let compiled = builder.build_gpu_ir().unwrap();
    assert!(matches!(
        &compiled.view().geometry[0],
        GpuGeometry::CurveMesh(mesh) if mesh.curves.len() == 1
    ));
}

#[test]
fn scene_builder_compiles_triangle_plymesh() {
    let file = tempfile::Builder::new().suffix(".ply").tempfile().unwrap();
    std::fs::write(
        file.path(),
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n",
    )
    .unwrap();

    let mut builder = SceneBuilder::new();
    let scene = format!(
        r#"
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
            WorldBegin
            Shape "plymesh" "string filename" ["{}"]
            WorldEnd
        "#,
        file.path().display()
    );
    parse_string(&scene, &mut builder).unwrap();

    let compiled = builder.build_gpu_ir().unwrap();
    assert!(matches!(
        &compiled.view().geometry[0],
        GpuGeometry::TriangleMesh(mesh)
            if mesh.positions.len() == 3 && mesh.indices == vec![[0, 1, 2]]
    ));
}

#[test]
fn scene_builder_compiles_ply_shape_displacement_into_minmax_ir() {
    let file = tempfile::Builder::new().suffix(".ply").tempfile().unwrap();
    std::fs::write(
        file.path(),
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty float u\nproperty float v\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0 0 0\n1 0 0 1 0\n0 1 0 0 1\n3 0 1 2\n",
    )
    .unwrap();

    let mut builder = SceneBuilder::new();
    let scene = format!(
        r#"
            Texture "disp" "float" "constant" "float value" [0.25]
            Film "rgb"
            PixelFilter "box"
            Sampler "independent"
            Integrator "volpath"
            WorldBegin
            Shape "plymesh" "string filename" ["{}"] "texture displacement" ["disp"]
            WorldEnd
        "#,
        file.path().display()
    );
    parse_string(&scene, &mut builder).unwrap();

    let compiled = builder.build_gpu_ir().unwrap();
    let view = compiled.view();
    assert_eq!(view.geometry.len(), 2);
    assert_eq!(view.primitives[0].geometry.0, 1);
    assert!(matches!(
        &view.geometry[1],
        GpuGeometry::DisplacedTriangleMesh(mesh)
            if mesh.base_mesh.0 == 0
                && mesh.triangle_roots.len() == 1
                && mesh.min_max_nodes[0].displacement_min == 0.25
                && mesh.min_max_nodes[0].displacement_max == 0.25
    ));
}
