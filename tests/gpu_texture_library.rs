use std::sync::Arc;

use image::{ImageBuffer, Luma};
use pbrt_r4::gpu::flat::texture::{
    compile_texture_library, evaluate_texture_root, evaluate_texture_root_at,
    evaluate_texture_root_with_context, ColorSpace, ImageCompiler, ImageDecoder, ImageFilterMode,
    ImageOptimizationPolicy, ImageValueType, ImageWrapMode, Mipmap, MipmapEncoding, MipmapLevel,
    MipmapLevelData, TextureEvaluationContext, TextureInstruction, TextureRoot, TextureRootSpec,
    TextureValue, TextureValueType,
};
use pbrt_r4::gpu::node::TextureNode;
use pbrt_r4::gpu::node::{
    Texture, TextureComponent, TextureKind, TextureMapping, Transform, UvMapping,
};
use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::util::base::inverse_gamma_correct;
use pbrt_r4::util::spectrum::{Spectrum, SpectrumType};
use tempfile::tempdir;

#[test]
fn texture_library_keeps_root_interpretation_outside_programs() {
    let mut node = TextureNode::new("constant");
    node.components.push(TextureComponent::Texture(Texture {
        name: "constant".to_string(),
        kind: TextureKind::Float,
        params: ParameterDictionary::default(),
    }));
    let node = Arc::new(node);
    let roots = [
        TextureRootSpec::Float { node: node.clone() },
        TextureRootSpec::Spectrum {
            node,
            spectrum_type: SpectrumType::Unbounded,
        },
    ];

    let library = compile_texture_library(&roots).unwrap();
    assert_eq!(library.programs.len(), 1);
    assert!(library.image_views.is_empty());
    assert_eq!(library.roots.len(), 2);

    assert!(matches!(
        library.roots[0],
        TextureRoot::Float { program: 0 }
    ));
    assert!(matches!(
        library.roots[1],
        TextureRoot::Spectrum {
            program: 0,
            spectrum_type: SpectrumType::Unbounded,
        }
    ));
}

#[test]
fn image_compiler_projects_float_images_and_applies_f16_policy() {
    let source = Arc::new(Mipmap {
        levels: vec![MipmapLevel {
            resolution: [2, 1],
            channels: 3,
            data: MipmapLevelData::F32(vec![0.0, 0.3, 0.6, 0.5, 0.5, 0.5]),
        }],
        color_space: ColorSpace::Srgb,
        encoding: MipmapEncoding::Linear,
    });
    let mut compiler = ImageCompiler::new(ImageOptimizationPolicy {
        allow_f16: true,
        max_absolute_error: 0.001,
        max_relative_error: 0.001,
    });

    let projected = compiler.compile(&source, ImageValueType::Float).unwrap();
    assert_eq!(projected.levels[0].channels, 1);
    assert!(matches!(projected.levels[0].data, MipmapLevelData::F16(_)));
}

#[test]
fn image_compiler_projects_luminance_alpha_to_linear_rgb() {
    let source = Arc::new(Mipmap {
        levels: vec![MipmapLevel {
            resolution: [2, 1],
            channels: 2,
            data: MipmapLevelData::U8(vec![64, 10, 128, 20]),
        }],
        color_space: ColorSpace::Srgb,
        encoding: MipmapEncoding::U8Normalized,
    });
    let mut compiler = ImageCompiler::new(ImageOptimizationPolicy {
        allow_f16: false,
        max_absolute_error: 0.0,
        max_relative_error: 0.0,
    });

    let projected = compiler
        .compile(&source, ImageValueType::LinearRgb)
        .unwrap();
    assert_eq!(projected.levels[0].channels, 3);
    assert_eq!(projected.encoding, MipmapEncoding::Linear);
    assert_eq!(
        projected.levels[0].data,
        MipmapLevelData::F32(
            vec![64.0 / 255.0; 3]
                .into_iter()
                .chain(vec![128.0 / 255.0; 3])
                .collect()
        )
    );
}

#[test]
fn image_compiler_discards_alpha_and_linearizes_srgb_spectrum() {
    let source = Arc::new(Mipmap {
        levels: vec![MipmapLevel {
            resolution: [1, 1],
            channels: 4,
            data: MipmapLevelData::F32(vec![0.25, 0.5, 0.75, 0.125]),
        }],
        color_space: ColorSpace::Srgb,
        encoding: MipmapEncoding::SrgbEncoded,
    });
    let mut compiler = ImageCompiler::new(ImageOptimizationPolicy {
        allow_f16: false,
        max_absolute_error: 0.0,
        max_relative_error: 0.0,
    });

    let projected = compiler
        .compile(&source, ImageValueType::LinearRgb)
        .unwrap();
    assert_eq!(projected.levels[0].channels, 3);
    assert_eq!(projected.encoding, MipmapEncoding::Linear);
    assert_eq!(
        projected.levels[0].data,
        MipmapLevelData::F32(vec![
            inverse_gamma_correct(0.25),
            inverse_gamma_correct(0.5),
            inverse_gamma_correct(0.75),
        ])
    );
}

#[test]
fn default_image_compiler_uses_bounded_f16_storage() {
    let source = Arc::new(Mipmap {
        levels: vec![MipmapLevel {
            resolution: [1, 1],
            channels: 1,
            data: MipmapLevelData::F32(vec![0.25]),
        }],
        color_space: ColorSpace::Unknown,
        encoding: MipmapEncoding::Linear,
    });
    let mut compiler = ImageCompiler::default();
    let compiled = compiler
        .compile(&source, ImageValueType::LinearRgb)
        .unwrap();
    assert!(matches!(compiled.levels[0].data, MipmapLevelData::F16(_)));
}

#[test]
fn image_decoder_interns_canonical_file_identity() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("image.png");
    ImageBuffer::<Luma<u8>, _>::from_raw(1, 1, vec![128])
        .unwrap()
        .save(&path)
        .unwrap();
    let alternate = directory.path().join(".").join("image.png");

    let mut decoder = ImageDecoder::default();
    let first = decoder.decode(&path, "linear").unwrap();
    let second = decoder.decode(&alternate, "linear").unwrap();

    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn image_compiler_chooses_one_storage_format_for_the_full_mipmap() {
    let source = Arc::new(Mipmap {
        levels: vec![
            MipmapLevel {
                resolution: [2, 1],
                channels: 1,
                data: MipmapLevelData::F32(vec![0.25, 0.5]),
            },
            MipmapLevel {
                resolution: [1, 1],
                channels: 1,
                data: MipmapLevelData::F32(vec![100_000.0]),
            },
        ],
        color_space: ColorSpace::Unknown,
        encoding: MipmapEncoding::Linear,
    });
    let compiled = ImageCompiler::default()
        .compile(&source, ImageValueType::LinearRgb)
        .unwrap();
    assert!(compiled
        .levels
        .iter()
        .all(|level| matches!(level.data, MipmapLevelData::F32(_))));
}

#[test]
fn image_compiler_rejects_inconsistent_mipmap_storage() {
    let source = Arc::new(Mipmap {
        levels: vec![
            MipmapLevel {
                resolution: [2, 1],
                channels: 1,
                data: MipmapLevelData::F32(vec![0.25, 0.5]),
            },
            MipmapLevel {
                resolution: [1, 1],
                channels: 1,
                data: MipmapLevelData::F16(vec![half::f16::from_f32(0.5).to_bits()]),
            },
        ],
        color_space: ColorSpace::Unknown,
        encoding: MipmapEncoding::Linear,
    });
    let error = ImageCompiler::default()
        .compile(&source, ImageValueType::LinearRgb)
        .unwrap_err();
    assert!(error.to_string().contains("inconsistent storage formats"));
}

#[test]
fn texture_program_is_typed_post_order() {
    let mut child = TextureNode::new("constant");
    child.components.push(TextureComponent::Texture(Texture {
        name: "constant".to_string(),
        kind: TextureKind::Float,
        params: ParameterDictionary::default(),
    }));

    let mut root = TextureNode::new("scale");
    let mut params = ParameterDictionary::default();
    params.add_float("float scale", 2.0);
    root.components.push(TextureComponent::Texture(Texture {
        name: "scale".to_string(),
        kind: TextureKind::Float,
        params,
    }));
    root.children.push(Arc::new(child));

    let library = compile_texture_library(&[TextureRootSpec::Float {
        node: Arc::new(root),
    }])
    .unwrap();
    let program = &library.programs[0];

    assert_eq!(program.result, 0);
    assert_eq!(program.slot_types, vec![TextureValueType::Float]);
    assert_eq!(program.slot_last_use, vec![0]);
    assert!(matches!(
        program.instructions[0],
        TextureInstruction::ConstantFloat { dst: 0, value: 0.0 }
    ));
    assert_eq!(
        evaluate_texture_root(&library, 0).unwrap(),
        TextureValue::Float(0.0)
    );
}

#[test]
fn texture_program_evaluates_a_dynamic_float_scale_for_spectrum() {
    let mut root = TextureNode::new("dynamic-spectrum-scale");
    root.components.push(TextureComponent::Texture(Texture {
        name: "scale".to_string(),
        kind: TextureKind::Spectrum,
        params: ParameterDictionary::default(),
    }));
    root.children
        .push(spectrum_constant("base", [0.2, 0.4, 0.6]));
    root.children.push(float_bilerp("factor", 0.5));

    let library = compile_texture_library(&[TextureRootSpec::Spectrum {
        node: Arc::new(root),
        spectrum_type: SpectrumType::Albedo,
    }])
    .unwrap();
    let program = &library.programs[0];

    assert!(program.instructions.iter().any(|instruction| matches!(
        instruction,
        TextureInstruction::Scale { scale: Some(_), .. }
    )));
    let base = program
        .instructions
        .iter()
        .find_map(|instruction| match instruction {
            TextureInstruction::ConstantRgb { value, .. } => Some(*value),
            _ => None,
        })
        .expect("spectrum input should remain in the program");
    assert_eq!(
        evaluate_texture_root_at(&library, 0, [0.25, 0.75]).unwrap(),
        TextureValue::LinearRgb(base.map(|value| value * 0.5))
    );
}

#[test]
fn dynamic_zero_scale_returns_typed_zero_without_nan_propagation() {
    let mut root = TextureNode::new("zero-spectrum-scale");
    root.components.push(TextureComponent::Texture(Texture {
        name: "scale".to_string(),
        kind: TextureKind::Spectrum,
        params: ParameterDictionary::default(),
    }));
    root.children
        .push(spectrum_constant("nan-base", [f32::NAN; 3]));
    root.children.push(float_bilerp("zero-factor", 0.0));

    let library = compile_texture_library(&[TextureRootSpec::Spectrum {
        node: Arc::new(root),
        spectrum_type: SpectrumType::Albedo,
    }])
    .unwrap();

    assert_eq!(
        evaluate_texture_root(&library, 0).unwrap(),
        TextureValue::LinearRgb([0.0; 3])
    );
}

#[test]
fn image_instruction_keeps_sampling_interpretation() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("value.png");
    ImageBuffer::<Luma<u8>, _>::from_raw(1, 1, vec![64])
        .unwrap()
        .save(&path)
        .unwrap();
    let mut params = ParameterDictionary::default();
    params.add_string("string wrap", "black");
    params.add_string("string filter", "nearest");
    params.add_float("float scale", 3.0);
    params.add_bool("bool invert", true);
    params.add_string("string filename", &path.to_string_lossy());
    let mut node = TextureNode::new("imagemap");
    node.components.push(TextureComponent::Texture(Texture {
        name: "imagemap".to_string(),
        kind: TextureKind::Float,
        params,
    }));
    node.components
        .push(TextureComponent::Mapping(TextureMapping::Uv(UvMapping {
            uscale: 2.0,
            vscale: 3.0,
            udelta: 0.1,
            vdelta: 0.2,
        })));

    let library = compile_texture_library(&[TextureRootSpec::Float {
        node: Arc::new(node),
    }])
    .unwrap();
    assert_eq!(library.mipmaps.len(), 1);
    assert_eq!(library.image_views.len(), 1);
    let TextureInstruction::SampleImage { image_view, .. } = &library.programs[0].instructions[0]
    else {
        panic!("expected image sample instruction");
    };
    let view = &library.image_views[*image_view as usize];
    assert_eq!(view.mipmap, 0);
    assert_eq!(view.swrap, ImageWrapMode::Black);
    assert_eq!(view.filter, ImageFilterMode::Nearest);
    assert_eq!(view.scale, 3.0);
    assert!(view.invert);
    assert!(matches!(
        &library.programs[0].instructions[0],
        TextureInstruction::SampleImage {
            mapping: Some(TextureMapping::Uv(UvMapping { uscale: 2.0, .. })),
            ..
        }
    ));
    let TextureValue::Float(value) = evaluate_texture_root_at(&library, 0, [0.2, 0.2]).unwrap()
    else {
        panic!("expected float texture value");
    };
    let expected = 3.0 * (1.0 - inverse_gamma_correct(64.0 / 255.0) as f32);
    assert!((value - expected).abs() < 1e-6);
}

#[test]
fn image_views_are_shared_across_distinct_programs() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("shared.png");
    ImageBuffer::<Luma<u8>, _>::from_raw(1, 1, vec![128])
        .unwrap()
        .save(&path)
        .unwrap();
    let make_node = |name: &str| {
        let mut params = ParameterDictionary::default();
        params.add_string("string filename", &path.to_string_lossy());
        let mut node = TextureNode::new(name);
        node.components.push(TextureComponent::Texture(Texture {
            name: "imagemap".to_string(),
            kind: TextureKind::Float,
            params,
        }));
        Arc::new(node)
    };
    let library = compile_texture_library(&[
        TextureRootSpec::Float {
            node: make_node("first"),
        },
        TextureRootSpec::Float {
            node: make_node("second"),
        },
    ])
    .unwrap();
    assert_eq!(library.programs.len(), 1);
    assert_eq!(library.mipmaps.len(), 1);
    assert_eq!(library.image_views.len(), 1);
    assert_eq!(library.image_views[0].mipmap, 0);
    for program in &library.programs {
        assert!(matches!(
            program.instructions[0],
            TextureInstruction::SampleImage { image_view: 0, .. }
        ));
    }
}

#[test]
fn constant_folding_does_not_mutate_a_shared_child() {
    let shared = float_constant("shared", 2.0);

    let mut scale_params = ParameterDictionary::default();
    scale_params.add_float("float scale", 3.0);
    let mut scaled = TextureNode::new("scaled");
    scaled.components.push(TextureComponent::Texture(Texture {
        name: "scale".to_string(),
        kind: TextureKind::Float,
        params: scale_params,
    }));
    scaled.children.push(shared.clone());

    let mut mix_params = ParameterDictionary::default();
    mix_params.add_float("float amount", 0.5);
    let mut root = TextureNode::new("root");
    root.components.push(TextureComponent::Texture(Texture {
        name: "mix".to_string(),
        kind: TextureKind::Float,
        params: mix_params,
    }));
    root.children.push(shared);
    root.children.push(Arc::new(scaled));

    let library = compile_texture_library(&[TextureRootSpec::Float {
        node: Arc::new(root),
    }])
    .unwrap();
    let program = &library.programs[0];
    assert_eq!(program.instructions.len(), 1);
    assert!(matches!(
        program.instructions[0],
        TextureInstruction::ConstantFloat { dst: 0, value: 4.0 }
    ));
    assert_eq!(
        evaluate_texture_root(&library, 0).unwrap(),
        TextureValue::Float(4.0)
    );
}

#[test]
fn texture_program_rejects_mixed_operand_types() {
    let mut spectrum_params = ParameterDictionary::default();
    spectrum_params.add_spectrum("spectrum value", &Spectrum::from(1.0));
    let mut spectrum = TextureNode::new("spectrum");
    spectrum.components.push(TextureComponent::Texture(Texture {
        name: "constant".to_string(),
        kind: TextureKind::Spectrum,
        params: spectrum_params,
    }));

    let mut root = TextureNode::new("invalid-mix");
    root.components.push(TextureComponent::Texture(Texture {
        name: "mix".to_string(),
        kind: TextureKind::Float,
        params: ParameterDictionary::default(),
    }));
    root.children.push(float_constant("float", 0.0));
    root.children.push(Arc::new(spectrum));

    let error = compile_texture_library(&[TextureRootSpec::Float {
        node: Arc::new(root),
    }])
    .unwrap_err();
    assert!(error.to_string().contains("incompatible value types"));
}

#[test]
fn texture_program_shares_structurally_equal_instructions() {
    let mut root = TextureNode::new("checkerboard");
    root.components.push(TextureComponent::Texture(Texture {
        name: "checkerboard".to_string(),
        kind: TextureKind::Float,
        params: ParameterDictionary::default(),
    }));
    root.children.push(float_constant("first", 0.5));
    root.children.push(float_constant("second", 0.5));

    let library = compile_texture_library(&[TextureRootSpec::Float {
        node: Arc::new(root),
    }])
    .unwrap();
    let program = &library.programs[0];
    assert_eq!(program.instructions.len(), 2);
    assert!(matches!(
        &program.instructions[1],
        TextureInstruction::Procedural { operands, .. } if operands == &[0, 0]
    ));
}

#[test]
fn texture_program_cse_preserves_signed_zero() {
    let mut root = TextureNode::new("checkerboard");
    root.components.push(TextureComponent::Texture(Texture {
        name: "checkerboard".to_string(),
        kind: TextureKind::Float,
        params: ParameterDictionary::default(),
    }));
    root.children.push(float_constant("positive-zero", 0.0));
    root.children.push(float_constant("negative-zero", -0.0));

    let library = compile_texture_library(&[TextureRootSpec::Float {
        node: Arc::new(root),
    }])
    .unwrap();
    let program = &library.programs[0];
    assert_eq!(program.instructions.len(), 3);
    assert!(matches!(
        &program.instructions[2],
        TextureInstruction::Procedural { operands, .. } if operands == &[0, 1]
    ));
}

#[test]
fn reference_vm_evaluates_bilerp_in_post_order() {
    let mut root = TextureNode::new("bilerp");
    root.components.push(TextureComponent::Texture(Texture {
        name: "bilerp".to_string(),
        kind: TextureKind::Float,
        params: ParameterDictionary::default(),
    }));
    root.components
        .push(TextureComponent::Mapping(TextureMapping::Uv(UvMapping {
            uscale: 1.0,
            vscale: 1.0,
            udelta: 0.0,
            vdelta: 0.0,
        })));
    root.children.push(float_constant("v00", 0.0));
    root.children.push(float_constant("v01", 1.0));
    root.children.push(float_constant("v10", 2.0));
    root.children.push(float_constant("v11", 3.0));

    let library = compile_texture_library(&[TextureRootSpec::Float {
        node: Arc::new(root),
    }])
    .unwrap();
    let value = evaluate_texture_root_with_context(
        &library,
        0,
        TextureEvaluationContext {
            uv: [0.25, 0.75],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(value, TextureValue::Float(1.25));
}

#[test]
fn reference_vm_uses_v4_spherical_mapping_coordinates() {
    let value = evaluate_mapped_bilerp(
        TextureMapping::Spherical(Transform::default()),
        [0.0, 1.0, 0.0],
    );

    // v4 maps +Y to (theta / pi, phi / 2pi) = (0.5, 0.25).
    assert!((value - 1.25).abs() < 1e-6);
}

#[test]
fn reference_vm_uses_v4_cylindrical_mapping_coordinates() {
    let value = evaluate_mapped_bilerp(
        TextureMapping::Cylindrical(Transform::default()),
        [1.0, 0.0, 0.25],
    );

    // v4 maps +X to s=0.5 and preserves z as t.
    assert!((value - 1.25).abs() < 1e-6);
}

#[test]
fn reference_vm_applies_non_uv_mapping_to_image_textures() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("mapped.png");
    ImageBuffer::<Luma<u8>, _>::from_raw(1, 2, vec![0, 255])
        .unwrap()
        .save(&path)
        .unwrap();
    let mut params = ParameterDictionary::default();
    params.add_string("string filename", &path.to_string_lossy());
    params.add_string("string filter", "nearest");
    let mut node = TextureNode::new("mapped-image");
    node.components.push(TextureComponent::Texture(Texture {
        name: "imagemap".to_string(),
        kind: TextureKind::Float,
        params,
    }));
    node.components
        .push(TextureComponent::Mapping(TextureMapping::Cylindrical(
            Transform::default(),
        )));
    let library = compile_texture_library(&[TextureRootSpec::Float {
        node: Arc::new(node),
    }])
    .unwrap();

    let value = evaluate_texture_root_with_context(
        &library,
        0,
        TextureEvaluationContext {
            position: [1.0, 0.0, 0.25],
            ..Default::default()
        },
    )
    .unwrap();

    let TextureValue::Float(value) = value else {
        panic!("expected float texture value");
    };
    assert_eq!(value, 1.0);
}

#[test]
fn reference_vm_returns_black_outside_non_uv_image_mapping() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("black-wrap.png");
    ImageBuffer::<Luma<u8>, _>::from_raw(1, 1, vec![255])
        .unwrap()
        .save(&path)
        .unwrap();
    let mut params = ParameterDictionary::default();
    params.add_string("string filename", &path.to_string_lossy());
    params.add_string("string filter", "nearest");
    params.add_string("string wrap", "black");
    let mut node = TextureNode::new("black-wrap-image");
    node.components.push(TextureComponent::Texture(Texture {
        name: "imagemap".to_string(),
        kind: TextureKind::Float,
        params,
    }));
    node.components
        .push(TextureComponent::Mapping(TextureMapping::Planar(
            Transform::default(),
        )));
    let library = compile_texture_library(&[TextureRootSpec::Float {
        node: Arc::new(node),
    }])
    .unwrap();

    let value = evaluate_texture_root_with_context(
        &library,
        0,
        TextureEvaluationContext {
            position: [-0.25, 0.5, 0.0],
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(value, TextureValue::Float(0.0));
}

fn evaluate_mapped_bilerp(mapping: TextureMapping, position: [f32; 3]) -> f32 {
    let mut root = TextureNode::new("mapped-bilerp");
    root.components.push(TextureComponent::Texture(Texture {
        name: "bilerp".to_string(),
        kind: TextureKind::Float,
        params: ParameterDictionary::default(),
    }));
    root.components.push(TextureComponent::Mapping(mapping));
    root.children.push(float_constant("v00", 0.0));
    root.children.push(float_constant("v01", 1.0));
    root.children.push(float_constant("v10", 2.0));
    root.children.push(float_constant("v11", 3.0));
    let library = compile_texture_library(&[TextureRootSpec::Float {
        node: Arc::new(root),
    }])
    .unwrap();
    let value = evaluate_texture_root_with_context(
        &library,
        0,
        TextureEvaluationContext {
            position,
            ..Default::default()
        },
    )
    .unwrap();
    let TextureValue::Float(value) = value else {
        panic!("expected float texture value");
    };
    value
}

fn float_constant(name: &str, value: f32) -> Arc<TextureNode> {
    let mut params = ParameterDictionary::default();
    params.add_float("float value", value as _);
    let mut node = TextureNode::new(name);
    node.components.push(TextureComponent::Texture(Texture {
        name: "constant".to_string(),
        kind: TextureKind::Float,
        params,
    }));
    Arc::new(node)
}

fn spectrum_constant(name: &str, value: [f32; 3]) -> Arc<TextureNode> {
    let mut params = ParameterDictionary::default();
    params.add_spectrum("rgb value", &Spectrum::from(value));
    let mut node = TextureNode::new(name);
    node.components.push(TextureComponent::Texture(Texture {
        name: "constant".to_string(),
        kind: TextureKind::Spectrum,
        params,
    }));
    Arc::new(node)
}

fn float_bilerp(name: &str, value: f32) -> Arc<TextureNode> {
    let mut node = TextureNode::new(name);
    node.components.push(TextureComponent::Texture(Texture {
        name: "bilerp".to_string(),
        kind: TextureKind::Float,
        params: ParameterDictionary::default(),
    }));
    for slot in ["v00", "v01", "v10", "v11"] {
        node.children
            .push(float_constant(&format!("{name}:{slot}"), value));
    }
    Arc::new(node)
}
