use std::sync::Arc;

use pbrt_r4::gpu::node::TextureNode;
use pbrt_r4::gpu::node::{Texture, TextureComponent, TextureKind, TextureMapping, UvMapping};
use pbrt_r4::gpu::texture::{
    compile_texture_library, evaluate_texture_program, evaluate_texture_program_at, ColorSpace,
    ImageCompiler, ImageOptimizationPolicy, ImageValueType, Mipmap, MipmapEncoding, MipmapLevel,
    MipmapLevelData, TextureInstruction, TextureRoot, TextureRootSpec, TextureValue,
    TextureValueType,
};
use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::util::spectrum::SpectrumType;

#[test]
fn texture_library_keeps_root_interpretation_outside_programs() {
    let mut node = TextureNode::new("constant");
    node.components.push(TextureComponent::Texture(Texture {
        name: "constant".to_string(),
        kind: TextureKind::Float,
        params: ParameterDictionary::default(),
        mipmap: None,
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
fn texture_program_is_typed_post_order() {
    let mut child = TextureNode::new("constant");
    child.components.push(TextureComponent::Texture(Texture {
        name: "constant".to_string(),
        kind: TextureKind::Float,
        params: ParameterDictionary::default(),
        mipmap: None,
    }));

    let mut root = TextureNode::new("scale");
    let mut params = ParameterDictionary::default();
    params.add_float("float scale", 2.0);
    root.components.push(TextureComponent::Texture(Texture {
        name: "scale".to_string(),
        kind: TextureKind::Float,
        params,
        mipmap: None,
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
        evaluate_texture_program(program).unwrap(),
        TextureValue::Float(0.0)
    );
}

#[test]
fn image_instruction_keeps_sampling_interpretation() {
    let mut params = ParameterDictionary::default();
    params.add_string("string wrap", "black");
    params.add_string("string filter", "nearest");
    params.add_float("float scale", 3.0);
    params.add_bool("bool invert", true);
    let mut node = TextureNode::new("imagemap");
    node.components.push(TextureComponent::Texture(Texture {
        name: "imagemap".to_string(),
        kind: TextureKind::Float,
        params,
        mipmap: Some(Arc::new(Mipmap {
            levels: vec![MipmapLevel {
                resolution: [1, 1],
                channels: 1,
                data: MipmapLevelData::F32(vec![0.25]),
            }],
            color_space: ColorSpace::Unknown,
            encoding: MipmapEncoding::Linear,
        })),
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
    assert_eq!(library.image_views.len(), 1);
    let TextureInstruction::SampleImage { image_view, .. } = &library.programs[0].instructions[0]
    else {
        panic!("expected image sample instruction");
    };
    let view = &library.programs[0].image_views[*image_view as usize];
    assert_eq!(view.swrap, pbrt_r4::gpu::texture::ImageWrapMode::Black);
    assert_eq!(view.filter, pbrt_r4::gpu::texture::ImageFilterMode::Nearest);
    assert_eq!(view.scale, 3.0);
    assert!(view.invert);
    assert!(matches!(
        &library.programs[0].instructions[0],
        TextureInstruction::SampleImage {
            mapping: Some(TextureMapping::Uv(UvMapping { uscale: 2.0, .. })),
            ..
        }
    ));
    assert_eq!(
        evaluate_texture_program_at(&library.programs[0], [0.2, 0.2]).unwrap(),
        TextureValue::Float(2.25)
    );
}
