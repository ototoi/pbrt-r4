use std::sync::Arc;

use pbrt_r4::gpu::node::TextureNode;
use pbrt_r4::gpu::texture::{
    compile_texture_library, ColorSpace, ImageCompiler, ImageOptimizationPolicy, ImageValueType,
    Mipmap, MipmapEncoding, MipmapLevel, MipmapLevelData, TextureRoot, TextureRootSpec,
};
use pbrt_r4::util::spectrum::SpectrumType;

#[test]
fn texture_library_keeps_root_interpretation_outside_programs() {
    let node = Arc::new(TextureNode::new("shared"));
    let roots = [
        TextureRootSpec::Float { node: node.clone() },
        TextureRootSpec::Spectrum {
            node,
            spectrum_type: SpectrumType::Unbounded,
        },
    ];

    let library = compile_texture_library(&roots).unwrap();
    assert_eq!(library.programs.len(), 2);
    assert_eq!(library.roots.len(), 2);

    assert!(matches!(
        library.roots[0],
        TextureRoot::Float { program: 0 }
    ));
    assert!(matches!(
        library.roots[1],
        TextureRoot::Spectrum {
            program: 1,
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
