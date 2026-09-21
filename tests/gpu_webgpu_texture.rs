use pbrt_r4::gpu::flat::texture::{
    ColorSpace, TextureInstruction, TextureLibrary, TextureRoot, TextureValueType,
    TypedTextureProgram,
};
use pbrt_r4::gpu::webgpu::abi::TEXTURE_OPERATION_SCALE;
use pbrt_r4::gpu::webgpu::scene::lower_texture_library_records;

#[test]
fn webgpu_scale_lowering_uses_one_or_two_children() {
    let library = TextureLibrary {
        programs: vec![
            TypedTextureProgram {
                instructions: vec![
                    TextureInstruction::ConstantRgb {
                        dst: 0,
                        value: [0.2, 0.4, 0.6],
                        color_space: ColorSpace::Srgb,
                    },
                    TextureInstruction::ConstantFloat { dst: 1, value: 0.5 },
                    TextureInstruction::Scale {
                        dst: 2,
                        input: 0,
                        scale: Some(1),
                        constant_scale: 7.0,
                    },
                ],
                slot_types: vec![
                    TextureValueType::LinearRgb(ColorSpace::Srgb),
                    TextureValueType::Float,
                    TextureValueType::LinearRgb(ColorSpace::Srgb),
                ],
                slot_last_use: vec![2, 2, 2],
                result: 2,
            },
            TypedTextureProgram {
                instructions: vec![
                    TextureInstruction::ConstantFloat { dst: 0, value: 2.0 },
                    TextureInstruction::Scale {
                        dst: 1,
                        input: 0,
                        scale: None,
                        constant_scale: 0.5,
                    },
                ],
                slot_types: vec![TextureValueType::Float; 2],
                slot_last_use: vec![1, 1],
                result: 1,
            },
        ],
        roots: vec![
            TextureRoot::Spectrum {
                program: 0,
                spectrum_type: pbrt_r4::util::spectrum::SpectrumType::Albedo,
            },
            TextureRoot::Float { program: 1 },
        ],
        mipmaps: Vec::new(),
        image_views: Vec::new(),
    };

    let (nodes, children, _) = lower_texture_library_records(&library).unwrap();
    let scale_nodes = nodes
        .iter()
        .filter(|node| node.operation == TEXTURE_OPERATION_SCALE)
        .collect::<Vec<_>>();

    assert_eq!(scale_nodes.len(), 2);
    assert_eq!(scale_nodes[0].child_count, 2);
    assert_eq!(scale_nodes[0].constant_value[0], 0.0);
    assert_eq!(scale_nodes[1].child_count, 1);
    assert_eq!(scale_nodes[1].constant_value[0], 0.5);
    assert_eq!(children, vec![0, 1, 0]);
}
