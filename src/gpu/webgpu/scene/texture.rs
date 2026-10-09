use std::collections::HashMap;
use std::sync::Arc;

use crate::gpu::flat;
use crate::gpu::flat::texture::{
    ColorSpace, ImageFilterMode, ImageValueType, ImageView, ImageWrapMode, Mipmap,
    ProceduralOperation, TextureInstruction, TextureLibrary, TextureRoot, TextureValueType,
};
use crate::gpu::node::TextureMapping;
use crate::util::error::PbrtError;
use crate::util::spectrum::SpectrumType;

use super::super::abi::{
    row_major_to_columns, TextureNodeRecord, TextureRootRecord, INVALID_INDEX,
    TEXTURE_OPERATION_BILERP, TEXTURE_OPERATION_CHECKERBOARD, TEXTURE_OPERATION_CONSTANT,
    TEXTURE_OPERATION_DIRECTION_MIX, TEXTURE_OPERATION_DOTS, TEXTURE_OPERATION_FBM,
    TEXTURE_OPERATION_IMAGE, TEXTURE_OPERATION_MARBLE, TEXTURE_OPERATION_MIX,
    TEXTURE_OPERATION_SCALE, TEXTURE_OPERATION_WINDY, TEXTURE_OPERATION_WRINKLED,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct SamplerKey {
    pub(super) swrap: ImageWrapMode,
    pub(super) twrap: ImageWrapMode,
    pub(super) filter: ImageFilterMode,
}

pub(super) struct TextureBindingPlan {
    pub(super) image_views: Vec<usize>,
    pub(super) samplers: Vec<SamplerKey>,
    view_bindings: Vec<(u32, u32)>,
}

const INFINITE_IMAGE_BINDING_MASK: u32 = 0x0fff_ffff;

pub(super) fn validate_image_infinite_buffer_size(
    label: &str,
    byte_size: usize,
    limits: &wgpu::Limits,
) -> Result<(), PbrtError> {
    let byte_size = u64::try_from(byte_size)
        .map_err(|_| PbrtError::error("Image infinite light buffer size exceeds u64."))?;
    if byte_size > limits.max_buffer_size
        || byte_size > u64::from(limits.max_storage_buffer_binding_size)
    {
        return Err(PbrtError::error(&format!(
            "Image infinite light {label} buffer ({byte_size} bytes) exceeds WebGPU buffer limits (max buffer {}, max storage binding {}).",
            limits.max_buffer_size, limits.max_storage_buffer_binding_size
        )));
    }
    Ok(())
}

pub(super) fn infinite_image_payload(
    binding: usize,
    color_space: ColorSpace,
) -> Result<u32, PbrtError> {
    let binding = u32::try_from(binding)
        .map_err(|_| PbrtError::error("Infinite light image binding exceeds u32."))?;
    if binding > INFINITE_IMAGE_BINDING_MASK {
        return Err(PbrtError::error(
            "Infinite light image binding exceeds the packed payload range.",
        ));
    }
    let color_space = match color_space {
        ColorSpace::Unknown | ColorSpace::Srgb => 0,
        ColorSpace::Aces2065 => 1,
        ColorSpace::DciP3 => 2,
        ColorSpace::Rec2020 => 3,
    };
    Ok(binding | (color_space << 28))
}

pub fn resolve_infinite_image_bindings(
    infinite_lights: &[flat::Light],
    mipmaps: &[Arc<Mipmap>],
    texture_views: &[ImageView],
    image_view_bindings: &[usize],
) -> Result<HashMap<u32, u32>, PbrtError> {
    Ok(infinite_lights
        .iter()
        .map(|light| {
            if light.image_index == flat::INVALID_INDEX {
                return Ok(None);
            }
            let view_index = texture_views
                .iter()
                .position(|view| view.mipmap == light.image_index)
                .ok_or_else(|| PbrtError::error("Infinite light image view was not registered."))?;
            let binding = image_view_bindings
                .iter()
                .position(|&index| index == view_index)
                .ok_or_else(|| {
                    PbrtError::error("Infinite light image binding was not generated.")
                })?;
            let mipmap = mipmaps
                .get(light.image_index as usize)
                .ok_or_else(|| PbrtError::error("Infinite light references an invalid mipmap."))?;
            let payload = infinite_image_payload(binding, mipmap.color_space)?;
            Ok(Some((light.sampling_model, payload)))
        })
        .collect::<Result<Vec<_>, PbrtError>>()?
        .into_iter()
        .flatten()
        .collect::<HashMap<_, _>>())
}

pub(super) fn texture_binding_plan(views: &[ImageView]) -> Result<TextureBindingPlan, PbrtError> {
    let mut image_views = Vec::new();
    let mut images_by_mipmap = HashMap::new();
    let mut samplers = Vec::new();
    let mut samplers_by_key = HashMap::new();
    let mut view_bindings = Vec::with_capacity(views.len());
    for (view_index, view) in views.iter().enumerate() {
        let image = if let Some(&index) = images_by_mipmap.get(&view.mipmap) {
            index
        } else {
            let index = u32::try_from(image_views.len())
                .map_err(|_| PbrtError::error("Texture image table exceeds u32."))?;
            image_views.push(view_index);
            images_by_mipmap.insert(view.mipmap, index);
            index
        };
        let sampler_key = SamplerKey {
            swrap: view.swrap,
            twrap: view.twrap,
            filter: view.filter,
        };
        let sampler = if let Some(&index) = samplers_by_key.get(&sampler_key) {
            index
        } else {
            let index = u32::try_from(samplers.len())
                .map_err(|_| PbrtError::error("Texture sampler table exceeds u32."))?;
            samplers.push(sampler_key);
            samplers_by_key.insert(sampler_key, index);
            index
        };
        view_bindings.push((image, sampler));
    }
    Ok(TextureBindingPlan {
        image_views,
        samplers,
        view_bindings,
    })
}

pub(super) fn scene_texture_views(flat: &flat::Scene) -> (Vec<ImageView>, usize) {
    let mut views = Vec::new();
    for light in &flat.infinite_lights {
        if light.image_index == flat::INVALID_INDEX {
            continue;
        }
        views.push(ImageView {
            mipmap: light.image_index,
            value_type: ImageValueType::LinearRgb,
            swrap: ImageWrapMode::Clamp,
            twrap: ImageWrapMode::Clamp,
            filter: ImageFilterMode::Nearest,
            scale: 1.0,
            invert: false,
        });
    }
    let material_view_offset = views.len();
    views.extend(flat.texture_library.image_views.iter().cloned());
    (views, material_view_offset)
}

pub fn texture_binding_counts(flat: &flat::Scene) -> Result<(u32, u32), PbrtError> {
    let (views, _) = scene_texture_views(flat);
    let plan = texture_binding_plan(&views)?;
    Ok((
        u32::try_from(plan.image_views.len())
            .map_err(|_| PbrtError::error("Texture image table exceeds u32."))?,
        u32::try_from(plan.samplers.len())
            .map_err(|_| PbrtError::error("Texture sampler table exceeds u32."))?,
    ))
}

fn stable_texture_hash(value: &str) -> u32 {
    value.bytes().fold(2166136261u32, |hash, byte| {
        (hash ^ u32::from(byte)).wrapping_mul(16777619)
    })
}

pub(super) fn lower_texture_library(
    library: &TextureLibrary,
    binding_plan: &TextureBindingPlan,
    binding_view_offset: usize,
) -> Result<(Vec<TextureNodeRecord>, Vec<u32>, Vec<TextureRootRecord>), PbrtError> {
    let mut nodes = Vec::new();
    let mut children = Vec::new();
    let mut program_offsets = Vec::with_capacity(library.programs.len());
    for program in &library.programs {
        if program.instructions.len() > 256 {
            return Err(PbrtError::error(
                "Texture program exceeds the WebGPU post-order VM slot limit.",
            ));
        }
        let offset = u32::try_from(nodes.len())
            .map_err(|_| PbrtError::error("Texture node table exceeds u32."))?;
        program_offsets.push(offset);
        for (instruction_index, instruction) in program.instructions.iter().enumerate() {
            let operands = texture_instruction_operands(instruction);
            let first_child = u32::try_from(children.len())
                .map_err(|_| PbrtError::error("Texture child table exceeds u32."))?;
            children.extend(operands.iter().copied());
            let lowered = lower_texture_instruction(
                instruction,
                program.slot_types[instruction_index],
                &library.image_views,
                binding_plan,
                binding_view_offset,
            )?;
            nodes.push(TextureNodeRecord {
                kind: lowered.kind,
                first_child,
                child_count: u32::try_from(operands.len())
                    .map_err(|_| PbrtError::error("Texture child count exceeds u32."))?,
                implementation_hash: lowered.implementation_hash,
                swrap_mode: lowered.image_view.2,
                twrap_mode: lowered.image_view.3,
                color_space: lowered.color_space,
                texture_index: lowered.image_view.0,
                operation: lowered.operation,
                mapping_kind: lowered.mapping.0,
                sampler: lowered.image_view.1,
                image_filter_mode: lowered.image_view.4,
                constant_value: lowered.constant_value,
                mapping: lowered.mapping.1,
            });
        }
    }
    let roots =
        library
            .roots
            .iter()
            .map(|root| match root {
                TextureRoot::Float { program } => Ok(TextureRootRecord {
                    texture_node: program_offsets.get(*program as usize).copied().ok_or_else(
                        || PbrtError::error("Texture root references invalid program."),
                    )?,
                    instruction_count: u32::try_from(
                        library.programs[*program as usize].instructions.len(),
                    )
                    .map_err(|_| PbrtError::error("Texture program length exceeds u32."))?,
                    result: library.programs[*program as usize].result,
                    spectrum_type: 0,
                }),
                TextureRoot::Spectrum {
                    program,
                    spectrum_type,
                } => Ok(TextureRootRecord {
                    texture_node: program_offsets.get(*program as usize).copied().ok_or_else(
                        || PbrtError::error("Texture root references invalid program."),
                    )?,
                    instruction_count: u32::try_from(
                        library.programs[*program as usize].instructions.len(),
                    )
                    .map_err(|_| PbrtError::error("Texture program length exceeds u32."))?,
                    result: library.programs[*program as usize].result,
                    spectrum_type: match spectrum_type {
                        SpectrumType::Albedo => 0,
                        SpectrumType::Unbounded => 1,
                        SpectrumType::Illuminant => 2,
                    },
                }),
            })
            .collect::<Result<Vec<_>, PbrtError>>()?;
    Ok((nodes, children, roots))
}

/// Lower a backend-independent texture library into the records consumed by
/// the WebGPU texture VM. This entry point is also useful to validate the
/// host-side ABI without creating a WebGPU device.
pub fn lower_texture_library_records(
    library: &TextureLibrary,
) -> Result<(Vec<TextureNodeRecord>, Vec<u32>, Vec<TextureRootRecord>), PbrtError> {
    let binding_plan = texture_binding_plan(&library.image_views)?;
    lower_texture_library(library, &binding_plan, 0)
}

fn texture_instruction_operands(instruction: &TextureInstruction) -> Vec<u32> {
    match instruction {
        TextureInstruction::Scale { input, scale, .. } => std::iter::once(*input)
            .chain(scale.iter().copied())
            .collect(),
        TextureInstruction::Mix {
            first,
            second,
            amount,
            ..
        } => [*first, *second]
            .into_iter()
            .chain(amount.iter().copied())
            .collect(),
        TextureInstruction::Procedural { operands, .. } => operands.clone(),
        TextureInstruction::ConstantFloat { .. }
        | TextureInstruction::ConstantRgb { .. }
        | TextureInstruction::SampleImage { .. } => Vec::new(),
    }
}

struct LoweredTextureInstruction {
    kind: u32,
    implementation_hash: u32,
    operation: u32,
    constant_value: [f32; 4],
    color_space: u32,
    image_view: (u32, u32, u32, u32, u32),
    mapping: (u32, [[f32; 4]; 4]),
}

fn lower_texture_instruction(
    instruction: &TextureInstruction,
    slot_type: TextureValueType,
    image_views: &[ImageView],
    binding_plan: &TextureBindingPlan,
    binding_view_offset: usize,
) -> Result<LoweredTextureInstruction, PbrtError> {
    let identity = row_major_to_columns([
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    let value_type = |value_type: &TextureValueType| match value_type {
        TextureValueType::Float => (0, 0),
        TextureValueType::LinearRgb(color_space) => (1, color_space_id(*color_space)),
    };
    let empty_image = (INVALID_INDEX, INVALID_INDEX, 0, 0, 0);
    match instruction {
        TextureInstruction::ConstantFloat { value, .. } => Ok(LoweredTextureInstruction {
            kind: 0,
            implementation_hash: stable_texture_hash("constant"),
            operation: TEXTURE_OPERATION_CONSTANT,
            constant_value: [*value; 4],
            color_space: 0,
            image_view: empty_image,
            mapping: (0, identity),
        }),
        TextureInstruction::ConstantRgb {
            value, color_space, ..
        } => Ok(LoweredTextureInstruction {
            kind: 1,
            implementation_hash: stable_texture_hash("constant"),
            operation: TEXTURE_OPERATION_CONSTANT,
            constant_value: [
                value[0],
                value[1],
                value[2],
                (value[0] + value[1] + value[2]) / 3.0,
            ],
            color_space: color_space_id(*color_space),
            image_view: empty_image,
            mapping: (0, identity),
        }),
        TextureInstruction::SampleImage {
            image_view,
            mapping,
            value_type: texture_type,
            ..
        } => {
            let view = image_views
                .get(*image_view as usize)
                .ok_or_else(|| PbrtError::error("Texture instruction has invalid image view."))?;
            let (image, sampler) = *binding_plan
                .view_bindings
                .get(binding_view_offset + *image_view as usize)
                .ok_or_else(|| PbrtError::error("Texture instruction has invalid binding."))?;
            let wrap_mode = |mode| match mode {
                ImageWrapMode::Repeat => 0,
                ImageWrapMode::Clamp => 1,
                ImageWrapMode::Black => 2,
            };
            let filter_mode = match view.filter {
                ImageFilterMode::Nearest => 0,
                ImageFilterMode::Bilinear => 1,
                ImageFilterMode::Trilinear => 2,
            };
            Ok(LoweredTextureInstruction {
                kind: value_type(texture_type).0,
                implementation_hash: stable_texture_hash("imagemap"),
                operation: TEXTURE_OPERATION_IMAGE,
                constant_value: [view.scale, if view.invert { 1.0 } else { 0.0 }, 0.0, 0.0],
                color_space: value_type(texture_type).1,
                image_view: (
                    image,
                    sampler,
                    wrap_mode(view.swrap),
                    wrap_mode(view.twrap),
                    filter_mode,
                ),
                mapping: lower_mapping(mapping.as_ref(), identity),
            })
        }
        TextureInstruction::Scale {
            scale,
            constant_scale,
            ..
        } => Ok(LoweredTextureInstruction {
            kind: value_type(&slot_type).0,
            implementation_hash: stable_texture_hash("scale"),
            operation: TEXTURE_OPERATION_SCALE,
            constant_value: [
                if scale.is_some() {
                    0.0
                } else {
                    *constant_scale
                },
                0.0,
                0.0,
                0.0,
            ],
            color_space: value_type(&slot_type).1,
            image_view: empty_image,
            mapping: (0, identity),
        }),
        TextureInstruction::Mix {
            constant_amount, ..
        } => Ok(LoweredTextureInstruction {
            kind: value_type(&slot_type).0,
            implementation_hash: stable_texture_hash("mix"),
            operation: TEXTURE_OPERATION_MIX,
            constant_value: [*constant_amount, 0.0, 0.0, 0.0],
            color_space: 0,
            image_view: empty_image,
            mapping: (0, identity),
        }),
        TextureInstruction::Procedural {
            operation,
            parameters,
            mapping,
            value_type: texture_type,
            ..
        } => Ok(LoweredTextureInstruction {
            kind: value_type(texture_type).0,
            implementation_hash: stable_texture_hash(operation.name()),
            operation: procedural_operation(*operation)?,
            constant_value: *parameters,
            color_space: value_type(texture_type).1,
            image_view: empty_image,
            mapping: lower_mapping(mapping.as_ref(), identity),
        }),
    }
}

fn color_space_id(color_space: ColorSpace) -> u32 {
    match color_space {
        ColorSpace::Unknown | ColorSpace::Srgb => 0,
        ColorSpace::Aces2065 => 1,
        ColorSpace::DciP3 => 2,
        ColorSpace::Rec2020 => 3,
    }
}

fn lower_mapping(
    mapping: Option<&TextureMapping>,
    identity: [[f32; 4]; 4],
) -> (u32, [[f32; 4]; 4]) {
    match mapping {
        Some(TextureMapping::Uv(uv)) => {
            let mut matrix = [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ];
            matrix[0] = uv.uscale;
            matrix[5] = uv.vscale;
            matrix[3] = uv.udelta;
            matrix[7] = uv.vdelta;
            (0, row_major_to_columns(matrix))
        }
        Some(TextureMapping::Planar(transform)) => (1, row_major_to_columns(transform.matrix)),
        Some(TextureMapping::Spherical(transform)) => (2, row_major_to_columns(transform.matrix)),
        Some(TextureMapping::Cylindrical(transform)) => (3, row_major_to_columns(transform.matrix)),
        Some(TextureMapping::PointTransform(transform)) => {
            (4, row_major_to_columns(transform.matrix))
        }
        None => (0, identity),
    }
}

fn procedural_operation(operation: ProceduralOperation) -> Result<u32, PbrtError> {
    match operation {
        ProceduralOperation::Checkerboard => Ok(TEXTURE_OPERATION_CHECKERBOARD),
        ProceduralOperation::DirectionMix => Ok(TEXTURE_OPERATION_DIRECTION_MIX),
        ProceduralOperation::Bilerp => Ok(TEXTURE_OPERATION_BILERP),
        ProceduralOperation::Dots => Ok(TEXTURE_OPERATION_DOTS),
        ProceduralOperation::Fbm => Ok(TEXTURE_OPERATION_FBM),
        ProceduralOperation::Wrinkled => Ok(TEXTURE_OPERATION_WRINKLED),
        ProceduralOperation::Windy => Ok(TEXTURE_OPERATION_WINDY),
        ProceduralOperation::Marble => Ok(TEXTURE_OPERATION_MARBLE),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        lower_texture_instruction, texture_binding_plan, validate_image_infinite_buffer_size,
    };
    use crate::gpu::flat::texture::{
        ColorSpace, ImageFilterMode, ImageValueType, ImageView, ImageWrapMode, TextureInstruction,
        TextureValueType,
    };

    #[test]
    fn image_infinite_buffers_are_checked_against_webgpu_limits() {
        let limits = wgpu::Limits {
            max_buffer_size: 1024,
            max_storage_buffer_binding_size: 512,
            ..wgpu::Limits::default()
        };

        assert!(validate_image_infinite_buffer_size("test", 512, &limits).is_ok());
        assert!(validate_image_infinite_buffer_size("test", 513, &limits)
            .unwrap_err()
            .to_string()
            .contains("max storage binding 512"));
        assert!(validate_image_infinite_buffer_size("test", 1025, &limits)
            .unwrap_err()
            .to_string()
            .contains("max buffer 1024"));
    }

    #[test]
    fn texture_binding_plan_shares_images_and_samplers_independently() {
        let make_view = |mipmap, filter| ImageView {
            mipmap,
            value_type: ImageValueType::LinearRgb,
            swrap: ImageWrapMode::Repeat,
            twrap: ImageWrapMode::Clamp,
            filter,
            scale: 1.0,
            invert: false,
        };
        let views = vec![
            make_view(0, ImageFilterMode::Bilinear),
            make_view(0, ImageFilterMode::Trilinear),
            make_view(1, ImageFilterMode::Bilinear),
        ];

        let plan = texture_binding_plan(&views).unwrap();

        assert_eq!(plan.image_views.len(), 2);
        assert_eq!(plan.samplers.len(), 2);
        assert_eq!(plan.view_bindings, vec![(0, 0), (0, 1), (1, 0)]);
    }

    #[test]
    fn material_image_binding_follows_prefixed_infinite_image_view() {
        let make_view = |mipmap, filter| ImageView {
            mipmap,
            value_type: ImageValueType::LinearRgb,
            swrap: ImageWrapMode::Repeat,
            twrap: ImageWrapMode::Clamp,
            filter,
            scale: 1.0,
            invert: false,
        };
        let infinite_view = make_view(0, ImageFilterMode::Nearest);
        let material_view = make_view(1, ImageFilterMode::Bilinear);
        let plan = texture_binding_plan(&[infinite_view, material_view.clone()]).unwrap();
        let instruction = TextureInstruction::SampleImage {
            dst: 0,
            image_view: 0,
            mapping: None,
            value_type: TextureValueType::LinearRgb(ColorSpace::Srgb),
        };

        let lowered = lower_texture_instruction(
            &instruction,
            TextureValueType::LinearRgb(ColorSpace::Srgb),
            &[material_view],
            &plan,
            1,
        )
        .unwrap();

        assert_eq!(lowered.image_view.0, 1);
        assert_eq!(lowered.image_view.1, 1);
    }
}
