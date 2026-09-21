use bytemuck::cast_slice;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::gpu::flat;
use crate::gpu::flat::texture::{
    ColorSpace, ImageFilterMode, ImageValueType, ImageView, ImageWrapMode, MipmapEncoding,
    MipmapLevel, MipmapLevelData, ProceduralOperation, TextureInstruction, TextureLibrary,
    TextureRoot, TextureValueType,
};
use crate::gpu::node::TextureMapping;
use crate::util::error::PbrtError;

use super::abi::{
    camera_uniform, film_uniform, inverse_transpose_linear, light_table_uniform,
    material_table_uniform, row_major_to_columns, viewport_uniform, AttributeRef, CameraUniform,
    DenseSpectrum, FilmUniform, Geometry, Instance, LightRecord, LightSamplingModel,
    LightTableUniform, MaterialNode, MaterialRoot, MaterialTableUniform, MeasuredBsdfRecord,
    MeasuredTableRecord, TextureNodeRecord, TextureRootRecord, TriangleDistributionEntry, Vertex,
    ViewportUniform, INVALID_INDEX, LIGHT_KIND_AREA, LIGHT_KIND_DISTANT, LIGHT_KIND_IMAGE_INFINITE,
    LIGHT_KIND_POINT, LIGHT_KIND_PORTAL_IMAGE_INFINITE, LIGHT_KIND_SPOT,
    LIGHT_KIND_UNIFORM_INFINITE, TEXTURE_OPERATION_BILERP, TEXTURE_OPERATION_CHECKERBOARD,
    TEXTURE_OPERATION_CONSTANT, TEXTURE_OPERATION_DIRECTION_MIX, TEXTURE_OPERATION_DOTS,
    TEXTURE_OPERATION_FBM, TEXTURE_OPERATION_IMAGE, TEXTURE_OPERATION_MARBLE,
    TEXTURE_OPERATION_MIX, TEXTURE_OPERATION_SCALE, TEXTURE_OPERATION_WINDY,
    TEXTURE_OPERATION_WRINKLED,
};
use super::acceleration::{self, Acceleration};
use super::light_bvh::pack_light_bvh;
use super::light_sampler::{resolve_scene_light_sampler_count, LightSamplerKind};
use super::material::MaterialKind;
use super::material::MaterialTable;
use super::output::Output;
use super::render_settings::RenderSettings;
use super::sampler::SamplerResources;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SamplerKey {
    swrap: ImageWrapMode,
    twrap: ImageWrapMode,
    filter: ImageFilterMode,
}

struct TextureBindingPlan {
    image_views: Vec<usize>,
    samplers: Vec<SamplerKey>,
    view_bindings: Vec<(u32, u32)>,
}

const INFINITE_IMAGE_BINDING_MASK: u32 = 0x0fff_ffff;

fn infinite_image_payload(binding: usize, color_space: ColorSpace) -> Result<u32, PbrtError> {
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

fn texture_binding_plan(views: &[ImageView]) -> Result<TextureBindingPlan, PbrtError> {
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

fn scene_texture_views(flat: &flat::Scene) -> (Vec<ImageView>, usize) {
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

fn lower_texture_library(
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
                _operation_padding: 0,
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
                        crate::util::spectrum::SpectrumType::Albedo => 0,
                        crate::util::spectrum::SpectrumType::Unbounded => 1,
                        crate::util::spectrum::SpectrumType::Illuminant => 2,
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
    image_view: (u32, u32, u32, u32),
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
    let empty_image = (INVALID_INDEX, INVALID_INDEX, 0, 0);
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
            Ok(LoweredTextureInstruction {
                kind: value_type(texture_type).0,
                implementation_hash: stable_texture_hash("imagemap"),
                operation: TEXTURE_OPERATION_IMAGE,
                constant_value: [view.scale, if view.invert { 1.0 } else { 0.0 }, 0.0, 0.0],
                color_space: value_type(texture_type).1,
                image_view: (image, sampler, wrap_mode(view.swrap), wrap_mode(view.twrap)),
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
            color_space: 0,
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

fn mip_level_rgba(
    level: &MipmapLevel,
    value_type: ImageValueType,
    encoding: MipmapEncoding,
) -> Result<(u32, u32, Vec<f32>), PbrtError> {
    let width = level.resolution[0];
    let height = level.resolution[1];
    if width == 0 || height == 0 {
        return Err(PbrtError::error(
            "Texture mipmap has an invalid resolution.",
        ));
    }
    if encoding != MipmapEncoding::Linear {
        return Err(PbrtError::error(
            "WebGPU texture upload requires linear Flat IR mipmaps.",
        ));
    }
    let expected_channels = match value_type {
        ImageValueType::Float => 1,
        ImageValueType::LinearRgb => 3,
    };
    if level.channels != expected_channels {
        return Err(PbrtError::error(&format!(
            "WebGPU texture upload expected {expected_channels} channels, got {}.",
            level.channels
        )));
    }
    let values = match &level.data {
        MipmapLevelData::F32(values) => values.clone(),
        MipmapLevelData::F16(values) => values
            .iter()
            .map(|value| half::f16::from_bits(*value).to_f32())
            .collect(),
        MipmapLevelData::U8(values) => values
            .iter()
            .map(|value| f32::from(*value) / 255.0)
            .collect(),
    };
    let pixel_count = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or_else(|| PbrtError::error("Texture mipmap resolution overflowed."))?;
    let channels = usize::try_from(level.channels)
        .map_err(|_| PbrtError::error("Texture channel count does not fit usize."))?;
    if values.len() != pixel_count.saturating_mul(channels) {
        return Err(PbrtError::error(
            "Texture mipmap data size is inconsistent.",
        ));
    }
    let mut rgba = vec![0.0f32; pixel_count * 4];
    for pixel in 0..pixel_count {
        let source = pixel * channels;
        let rgb = match value_type {
            ImageValueType::Float => [values[source]; 3],
            ImageValueType::LinearRgb => [values[source], values[source + 1], values[source + 2]],
        };
        for (channel, value) in rgb.into_iter().enumerate() {
            rgba[pixel * 4 + channel] = value;
        }
        rgba[pixel * 4 + 3] = 1.0;
    }
    Ok((width, height, rgba))
}

fn upload_texture_images(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mipmaps: &[Arc<crate::gpu::flat::texture::Mipmap>],
    views: &[ImageView],
    image_views: &[usize],
) -> Result<Vec<wgpu::Texture>, PbrtError> {
    let mut images = Vec::new();
    for &view_index in image_views {
        let view = views
            .get(view_index)
            .ok_or_else(|| PbrtError::error("Texture binding references an invalid image view."))?;
        let mipmap = mipmaps
            .get(view.mipmap as usize)
            .ok_or_else(|| PbrtError::error("Texture view references an invalid mipmap."))?;
        let base_level = mipmap
            .levels
            .first()
            .ok_or_else(|| PbrtError::error("WebGPU texture upload received an empty mipmap."))?;
        let (width, height, _) = mip_level_rgba(base_level, view.value_type, mipmap.encoding)?;
        let mip_level_count = u32::try_from(mipmap.levels.len())
            .map_err(|_| PbrtError::error("Texture mipmap has too many levels."))?;
        for (level_index, level) in mipmap.levels.iter().enumerate() {
            let expected_width = (width >> level_index).max(1);
            let expected_height = (height >> level_index).max(1);
            if level.resolution != [expected_width, expected_height] {
                return Err(PbrtError::error(&format!(
                    "Texture mipmap level {level_index} has resolution {:?}, expected [{expected_width}, {expected_height}].",
                    level.resolution
                )));
            }
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pbrt-r4 image texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (mip_level, level) in mipmap.levels.iter().enumerate() {
            let (level_width, level_height, rgba) =
                mip_level_rgba(level, view.value_type, mipmap.encoding)?;
            let mip_level = u32::try_from(mip_level)
                .map_err(|_| PbrtError::error("Texture mipmap level index overflowed."))?;
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&rgba),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(level_width * 16),
                    rows_per_image: Some(level_height),
                },
                wgpu::Extent3d {
                    width: level_width,
                    height: level_height,
                    depth_or_array_layers: 1,
                },
            );
        }
        images.push(texture);
    }
    Ok(images)
}

fn upload_measured_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pages: &[flat::MeasuredAtlasPage],
) -> Result<Vec<wgpu::Texture>, PbrtError> {
    pages
        .iter()
        .map(|page| {
            let [width, height] = page.resolution;
            let expected = usize::try_from(width)
                .ok()
                .and_then(|width| {
                    usize::try_from(height)
                        .ok()
                        .and_then(|height| width.checked_mul(height))
                })
                .and_then(|texels| texels.checked_mul(4))
                .ok_or_else(|| PbrtError::error("Measured BSDF atlas size overflowed."))?;
            if width == 0 || height == 0 || page.texels.len() != expected {
                return Err(PbrtError::error(
                    "Measured BSDF atlas page has inconsistent dimensions.",
                ));
            }
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("pbrt-r4 measured BSDF atlas"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&page.texels),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 16),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            Ok(texture)
        })
        .collect()
}

pub struct Scene {
    pub camera: CameraUniform,
    pub viewport: ViewportUniform,
    pub film: FilmUniform,
    pub film_output_matrix: [[f32; 3]; 3],
    pub film_scale: f32,
    pub material_table: MaterialTableUniform,
    pub light_table: LightTableUniform,
    pub sampler: SamplerResources,
    pub output: Output,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub geometry_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub material_root_buffer: wgpu::Buffer,
    pub material_node_buffer: wgpu::Buffer,
    pub attribute_ref_buffer: wgpu::Buffer,
    pub scalar_attribute_buffer: wgpu::Buffer,
    pub spectrum_attribute_buffer: wgpu::Buffer,
    pub measured_bsdf_buffer: wgpu::Buffer,
    pub measured_table_buffer: wgpu::Buffer,
    pub texture_root_buffer: wgpu::Buffer,
    pub texture_node_buffer: wgpu::Buffer,
    pub texture_child_buffer: wgpu::Buffer,
    pub rgb_spectrum_table_buffer: wgpu::Buffer,
    pub texture_images: Vec<wgpu::Texture>,
    pub texture_image_view: wgpu::TextureView,
    pub texture_image_views: Vec<wgpu::TextureView>,
    pub texture_samplers: Vec<wgpu::Sampler>,
    pub light_record_buffer: wgpu::Buffer,
    pub light_sampling_model_buffer: wgpu::Buffer,
    pub light_position_buffer: wgpu::Buffer,
    pub distribution_buffer: wgpu::Buffer,
    pub light_bvh_header_buffer: wgpu::Buffer,
    pub light_bvh_node_buffer: wgpu::Buffer,
    pub light_leaf_buffer: wgpu::Buffer,
    pub geometries: Vec<Geometry>,
    pub instances: Vec<Instance>,
    pub material_nodes: Vec<MaterialNode>,
    pub light_sampling_models: Vec<LightSamplingModel>,
    pub light_records: Vec<LightRecord>,
    pub light_sampler_kind: LightSamplerKind,
    pub render_settings: RenderSettings,
    pub acceleration: Acceleration,
}

impl Scene {
    pub fn from_flat(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        flat: flat::Scene,
    ) -> Result<Self, PbrtError> {
        if flat.output.filename.is_empty() {
            return Err(PbrtError::error(
                "WebGPU output filename must not be empty.",
            ));
        }
        let (vertices, geometries, indices) = convert_geometry(&flat)?;
        let instances = flat
            .instances
            .iter()
            .enumerate()
            .map(|(index, instance)| {
                if instance.geometry as usize >= geometries.len() {
                    return Err(PbrtError::error(&format!(
                        "Flat instance {index} references an invalid geometry."
                    )));
                }
                if instance.material_root as usize >= flat.material_roots.len() {
                    return Err(PbrtError::error(&format!(
                        "Flat instance {index} references an invalid material."
                    )));
                }
                validate_instance_area_lights(index, instance, &flat)?;
                let label = format!("Flat instance {index}");
                Ok(Instance {
                    geometry: instance.geometry,
                    material_root: instance.material_root,
                    area_light: instance.area_light,
                    orientation_flags: u32::from(instance.reverse_orientation)
                        | (u32::from(flat::transform_swaps_handedness(instance.transform)) << 1),
                    world_from_object: row_major_to_columns(instance.transform),
                    normal_from_object: inverse_transpose_linear(instance.transform, &label)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let material_table = MaterialTable::from_flat(&flat)?;
        let material_nodes = material_table.nodes;
        let measured_bsdfs = flat
            .measured_bsdfs
            .bsdfs
            .iter()
            .map(|record| MeasuredBsdfRecord {
                ndf: record.ndf,
                sigma: record.sigma,
                vndf: record.vndf,
                luminance: record.luminance,
                spectra: record.spectra,
                isotropic: u32::from(record.isotropic),
                padding: [0; 2],
            })
            .collect::<Vec<_>>();
        let measured_tables = flat
            .measured_bsdfs
            .tables
            .iter()
            .map(|record| MeasuredTableRecord {
                size: record.size,
                parameter_count: record.parameter_count,
                padding0: 0,
                parameter_sizes: record.parameter_sizes,
                padding1: 0,
                parameter_strides: record.parameter_strides,
                padding2: 0,
                parameter_value_offsets: record.parameter_value_offsets,
                padding3: 0,
                data_offset: record.data_offset,
                marginal_cdf_offset: record.marginal_cdf_offset,
                conditional_cdf_offset: record.conditional_cdf_offset,
                padding4: 0,
            })
            .collect::<Vec<_>>();
        flat.texture_library.validate()?;
        let mut attribute_refs = material_table.attributes;
        let all_lights = flat
            .lights
            .iter()
            .chain(flat.infinite_lights.iter())
            .collect::<Vec<_>>();
        let light_attribute_offsets = all_lights
            .iter()
            .scan(attribute_refs.len() as u32, |offset, light| {
                let current = *offset;
                *offset = offset.saturating_add(light.attributes.len() as u32);
                Some(current)
            })
            .collect::<Vec<_>>();
        attribute_refs.extend(
            all_lights
                .iter()
                .flat_map(|light| light.attributes.iter())
                .map(|attribute| AttributeRef {
                    kind: match attribute.kind {
                        flat::AttributeKind::Scalar => 0,
                        flat::AttributeKind::Spectrum => 1,
                        flat::AttributeKind::Texture => 2,
                        flat::AttributeKind::Measured => 3,
                    },
                    index: attribute.index,
                }),
        );
        let scalar_attributes = flat.scalar_attributes.clone();
        // Keep the infinite-image sampler first. pbrt-v4 uses nearest lookup
        // for ImageInfiniteLight after equal-area sphere-to-square mapping.
        let (texture_views, material_view_offset) = scene_texture_views(&flat);
        let texture_binding_plan = texture_binding_plan(&texture_views)?;
        let (texture_nodes, texture_children, texture_roots) = lower_texture_library(
            &flat.texture_library,
            &texture_binding_plan,
            material_view_offset,
        )?;
        let infinite_image_bindings = flat
            .infinite_lights
            .iter()
            .map(|light| {
                if light.image_index == flat::INVALID_INDEX {
                    return Ok(None);
                }
                let view_index = texture_views
                    .iter()
                    .position(|view| view.mipmap == light.image_index)
                    .ok_or_else(|| {
                        PbrtError::error("Infinite light image view was not registered.")
                    })?;
                let binding = texture_binding_plan
                    .image_views
                    .iter()
                    .position(|&index| index == view_index)
                    .ok_or_else(|| {
                        PbrtError::error("Infinite light image binding was not generated.")
                    })?;
                let mipmap = flat
                    .texture_library
                    .mipmaps
                    .get(light.image_index as usize)
                    .ok_or_else(|| {
                        PbrtError::error("Infinite light references an invalid mipmap.")
                    })?;
                let payload = infinite_image_payload(binding, mipmap.color_space)?;
                Ok(Some((light.sampling_model, payload)))
            })
            .collect::<Result<Vec<_>, PbrtError>>()?
            .into_iter()
            .flatten()
            .collect::<HashMap<_, _>>();
        let light_sampling_models = flat
            .light_sampling_models
            .iter()
            .enumerate()
            .map(|(model_index, model)| LightSamplingModel {
                kind: match model.kind {
                    flat::LightKind::Point => LIGHT_KIND_POINT,
                    flat::LightKind::Spot => LIGHT_KIND_SPOT,
                    flat::LightKind::Area => LIGHT_KIND_AREA,
                    flat::LightKind::Distant => LIGHT_KIND_DISTANT,
                    flat::LightKind::UniformInfinite => LIGHT_KIND_UNIFORM_INFINITE,
                    flat::LightKind::ImageInfinite => LIGHT_KIND_IMAGE_INFINITE,
                    flat::LightKind::PortalImageInfinite => LIGHT_KIND_PORTAL_IMAGE_INFINITE,
                },
                geometry_kind: match model.geometry_kind {
                    flat::LightGeometryKind::Position => 0,
                    flat::LightGeometryKind::Instance => 1,
                    flat::LightGeometryKind::Direction => 2,
                },
                geometry_index: model.geometry_index,
                direction_index: model.direction_index,
                distribution_offset_words: model.distribution_offset,
                distribution_count: model.distribution_count,
                total_area: model.total_area,
                flags: infinite_image_bindings
                    .get(&(model_index as u32))
                    .copied()
                    .unwrap_or(model.flags),
                world_to_light: model.world_to_light,
            })
            .collect::<Vec<_>>();
        let light_records = flat
            .lights
            .iter()
            .enumerate()
            .chain(
                flat.infinite_lights
                    .iter()
                    .enumerate()
                    .map(|(i, r)| (flat.lights.len() + i, r)),
            )
            .map(|(light_index, record)| LightRecord {
                kind: match record.kind {
                    flat::LightKind::Point => LIGHT_KIND_POINT,
                    flat::LightKind::Spot => LIGHT_KIND_SPOT,
                    flat::LightKind::Area => LIGHT_KIND_AREA,
                    flat::LightKind::Distant => LIGHT_KIND_DISTANT,
                    flat::LightKind::UniformInfinite => LIGHT_KIND_UNIFORM_INFINITE,
                    flat::LightKind::ImageInfinite => LIGHT_KIND_IMAGE_INFINITE,
                    flat::LightKind::PortalImageInfinite => LIGHT_KIND_PORTAL_IMAGE_INFINITE,
                },
                attribute_offset: light_attribute_offsets[light_index],
                attribute_count: u32::try_from(record.attributes.len()).unwrap_or(0),
                sampling_model: record.sampling_model,
            })
            .collect::<Vec<_>>();
        let camera = camera_uniform(&flat.camera, &flat.viewport)?;
        let viewport = viewport_uniform(&flat.viewport, &flat.render_settings)?;
        let sampler = SamplerResources::new(
            device,
            queue,
            &flat.render_settings,
            flat.viewport.resolution,
        )?;
        let film = film_uniform(&flat.film);
        let film_output_matrix = flat.film.output_rgb_from_sensor_rgb;
        let film_scale = flat.film.scale;
        if vertices.is_empty()
            || indices.is_empty()
            || instances.is_empty()
            || material_nodes.is_empty()
        {
            return Err(PbrtError::error(
                "WebGPU primary-ray rendering requires non-empty geometry, instances, and materials.",
            ));
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 vertex SBO"),
            contents: buffer_contents(&vertices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 local index SBO"),
            contents: buffer_contents(&indices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        });
        let geometry_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 geometry SBO"),
            contents: buffer_contents(&geometries),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 instance SBO"),
            contents: buffer_contents(&instances),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let material_roots = flat
            .material_roots
            .iter()
            .map(|layout| MaterialRoot {
                node_offset: layout.node_offset,
                node_count: layout.node_count,
            })
            .collect::<Vec<_>>();
        let material_root_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 material tree layouts SBO"),
            contents: buffer_contents(&material_roots),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let material_node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 material tree nodes SBO"),
            contents: buffer_contents(&material_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let attribute_ref_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 material attribute refs SBO"),
            contents: buffer_contents(&attribute_refs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let scalar_attribute_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 scalar attributes SBO"),
                contents: buffer_contents(&scalar_attributes),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let texture_root_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 texture roots SBO"),
            contents: buffer_contents(&texture_roots),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let texture_node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 texture nodes SBO"),
            contents: buffer_contents(&texture_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let texture_child_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 texture child indices SBO"),
            contents: buffer_contents(&texture_children),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut rgb_spectrum_table = Vec::new();
        for table in [
            include_bytes!(concat!(env!("OUT_DIR"), "/rgb_to_spectrum_srgb.bin")),
            include_bytes!(concat!(env!("OUT_DIR"), "/rgb_to_spectrum_aces.bin")),
            include_bytes!(concat!(env!("OUT_DIR"), "/rgb_to_spectrum_dci_p3.bin")),
            include_bytes!(concat!(env!("OUT_DIR"), "/rgb_to_spectrum_rec2020.bin")),
        ] {
            rgb_spectrum_table.extend_from_slice(table);
        }
        let rgb_spectrum_table_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 RGB spectrum tables SBO"),
                contents: &rgb_spectrum_table,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let mut texture_images = upload_texture_images(
            device,
            queue,
            &flat.texture_library.mipmaps,
            &texture_views,
            &texture_binding_plan.image_views,
        )?;
        let measured_texture_binding_base = u32::try_from(texture_images.len())
            .map_err(|_| PbrtError::error("Texture image binding count exceeds u32."))?;
        texture_images.extend(upload_measured_atlas(
            device,
            queue,
            &flat.measured_bsdfs.atlas_pages,
        )?);
        if texture_images.is_empty() {
            texture_images.push(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("pbrt-r4 empty texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }));
        }
        let texture_image_view =
            texture_images[0].create_view(&wgpu::TextureViewDescriptor::default());
        let texture_image_views = texture_images
            .iter()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();
        let texture_samplers = if texture_binding_plan.samplers.is_empty() {
            vec![device.create_sampler(&wgpu::SamplerDescriptor::default())]
        } else {
            texture_binding_plan
                .samplers
                .iter()
                .map(|sampler| {
                    let address_mode = |mode| match mode {
                        ImageWrapMode::Clamp | ImageWrapMode::Black => {
                            wgpu::AddressMode::ClampToEdge
                        }
                        ImageWrapMode::Repeat => wgpu::AddressMode::Repeat,
                    };
                    let filter = if sampler.filter == ImageFilterMode::Nearest {
                        wgpu::FilterMode::Nearest
                    } else {
                        wgpu::FilterMode::Linear
                    };
                    device.create_sampler(&wgpu::SamplerDescriptor {
                        label: Some("pbrt-r4 texture sampler"),
                        address_mode_u: address_mode(sampler.swrap),
                        address_mode_v: address_mode(sampler.twrap),
                        mag_filter: filter,
                        min_filter: filter,
                        mipmap_filter: if sampler.filter == ImageFilterMode::Trilinear {
                            wgpu::MipmapFilterMode::Linear
                        } else {
                            wgpu::MipmapFilterMode::Nearest
                        },
                        ..Default::default()
                    })
                })
                .collect()
        };
        flat::validate_dense_spectra(&flat.spectrum_attributes)?;
        let spectrum_attributes = flat
            .spectrum_attributes
            .iter()
            .map(|spectrum| DenseSpectrum {
                samples: spectrum.samples,
                flags: spectrum.flags,
            })
            .collect::<Vec<_>>();
        let spectrum_attribute_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 dense spectra SBO"),
                contents: buffer_contents(&spectrum_attributes),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let measured_bsdf_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 measured BSDF records SBO"),
            contents: buffer_contents(&measured_bsdfs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let measured_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 measured BSDF tables SBO"),
            contents: buffer_contents(&measured_tables),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let distribution_entries = flat
            .triangle_distributions
            .iter()
            .map(|entry| TriangleDistributionEntry {
                primitive: entry.primitive,
                cdf: entry.cdf,
                area: entry.area,
                reserved: 0,
            })
            .collect::<Vec<_>>();
        let light_record_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 light record SBO"),
            contents: buffer_contents(&light_records),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let light_sampling_model_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 light sampling model SBO"),
                contents: buffer_contents(&light_sampling_models),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let light_positions = flat
            .light_positions
            .iter()
            .map(|p| [p[0], p[1], p[2], 1.0])
            .collect::<Vec<_>>();
        let light_position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 light position SBO"),
            contents: buffer_contents(&light_positions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let distribution_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 triangle distribution SBO"),
            contents: buffer_contents(&distribution_entries),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let packed_light_bvh = pack_light_bvh(&flat.light_bvh)?;
        let light_sampler_kind =
            resolve_scene_light_sampler_count(&flat.render_settings, light_records.len())?;
        let mut material_table = material_table_uniform(material_nodes.len())?;
        material_table.measured_texture_base = measured_texture_binding_base;
        material_table.measured_texture_width = flat::MEASURED_ATLAS_WIDTH;
        material_table.measured_texture_height = flat::MEASURED_ATLAS_HEIGHT;
        material_table.measured_texture_count =
            u32::try_from(flat.measured_bsdfs.atlas_pages.len())
                .map_err(|_| PbrtError::error("Measured BSDF atlas page count exceeds u32."))?;
        let mut light_table =
            light_table_uniform(flat.lights.len(), flat.infinite_lights.len(), 0)?;
        material_table.debug_material_kind = INVALID_INDEX;
        if let Some(packed) = &packed_light_bvh {
            if light_sampler_kind == LightSamplerKind::Bvh {
                light_table.light_sampler_kind = super::abi::LIGHT_SAMPLER_KIND_BVH;
            }
            light_table.light_sampler_data_offset = 0;
            light_table.light_bvh_node_offset = 0;
            light_table.light_bvh_node_count =
                u32::try_from(packed.node_words.len()).map_err(|_| {
                    PbrtError::error("WebGPU Light BVH node count does not fit in u32.")
                })?;
            light_table.light_leaf_offset = 0;
            light_table.light_leaf_count =
                u32::try_from(packed.handle_to_leaf.len()).map_err(|_| {
                    PbrtError::error("WebGPU Light BVH leaf count does not fit in u32.")
                })?;
        }
        let (light_bvh_header, light_bvh_nodes, light_leaf) = packed_light_bvh
            .as_ref()
            .map(|packed| {
                (
                    packed.header_words.to_vec(),
                    packed
                        .node_words
                        .iter()
                        .flat_map(|node| node.iter().copied())
                        .collect::<Vec<_>>(),
                    packed.handle_to_leaf.clone(),
                )
            })
            .unwrap_or_default();
        let light_bvh_header_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 light BVH header SBO"),
                contents: buffer_contents(&light_bvh_header),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let light_bvh_node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 light BVH node SBO"),
            contents: buffer_contents(&light_bvh_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let light_leaf_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 light BVH leaf SBO"),
            contents: buffer_contents(&light_leaf),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let acceleration = acceleration::build(
            device,
            queue,
            &vertex_buffer,
            &index_buffer,
            &geometries,
            &instances,
            &flat.instances,
        )?;
        Ok(Self {
            camera,
            viewport,
            film,
            film_output_matrix,
            film_scale,
            material_table,
            light_table,
            sampler,
            output: Output::from_flat(flat.output),
            vertex_buffer,
            index_buffer,
            geometry_buffer,
            instance_buffer,
            material_root_buffer,
            material_node_buffer,
            attribute_ref_buffer,
            scalar_attribute_buffer,
            spectrum_attribute_buffer,
            measured_bsdf_buffer,
            measured_table_buffer,
            texture_root_buffer,
            texture_node_buffer,
            texture_child_buffer,
            rgb_spectrum_table_buffer,
            texture_images,
            texture_image_view,
            texture_image_views,
            texture_samplers,
            light_record_buffer,
            light_sampling_model_buffer,
            light_position_buffer,
            distribution_buffer,
            light_bvh_header_buffer,
            light_bvh_node_buffer,
            light_leaf_buffer,
            geometries,
            instances,
            material_nodes,
            light_sampling_models,
            light_records,
            light_sampler_kind,
            render_settings: RenderSettings::from_flat(flat.render_settings),
            acceleration,
        })
    }

    pub fn replace_material_kind(&mut self, queue: &wgpu::Queue, kind: MaterialKind) {
        self.material_table.debug_material_kind = kind.tag();
        for material in &mut self.material_nodes {
            material.kind = kind.tag();
        }
        queue.write_buffer(
            &self.material_node_buffer,
            0,
            bytemuck::cast_slice(&self.material_nodes),
        );
    }
}

fn buffer_contents<T: bytemuck::Pod>(values: &[T]) -> &[u8] {
    if values.is_empty() {
        // WebGPU validates the minimum binding size against the declared
        // storage-array stride. Keep one zeroed element for an empty runtime
        // array; a fixed byte count is insufficient for larger records such
        // as TextureNodeRecord and DenseSpectrum.
        static EMPTY_STORAGE_ELEMENT: [u8; 2048] = [0; 2048];
        let element_size = std::mem::size_of::<T>();
        assert!(element_size <= EMPTY_STORAGE_ELEMENT.len());
        &EMPTY_STORAGE_ELEMENT[..element_size]
    } else {
        cast_slice(values)
    }
}

fn validate_instance_area_lights(
    instance_index: usize,
    instance: &flat::Instance,
    flat: &flat::Scene,
) -> Result<(), PbrtError> {
    if instance.area_light == flat::INVALID_INDEX {
        return Ok(());
    }
    let geometry = flat
        .geometries
        .get(instance.geometry as usize)
        .ok_or_else(|| PbrtError::error("Flat area-light instance has an invalid geometry."))?;
    let triangle_count = geometry.index_count / 3;
    if triangle_count == 0 {
        return Err(PbrtError::error(
            "Flat area-light instance geometry contains no triangles.",
        ));
    }
    let handle = instance.area_light;
    let record = flat.lights.get(handle as usize).ok_or_else(|| {
        PbrtError::error(&format!(
            "Flat instance {instance_index} has an invalid area-light handle."
        ))
    })?;
    if record.kind != flat::LightKind::Area {
        return Err(PbrtError::error(&format!(
            "Flat instance {instance_index} area-light range contains a non-area light."
        )));
    }
    let model = flat
        .light_sampling_models
        .get(record.sampling_model as usize)
        .ok_or_else(|| {
            PbrtError::error(&format!(
                "Flat instance {instance_index} references an invalid area-light payload."
            ))
        })?;
    if model.geometry_index as usize != instance_index {
        return Err(PbrtError::error(&format!(
            "Flat instance {instance_index} area-light range does not match its triangles."
        )));
    }
    let offset = usize::try_from(model.distribution_offset)
        .map_err(|_| PbrtError::error("Flat area-light distribution offset does not fit usize."))?;
    let count = usize::try_from(model.distribution_count)
        .map_err(|_| PbrtError::error("Flat area-light distribution count does not fit usize."))?;
    if count == 0 {
        return Err(PbrtError::error(&format!(
            "Flat instance {instance_index} area-light distribution is empty."
        )));
    }
    let end = offset
        .checked_add(count)
        .ok_or_else(|| PbrtError::error("Flat area-light distribution range overflowed."))?;
    let entries = flat
        .triangle_distributions
        .get(offset..end)
        .ok_or_else(|| {
            PbrtError::error(&format!(
                "Flat instance {instance_index} area-light distribution range is invalid."
            ))
        })?;
    if !model.total_area.is_finite() || model.total_area <= 0.0 {
        return Err(PbrtError::error(&format!(
            "Flat instance {instance_index} area-light total area is invalid."
        )));
    }
    let mut previous_cdf = 0.0;
    let mut area_sum = 0.0;
    for entry in entries {
        if entry.primitive >= triangle_count
            || !entry.area.is_finite()
            || entry.area <= 0.0
            || !entry.cdf.is_finite()
            || entry.cdf < previous_cdf
            || entry.cdf > 1.0
        {
            return Err(PbrtError::error(&format!(
                "Flat instance {instance_index} has an invalid area-light distribution entry."
            )));
        }
        previous_cdf = entry.cdf;
        area_sum += entry.area;
    }
    if previous_cdf != 1.0 || (area_sum - model.total_area).abs() > model.total_area.abs() * 1e-5 {
        return Err(PbrtError::error(&format!(
            "Flat instance {instance_index} area-light distribution does not match total area."
        )));
    }
    Ok(())
}

fn convert_geometry(
    flat: &flat::Scene,
) -> Result<(Vec<Vertex>, Vec<Geometry>, Vec<u32>), PbrtError> {
    let vertices = flat
        .vertices
        .iter()
        .map(|vertex| {
            if !vertex.position.iter().all(|value| value.is_finite()) {
                return Err(PbrtError::error(
                    "Flat vertex position contains a non-finite value.",
                ));
            }
            if !vertex.uv.iter().all(|value| value.is_finite()) {
                return Err(PbrtError::error(
                    "Flat vertex UV contains a non-finite value.",
                ));
            }
            if !vertex.normal.iter().all(|value| value.is_finite())
                || !vertex.tangent.iter().all(|value| value.is_finite())
            {
                return Err(PbrtError::error(
                    "Flat vertex normal or tangent contains a non-finite value.",
                ));
            }
            Ok(Vertex {
                position: [
                    vertex.position[0],
                    vertex.position[1],
                    vertex.position[2],
                    1.0,
                ],
                normal: [vertex.normal[0], vertex.normal[1], vertex.normal[2], 0.0],
                tangent: [vertex.tangent[0], vertex.tangent[1], vertex.tangent[2], 0.0],
                uv: vertex.uv,
                padding: [0; 2],
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut local_indices = Vec::new();
    let mut geometries = Vec::with_capacity(flat.geometries.len());
    for (index, geometry) in flat.geometries.iter().enumerate() {
        let vertex_end = geometry
            .first_vertex
            .checked_add(geometry.vertex_count)
            .ok_or_else(|| {
                PbrtError::error(&format!("Flat geometry {index} vertex range overflowed."))
            })?;
        let index_end = geometry
            .first_index
            .checked_add(geometry.index_count)
            .ok_or_else(|| {
                PbrtError::error(&format!("Flat geometry {index} index range overflowed."))
            })?;
        if geometry.index_count == 0 || geometry.index_count % 3 != 0 {
            return Err(PbrtError::error(&format!(
                "Flat geometry {index} must contain a non-empty multiple of three indices."
            )));
        }
        if vertex_end as usize > vertices.len() || index_end as usize > flat.indices.len() {
            return Err(PbrtError::error(&format!(
                "Flat geometry {index} range is out of bounds."
            )));
        }
        let index_offset = u32::try_from(local_indices.len()).map_err(|_| {
            PbrtError::error(&format!(
                "Flat geometry {index} index offset does not fit in u32."
            ))
        })?;
        for &absolute_index in &flat.indices[geometry.first_index as usize..index_end as usize] {
            if absolute_index < geometry.first_vertex || absolute_index >= vertex_end {
                return Err(PbrtError::error(&format!(
                    "Flat geometry {index} contains an index outside its vertex range."
                )));
            }
            local_indices.push(absolute_index - geometry.first_vertex);
        }
        for triangle in local_indices[index_offset as usize..].chunks_exact(3) {
            let p0 = vertices[geometry.first_vertex as usize + triangle[0] as usize].position;
            let p1 = vertices[geometry.first_vertex as usize + triangle[1] as usize].position;
            let p2 = vertices[geometry.first_vertex as usize + triangle[2] as usize].position;
            let edge0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let edge1 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let cross = [
                edge0[1] * edge1[2] - edge0[2] * edge1[1],
                edge0[2] * edge1[0] - edge0[0] * edge1[2],
                edge0[0] * edge1[1] - edge0[1] * edge1[0],
            ];
            let norm_squared = cross.iter().map(|value| value * value).sum::<f32>();
            if !cross.iter().all(|value| value.is_finite())
                || !norm_squared.is_finite()
                || norm_squared == 0.0
            {
                return Err(PbrtError::error(&format!(
                    "Flat geometry {index} contains a zero-area or non-finite triangle."
                )));
            }
        }
        geometries.push(Geometry {
            vertex_offset: geometry.first_vertex,
            vertex_count: geometry.vertex_count,
            index_offset,
            index_count: geometry.index_count,
        });
    }
    Ok((vertices, geometries, local_indices))
}

#[cfg(test)]
mod tests {
    use super::{lower_texture_instruction, mip_level_rgba, texture_binding_plan};
    use crate::gpu::flat::texture::{
        ColorSpace, ImageFilterMode, ImageValueType, ImageView, ImageWrapMode, MipmapEncoding,
        MipmapLevel, MipmapLevelData, TextureInstruction, TextureValueType,
    };

    #[test]
    fn upload_rejects_unprojected_spectrum_mipmap() {
        let level = MipmapLevel {
            resolution: [1, 1],
            channels: 2,
            data: MipmapLevelData::F32(vec![0.2, 0.75]),
        };
        let error =
            mip_level_rgba(&level, ImageValueType::LinearRgb, MipmapEncoding::Linear).unwrap_err();
        assert!(error.to_string().contains("expected 3 channels"));
    }

    #[test]
    fn spectrum_upload_preserves_rgb_and_alpha() {
        let rgb = MipmapLevel {
            resolution: [1, 1],
            channels: 3,
            data: MipmapLevelData::F32(vec![0.0, 0.3, 0.6]),
        };
        let (_, _, rgba) =
            mip_level_rgba(&rgb, ImageValueType::LinearRgb, MipmapEncoding::Linear).unwrap();
        assert_eq!(rgba, vec![0.0, 0.3, 0.6, 1.0]);
    }

    #[test]
    fn upload_rejects_non_linear_flat_mipmap() {
        let rgba_level = MipmapLevel {
            resolution: [1, 1],
            channels: 3,
            data: MipmapLevelData::F32(vec![0.1, 0.2, 0.3]),
        };
        let error = mip_level_rgba(
            &rgba_level,
            ImageValueType::LinearRgb,
            MipmapEncoding::SrgbEncoded,
        )
        .unwrap_err();
        assert!(error.to_string().contains("requires linear"));
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
