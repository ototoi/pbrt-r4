//! Reference evaluator for the backend-independent typed texture program.

use super::{
    ImageFilterMode, ImageValueType, ImageView, ImageWrapMode, MipmapEncoding, MipmapLevel,
    MipmapLevelData,
};
use crate::gpu::node::{TextureMapping, UvMapping};
use crate::util::base::inverse_gamma_correct;
use crate::util::error::PbrtError;

use super::program::{Instruction, TypedTextureProgram};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TextureValue {
    Float(f32),
    LinearRgb([f32; 3]),
}

pub fn evaluate_texture_program(program: &TypedTextureProgram) -> Result<TextureValue, PbrtError> {
    evaluate_texture_program_at(program, [0.0, 0.0])
}

pub fn evaluate_texture_program_at(
    program: &TypedTextureProgram,
    uv: [f32; 2],
) -> Result<TextureValue, PbrtError> {
    let mut values = Vec::with_capacity(program.slot_types.len());
    for instruction in &program.instructions {
        let value = match instruction {
            Instruction::ConstantFloat { value, .. } => TextureValue::Float(*value),
            Instruction::ConstantRgb { value, .. } => TextureValue::LinearRgb(*value),
            Instruction::Scale { input, factor, .. } => match values.get(*input as usize) {
                Some(TextureValue::Float(value)) => TextureValue::Float(value * factor),
                Some(TextureValue::LinearRgb(value)) => {
                    TextureValue::LinearRgb(value.map(|value| value * factor))
                }
                None => return Err(invalid_slot(*input)),
            },
            Instruction::Mix {
                first,
                second,
                amount,
                constant_amount,
                ..
            } => {
                let amount = match amount {
                    Some(slot) => match values.get(*slot as usize) {
                        Some(TextureValue::Float(value)) => *value,
                        Some(TextureValue::LinearRgb(_)) => {
                            return Err(PbrtError::error(
                                "Texture mix amount must be a float value.",
                            ));
                        }
                        None => return Err(invalid_slot(*slot)),
                    },
                    None => *constant_amount,
                };
                match (values.get(*first as usize), values.get(*second as usize)) {
                    (Some(TextureValue::Float(first)), Some(TextureValue::Float(second))) => {
                        TextureValue::Float(first * (1.0 - amount) + second * amount)
                    }
                    (
                        Some(TextureValue::LinearRgb(first)),
                        Some(TextureValue::LinearRgb(second)),
                    ) => TextureValue::LinearRgb(std::array::from_fn(|index| {
                        first[index] * (1.0 - amount) + second[index] * amount
                    })),
                    (Some(_), Some(_)) => {
                        return Err(PbrtError::error(
                            "Texture mix operands must have matching value types.",
                        ));
                    }
                    (None, _) => return Err(invalid_slot(*first)),
                    (_, None) => return Err(invalid_slot(*second)),
                }
            }
            Instruction::SampleImage {
                image_view,
                mapping,
                ..
            } => {
                let view = program
                    .image_views
                    .get(*image_view as usize)
                    .ok_or_else(|| invalid_slot(*image_view))?;
                let mipmap = program
                    .mipmaps
                    .get(view.mipmap as usize)
                    .ok_or_else(|| PbrtError::error("Image view has an invalid mipmap."))?;
                sample_image(view, mipmap, mapping.as_ref(), uv)?
            }
            Instruction::Procedural { name, .. } => {
                return Err(PbrtError::error(&format!(
                    "Reference texture evaluator does not support procedural texture \"{name}\" yet."
                )));
            }
        };
        values.push(value);
    }
    values
        .get(program.result as usize)
        .copied()
        .ok_or_else(|| PbrtError::error("Texture program has no result value."))
}

fn sample_image(
    view: &ImageView,
    mipmap: &super::image::Mipmap,
    mapping: Option<&TextureMapping>,
    uv: [f32; 2],
) -> Result<TextureValue, PbrtError> {
    let uv = match mapping {
        None => uv,
        Some(TextureMapping::Uv(UvMapping {
            uscale,
            vscale,
            udelta,
            vdelta,
        })) => [uv[0] * *uscale + *udelta, uv[1] * *vscale + *vdelta],
        Some(_) => {
            return Err(PbrtError::error(
                "Reference texture evaluator only supports UV image mapping.",
            ))
        }
    };
    let level = mipmap
        .levels
        .first()
        .ok_or_else(|| PbrtError::error("Image texture has no mipmap levels."))?;
    let width = level.resolution[0] as usize;
    let height = level.resolution[1] as usize;
    if width == 0 || height == 0 || !(1..=4).contains(&level.channels) {
        return Err(PbrtError::error("Image texture has invalid dimensions."));
    }
    let value = match view.filter {
        ImageFilterMode::Nearest | ImageFilterMode::Trilinear => {
            let u = wrap_coordinate(uv[0], view.swrap)?;
            let v = wrap_coordinate(uv[1], view.twrap)?;
            sample_texel(level, &mipmap.encoding, u, v, view)?
        }
        ImageFilterMode::Bilinear => {
            let u = wrap_coordinate(uv[0], view.swrap)?;
            let v = wrap_coordinate(uv[1], view.twrap)?;
            let x = u * width as f32 - 0.5;
            let y = v * height as f32 - 0.5;
            let x0 = x.floor();
            let y0 = y.floor();
            let tx = x - x0;
            let ty = y - y0;
            let a = sample_indexed(level, &mipmap.encoding, x0, y0, view)?;
            let b = sample_indexed(level, &mipmap.encoding, x0 + 1.0, y0, view)?;
            let c = sample_indexed(level, &mipmap.encoding, x0, y0 + 1.0, view)?;
            let d = sample_indexed(level, &mipmap.encoding, x0 + 1.0, y0 + 1.0, view)?;
            lerp_value(lerp_value(a, b, tx), lerp_value(c, d, tx), ty)
        }
    };
    Ok(apply_image_transform(value, view.scale, view.invert))
}

fn wrap_coordinate(value: f32, mode: ImageWrapMode) -> Result<f32, PbrtError> {
    match mode {
        ImageWrapMode::Repeat => Ok(value - value.floor()),
        ImageWrapMode::Clamp => Ok(value.clamp(0.0, 1.0)),
        ImageWrapMode::Black if !(0.0..=1.0).contains(&value) => Ok(0.0),
        ImageWrapMode::Black => Ok(value),
    }
}

fn sample_texel(
    level: &MipmapLevel,
    encoding: &MipmapEncoding,
    u: f32,
    v: f32,
    view: &ImageView,
) -> Result<TextureValue, PbrtError> {
    let x = (u * level.resolution[0] as f32).floor();
    let y = (v * level.resolution[1] as f32).floor();
    sample_indexed(level, encoding, x, y, view)
}

fn sample_indexed(
    level: &MipmapLevel,
    encoding: &MipmapEncoding,
    x: f32,
    y: f32,
    view: &ImageView,
) -> Result<TextureValue, PbrtError> {
    let width = level.resolution[0] as i32;
    let height = level.resolution[1] as i32;
    let pixel = (y as i32).clamp(0, height - 1) * width + (x as i32).clamp(0, width - 1);
    let offset = usize::try_from(pixel).unwrap_or(0) * level.channels as usize;
    let channels = level.channels as usize;
    let mut values = [0.0; 4];
    for channel in 0..channels {
        values[channel] = match &level.data {
            MipmapLevelData::F32(data) => *data.get(offset + channel).ok_or_else(invalid_data)?,
            MipmapLevelData::F16(data) => {
                half::f16::from_bits(*data.get(offset + channel).ok_or_else(invalid_data)?).to_f32()
            }
            MipmapLevelData::U8(data) => {
                f32::from(*data.get(offset + channel).ok_or_else(invalid_data)?) / 255.0
            }
        };
    }
    if matches!(encoding, MipmapEncoding::SrgbEncoded) {
        for value in values
            .iter_mut()
            .take(if channels == 4 { 3 } else { channels })
        {
            *value = inverse_gamma_correct(*value);
        }
    }
    Ok(match view.value_type {
        ImageValueType::Float => TextureValue::Float(values[0]),
        ImageValueType::LinearRgb => TextureValue::LinearRgb(if channels < 3 {
            [values[0]; 3]
        } else {
            [values[0], values[1], values[2]]
        }),
    })
}

fn invalid_data() -> PbrtError {
    PbrtError::error("Image mipmap data is inconsistent with its resolution.")
}

fn lerp_value(first: TextureValue, second: TextureValue, amount: f32) -> TextureValue {
    match (first, second) {
        (TextureValue::Float(first), TextureValue::Float(second)) => {
            TextureValue::Float(first * (1.0 - amount) + second * amount)
        }
        (TextureValue::LinearRgb(first), TextureValue::LinearRgb(second)) => {
            TextureValue::LinearRgb(std::array::from_fn(|index| {
                first[index] * (1.0 - amount) + second[index] * amount
            }))
        }
        _ => first,
    }
}

fn apply_image_transform(value: TextureValue, scale: f32, invert: bool) -> TextureValue {
    let transform = |value: f32| (if invert { 1.0 - value } else { value }) * scale;
    match value {
        TextureValue::Float(value) => TextureValue::Float(transform(value)),
        TextureValue::LinearRgb(value) => TextureValue::LinearRgb(value.map(transform)),
    }
}

fn invalid_slot(slot: u32) -> PbrtError {
    PbrtError::error(&format!("Texture program references invalid slot {slot}."))
}
