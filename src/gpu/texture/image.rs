//! Shared immutable image data used by texture graph and backend adapters.

use std::collections::HashMap;
use std::sync::Arc;

use crate::util::base::inverse_gamma_correct;
use crate::util::error::PbrtError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    Unknown,
    Srgb,
    Aces2065,
    DciP3,
    Rec2020,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MipmapEncoding {
    Linear,
    SrgbEncoded,
    U8Normalized,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MipmapLevelData {
    F32(Vec<f32>),
    F16(Vec<u16>),
    U8(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MipmapLevel {
    pub resolution: [u32; 2],
    pub channels: u32,
    pub data: MipmapLevelData,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mipmap {
    pub levels: Vec<MipmapLevel>,
    pub color_space: ColorSpace,
    pub encoding: MipmapEncoding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageValueType {
    Float,
    LinearRgb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageWrapMode {
    Repeat,
    Clamp,
    Black,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageFilterMode {
    Nearest,
    Bilinear,
    Trilinear,
}

/// Logical interpretation of shared image data.
///
/// Mapping remains a texture-program operation; this view owns only the
/// resource and sampling interpretation needed by backend adapters.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageView {
    pub mipmap: Arc<Mipmap>,
    pub value_type: ImageValueType,
    pub swrap: ImageWrapMode,
    pub twrap: ImageWrapMode,
    pub filter: ImageFilterMode,
    pub scale: f32,
    pub invert: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageOptimizationPolicy {
    pub allow_f16: bool,
    pub max_absolute_error: f32,
    pub max_relative_error: f32,
}

impl Default for ImageOptimizationPolicy {
    fn default() -> Self {
        Self {
            // GPU texture storage uses the bounded-error policy by default.
            // Values that do not satisfy both finite representation and this
            // tolerance remain F32, so high-dynamic-range inputs are not
            // silently clipped.
            allow_f16: true,
            max_absolute_error: 0.001,
            max_relative_error: 0.001,
        }
    }
}

/// Compiles one decoded GPU mipmap into the representation shared by Node and
/// Flat IR. The decoded image and its interpretation are interned separately
/// from image views, so sampling settings do not duplicate texel storage.
pub struct ImageCompiler {
    policy: ImageOptimizationPolicy,
    compiled: HashMap<(usize, ImageValueType), Arc<Mipmap>>,
}

impl Default for ImageCompiler {
    fn default() -> Self {
        Self::new(ImageOptimizationPolicy::default())
    }
}

impl ImageCompiler {
    pub fn new(policy: ImageOptimizationPolicy) -> Self {
        Self {
            policy,
            compiled: HashMap::new(),
        }
    }

    pub fn compile(
        &mut self,
        source: &Arc<Mipmap>,
        value_type: ImageValueType,
    ) -> Result<Arc<Mipmap>, PbrtError> {
        let key = (Arc::as_ptr(source) as usize, value_type);
        if let Some(mipmap) = self.compiled.get(&key) {
            return Ok(mipmap.clone());
        }
        let projected = match value_type {
            ImageValueType::Float => project_float_mipmap(source)?,
            ImageValueType::LinearRgb => source.clone(),
        };
        let optimized = self.optimize_storage(&projected)?;
        self.compiled.insert(key, optimized.clone());
        Ok(optimized)
    }

    fn optimize_storage(&self, source: &Arc<Mipmap>) -> Result<Arc<Mipmap>, PbrtError> {
        if !self.policy.allow_f16 {
            return Ok(source.clone());
        }
        let mut levels = Vec::with_capacity(source.levels.len());
        for level in &source.levels {
            let MipmapLevelData::F32(values) = &level.data else {
                levels.push(level.clone());
                continue;
            };
            let mut converted = Vec::with_capacity(values.len());
            let mut acceptable = true;
            for &value in values {
                if !value.is_finite() {
                    acceptable = false;
                    break;
                }
                let half = half::f16::from_f32(value);
                let round_trip = half.to_f32();
                if !round_trip.is_finite() {
                    acceptable = false;
                    break;
                }
                let absolute_error = (round_trip - value).abs();
                let relative_error = if value == 0.0 {
                    0.0
                } else {
                    absolute_error / value.abs()
                };
                if absolute_error > self.policy.max_absolute_error
                    && relative_error > self.policy.max_relative_error
                {
                    acceptable = false;
                    break;
                }
                converted.push(half.to_bits());
            }
            if acceptable {
                levels.push(MipmapLevel {
                    resolution: level.resolution,
                    channels: level.channels,
                    data: MipmapLevelData::F16(converted),
                });
            } else {
                levels.push(level.clone());
            }
        }
        Ok(Arc::new(Mipmap {
            levels,
            color_space: source.color_space,
            encoding: source.encoding,
        }))
    }
}

pub fn project_float_mipmap(mipmap: &Arc<Mipmap>) -> Result<Arc<Mipmap>, PbrtError> {
    let mut levels = Vec::with_capacity(mipmap.levels.len());
    for level in &mipmap.levels {
        if !(1..=4).contains(&level.channels) {
            return Err(PbrtError::error("Float texture has invalid channel count."));
        }
        let channels = usize::try_from(level.channels)
            .map_err(|_| PbrtError::error("Texture channel count does not fit usize."))?;
        let pixels = usize::try_from(level.resolution[0])
            .ok()
            .and_then(|width| {
                usize::try_from(level.resolution[1])
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or_else(|| PbrtError::error("Texture resolution overflowed."))?;
        let mut values = match &level.data {
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
        if values.len() != pixels.saturating_mul(channels) {
            return Err(PbrtError::error(
                "Texture mipmap data size is inconsistent.",
            ));
        }
        if matches!(mipmap.encoding, MipmapEncoding::SrgbEncoded) {
            for pixel in values.chunks_exact_mut(channels) {
                let color_channels = if channels == 4 { 3 } else { channels };
                for value in pixel.iter_mut().take(color_channels) {
                    *value = inverse_gamma_correct(*value);
                }
            }
        }
        let projected = values
            .chunks_exact(channels)
            .map(|pixel| {
                if channels == 4 {
                    pixel[3]
                } else if channels == 3 {
                    (pixel[0] + pixel[1] + pixel[2]) / 3.0
                } else {
                    pixel[0]
                }
            })
            .collect();
        levels.push(MipmapLevel {
            resolution: level.resolution,
            channels: 1,
            data: MipmapLevelData::F32(projected),
        });
    }
    Ok(Arc::new(Mipmap {
        levels,
        color_space: mipmap.color_space,
        encoding: MipmapEncoding::Linear,
    }))
}
