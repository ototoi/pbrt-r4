//! Decoded and optimized image data owned by Flat IR.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::util::base::inverse_gamma_correct;
use crate::util::error::PbrtError;
use crate::util::imageio::{read_raw_image_with_encoding, ColorEncoding};

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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DecodedImageKey {
    path: std::path::PathBuf,
    encoding: String,
}

/// Decodes resolved Node IR image references and creates their full mip chains
/// at the Node-to-Flat seam.
#[derive(Default)]
pub struct ImageDecoder {
    decoded: HashMap<DecodedImageKey, Arc<Mipmap>>,
}

impl ImageDecoder {
    pub fn decode(&mut self, path: &Path, encoding_name: &str) -> Result<Arc<Mipmap>, PbrtError> {
        let path = path.canonicalize().map_err(|error| {
            PbrtError::error(&format!(
                "Unable to resolve texture image \"{}\": {error}",
                path.display()
            ))
        })?;
        let key = DecodedImageKey {
            path: path.clone(),
            encoding: encoding_name.to_string(),
        };
        if let Some(mipmap) = self.decoded.get(&key) {
            return Ok(mipmap.clone());
        }

        let (raw, color_space) = if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("exr"))
        {
            let (raw, _, metadata) = crate::util::imageio::read_image_exr::read_raw_image_exr_with_channels_and_metadata(&path)?;
            let color_space = metadata
                .color_space
                .map(|space| match space.name {
                    "ACES2065-1" => ColorSpace::Aces2065,
                    "DCI-P3" => ColorSpace::DciP3,
                    "Rec2020" => ColorSpace::Rec2020,
                    _ => ColorSpace::Srgb,
                })
                .unwrap_or(ColorSpace::Srgb);
            (raw, color_space)
        } else {
            let encoding = ColorEncoding::parse(encoding_name)?;
            (
                read_raw_image_with_encoding(&path.to_string_lossy(), encoding)?,
                ColorSpace::Srgb,
            )
        };
        let channels = u32::try_from(raw.channels)
            .map_err(|_| PbrtError::error("Texture channel count exceeds u32."))?;
        let mut resolution = [raw.resolution.x as u32, raw.resolution.y as u32];
        let mut data = raw.data_f32();
        let mut levels = Vec::new();
        loop {
            levels.push(MipmapLevel {
                resolution,
                channels,
                data: MipmapLevelData::F32(data.clone()),
            });
            if resolution == [1, 1] {
                break;
            }
            let next_resolution = [(resolution[0] / 2).max(1), (resolution[1] / 2).max(1)];
            let channel_count = channels as usize;
            let mut next =
                vec![
                    0.0;
                    next_resolution[0] as usize * next_resolution[1] as usize * channel_count
                ];
            for y in 0..next_resolution[1] {
                for x in 0..next_resolution[0] {
                    let mut count = 0.0f32;
                    for oy in 0..2 {
                        for ox in 0..2 {
                            let sx = (2 * x + ox).min(resolution[0] - 1);
                            let sy = (2 * y + oy).min(resolution[1] - 1);
                            let source_offset = (sy * resolution[0] + sx) as usize * channel_count;
                            let target = (y * next_resolution[0] + x) as usize * channel_count;
                            for channel in 0..channel_count {
                                next[target + channel] += data[source_offset + channel];
                            }
                            count += 1.0;
                        }
                    }
                    let target = (y * next_resolution[0] + x) as usize * channel_count;
                    for channel in 0..channel_count {
                        next[target + channel] /= count;
                    }
                }
            }
            resolution = next_resolution;
            data = next;
        }
        let mipmap = Arc::new(Mipmap {
            levels,
            color_space,
            encoding: MipmapEncoding::Linear,
        });
        self.decoded.insert(key, mipmap.clone());
        Ok(mipmap)
    }
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
    pub mipmap: u32,
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

/// Compiles one decoded mipmap into optimized Flat IR storage. The decoded
/// image and its interpretation are interned separately from image views, so
/// sampling settings do not duplicate texel storage.
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
        validate_mipmap(source)?;
        let key = (Arc::as_ptr(source) as usize, value_type);
        if let Some(mipmap) = self.compiled.get(&key) {
            return Ok(mipmap.clone());
        }
        let projected = match value_type {
            ImageValueType::Float => project_float_mipmap(source)?,
            ImageValueType::LinearRgb => project_linear_rgb_mipmap(source)?,
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
                return Ok(source.clone());
            };
            let mut converted = Vec::with_capacity(values.len());
            for &value in values {
                if !value.is_finite() {
                    return Ok(source.clone());
                }
                let half = half::f16::from_f32(value);
                let round_trip = half.to_f32();
                if !round_trip.is_finite() {
                    return Ok(source.clone());
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
                    return Ok(source.clone());
                }
                converted.push(half.to_bits());
            }
            levels.push(MipmapLevel {
                resolution: level.resolution,
                channels: level.channels,
                data: MipmapLevelData::F16(converted),
            });
        }
        Ok(Arc::new(Mipmap {
            levels,
            color_space: source.color_space,
            encoding: source.encoding,
        }))
    }
}

pub fn validate_mipmap(mipmap: &Mipmap) -> Result<(), PbrtError> {
    let Some(first) = mipmap.levels.first() else {
        return Err(PbrtError::error("Texture mipmap has no levels."));
    };
    if first.resolution[0] == 0 || first.resolution[1] == 0 {
        return Err(PbrtError::error(
            "Texture mipmap has a zero-sized base level.",
        ));
    }
    if !(1..=4).contains(&first.channels) {
        return Err(PbrtError::error(
            "Texture mipmap has an invalid channel count.",
        ));
    }
    let storage = mipmap_storage(&first.data);
    let mut expected_resolution = first.resolution;
    for (index, level) in mipmap.levels.iter().enumerate() {
        if level.resolution != expected_resolution {
            return Err(PbrtError::error(&format!(
                "Texture mipmap level {index} has resolution {:?}, expected {:?}.",
                level.resolution, expected_resolution
            )));
        }
        if level.channels != first.channels {
            return Err(PbrtError::error(
                "Texture mipmap levels have inconsistent channel counts.",
            ));
        }
        if mipmap_storage(&level.data) != storage {
            return Err(PbrtError::error(
                "Texture mipmap levels have inconsistent storage formats.",
            ));
        }
        let expected_len = usize::try_from(level.resolution[0])
            .ok()
            .and_then(|width| {
                usize::try_from(level.resolution[1])
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixels| pixels.checked_mul(level.channels as usize))
            .ok_or_else(|| PbrtError::error("Texture mipmap data size overflowed."))?;
        if mipmap_data_len(&level.data) != expected_len {
            return Err(PbrtError::error(&format!(
                "Texture mipmap level {index} has an inconsistent data size."
            )));
        }
        expected_resolution = [
            (expected_resolution[0] / 2).max(1),
            (expected_resolution[1] / 2).max(1),
        ];
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MipmapStorage {
    F32,
    F16,
    U8,
}

fn mipmap_storage(data: &MipmapLevelData) -> MipmapStorage {
    match data {
        MipmapLevelData::F32(_) => MipmapStorage::F32,
        MipmapLevelData::F16(_) => MipmapStorage::F16,
        MipmapLevelData::U8(_) => MipmapStorage::U8,
    }
}

fn mipmap_data_len(data: &MipmapLevelData) -> usize {
    match data {
        MipmapLevelData::F32(values) => values.len(),
        MipmapLevelData::F16(values) => values.len(),
        MipmapLevelData::U8(values) => values.len(),
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

pub fn project_linear_rgb_mipmap(mipmap: &Arc<Mipmap>) -> Result<Arc<Mipmap>, PbrtError> {
    let mut levels = Vec::with_capacity(mipmap.levels.len());
    for level in &mipmap.levels {
        if !(1..=4).contains(&level.channels) {
            return Err(PbrtError::error(
                "Spectrum texture has invalid channel count.",
            ));
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
        let values = mipmap_level_values(level);
        if values.len() != pixels.saturating_mul(channels) {
            return Err(PbrtError::error(
                "Texture mipmap data size is inconsistent.",
            ));
        }
        let mut projected = Vec::with_capacity(pixels.saturating_mul(3));
        for pixel in values.chunks_exact(channels) {
            let mut rgb = if channels <= 2 {
                [pixel[0]; 3]
            } else {
                [pixel[0], pixel[1], pixel[2]]
            };
            if matches!(mipmap.encoding, MipmapEncoding::SrgbEncoded) {
                rgb = rgb.map(inverse_gamma_correct);
            }
            projected.extend_from_slice(&rgb);
        }
        levels.push(MipmapLevel {
            resolution: level.resolution,
            channels: 3,
            data: MipmapLevelData::F32(projected),
        });
    }
    Ok(Arc::new(Mipmap {
        levels,
        color_space: mipmap.color_space,
        encoding: MipmapEncoding::Linear,
    }))
}

fn mipmap_level_values(level: &MipmapLevel) -> Vec<f32> {
    match &level.data {
        MipmapLevelData::F32(values) => values.clone(),
        MipmapLevelData::F16(values) => values
            .iter()
            .map(|value| half::f16::from_bits(*value).to_f32())
            .collect(),
        MipmapLevelData::U8(values) => values
            .iter()
            .map(|value| f32::from(*value) / 255.0)
            .collect(),
    }
}
