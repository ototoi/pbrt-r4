use std::sync::Arc;

use crate::gpu::flat;
use crate::gpu::flat::texture::{
    validate_mipmap, ImageValueType, ImageView, Mipmap, MipmapEncoding, MipmapLevel,
    MipmapLevelData,
};
use crate::util::error::PbrtError;

/// Validated format and byte layout for one Flat IR image.
#[derive(Clone, Copy, Debug)]
pub struct ImageUploadFormat {
    pub format: wgpu::TextureFormat,
    pub bytes_per_texel: u32,
}

impl ImageUploadFormat {
    pub fn select(mipmap: &Mipmap, value_type: ImageValueType) -> Result<Self, PbrtError> {
        validate_mipmap(mipmap)?;
        let level = &mipmap.levels[0];
        let max_levels = 32 - level.resolution[0].max(level.resolution[1]).leading_zeros();
        if mipmap.levels.len() > max_levels as usize {
            return Err(PbrtError::error("Texture mipmap has too many levels."));
        }
        let channels = match value_type {
            ImageValueType::Float => 1,
            ImageValueType::LinearRgb => 3,
        };
        if level.channels != channels {
            return Err(PbrtError::error(&format!(
                "WebGPU texture upload expected {channels} channels, got {}.",
                level.channels
            )));
        }
        use wgpu::TextureFormat as F;
        let format = match (&level.data, channels, mipmap.encoding) {
            (MipmapLevelData::F16(_), 1, MipmapEncoding::Linear) => F::R16Float,
            (MipmapLevelData::F16(_), 3, MipmapEncoding::Linear) => F::Rgba16Float,
            (MipmapLevelData::F32(_), 1, MipmapEncoding::Linear) => F::R32Float,
            (MipmapLevelData::F32(_), 3, MipmapEncoding::Linear) => F::Rgba32Float,
            (MipmapLevelData::U8(_), 1, MipmapEncoding::Linear | MipmapEncoding::U8Normalized) => {
                F::R8Unorm
            }
            (MipmapLevelData::U8(_), 3, MipmapEncoding::Linear | MipmapEncoding::U8Normalized) => {
                F::Rgba8Unorm
            }
            (MipmapLevelData::U8(_), 3, MipmapEncoding::SrgbEncoded) => F::Rgba8UnormSrgb,
            _ => {
                return Err(PbrtError::error(
                    "Unsupported WebGPU texture storage/encoding combination.",
                ))
            }
        };
        let bytes_per_texel = match format {
            F::R8Unorm => 1,
            F::R16Float => 2,
            F::R32Float | F::Rgba8Unorm | F::Rgba8UnormSrgb => 4,
            F::Rgba16Float => 8,
            F::Rgba32Float => 16,
            _ => return Err(PbrtError::error("Invalid image upload format.")),
        };
        Ok(Self {
            format,
            bytes_per_texel,
        })
    }

    pub fn bytes_per_row(&self, width: u32) -> Result<u32, PbrtError> {
        width
            .checked_mul(self.bytes_per_texel)
            .ok_or_else(|| PbrtError::error("Texture row byte size overflowed."))
    }
}

/// Packs validated image storage without changing its precision or encoding.
pub fn pack_image_level(level: &MipmapLevel) -> Vec<u8> {
    fn pad<T: Copy>(values: &[T], channels: u32, alpha: T) -> Vec<T> {
        if channels == 1 {
            return values.to_vec();
        }
        values
            .chunks_exact(3)
            .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], alpha])
            .collect()
    }
    match &level.data {
        MipmapLevelData::F32(values) if level.channels == 1 => {
            bytemuck::cast_slice(values).to_vec()
        }
        MipmapLevelData::F16(values) if level.channels == 1 => {
            bytemuck::cast_slice(values).to_vec()
        }
        MipmapLevelData::F32(values) => {
            bytemuck::cast_slice(&pad(values, level.channels, 1.0)).to_vec()
        }
        MipmapLevelData::F16(values) => {
            bytemuck::cast_slice(&pad(values, level.channels, half::f16::ONE.to_bits())).to_vec()
        }
        MipmapLevelData::U8(values) => pad(values, level.channels, 255),
    }
}

pub fn upload_texture_images(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mipmaps: &[Arc<Mipmap>],
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
        let upload = ImageUploadFormat::select(mipmap, view.value_type)?;
        let [width, height] = base_level.resolution;
        let features = upload.format.guaranteed_format_features(device.features());
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        if !features.allowed_usages.contains(usage)
            || !features
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
        {
            return Err(PbrtError::error(
                "Device does not support the required filterable image format.",
            ));
        }
        let mip_level_count = u32::try_from(mipmap.levels.len())
            .map_err(|_| PbrtError::error("Texture mipmap has too many levels."))?;
        let logical_bytes = mipmap.levels.iter().try_fold(0u64, |total, level| {
            let row = upload.bytes_per_row(level.resolution[0])?;
            total
                .checked_add(u64::from(row) * u64::from(level.resolution[1]))
                .ok_or_else(|| PbrtError::error("Texture byte size overflowed."))
        })?;
        log::debug!(
            "GPU image: {width}x{height}, {} levels, {:?}, {logical_bytes} texel bytes",
            mipmap.levels.len(),
            upload.format
        );
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
            format: upload.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (mip_level, level) in mipmap.levels.iter().enumerate() {
            let [level_width, level_height] = level.resolution;
            let bytes = pack_image_level(level);
            let mip_level = u32::try_from(mip_level)
                .map_err(|_| PbrtError::error("Texture mipmap level index overflowed."))?;
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.bytes_per_row(level_width)?),
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

pub(super) fn upload_measured_atlas(
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
