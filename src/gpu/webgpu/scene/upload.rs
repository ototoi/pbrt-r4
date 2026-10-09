use std::sync::Arc;

use crate::gpu::flat;
use crate::gpu::flat::texture::{
    ImageValueType, ImageView, Mipmap, MipmapEncoding, MipmapLevel, MipmapLevelData,
};
use crate::util::error::PbrtError;

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

pub(super) fn upload_texture_images(
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

#[cfg(test)]
mod tests {
    use super::mip_level_rgba;

    use crate::gpu::flat::texture::{ImageValueType, MipmapEncoding, MipmapLevel, MipmapLevelData};

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
}
