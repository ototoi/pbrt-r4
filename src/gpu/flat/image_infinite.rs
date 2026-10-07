use crate::util::base::inverse_gamma_correct;
use crate::util::error::PbrtError;

use super::texture::{Mipmap, MipmapEncoding, MipmapLevelData};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageInfiniteDistributionTexel {
    pub weight: f32,
    pub conditional_cdf: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageInfiniteDistribution {
    pub resolution: [u32; 2],
    pub texels: Vec<ImageInfiniteDistributionTexel>,
    pub row_cdf: Vec<f32>,
    pub total_integral: f32,
}

impl ImageInfiniteDistribution {
    pub fn from_mipmap(mipmap: &Mipmap) -> Result<Self, PbrtError> {
        let level = mipmap
            .levels
            .first()
            .ok_or_else(|| PbrtError::error("Image infinite light has no mipmap levels."))?;
        let pixel_count = usize::try_from(level.resolution[0])
            .ok()
            .and_then(|width| {
                usize::try_from(level.resolution[1])
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or_else(|| PbrtError::error("Image infinite light resolution overflowed."))?;
        let channels = usize::try_from(level.channels)
            .map_err(|_| PbrtError::error("Image infinite light channel count is invalid."))?;
        if channels < 3 {
            return Err(PbrtError::error(
                "Image infinite light image does not have RGB channels.",
            ));
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
        if values.len() != pixel_count.saturating_mul(channels) {
            return Err(PbrtError::error(
                "Image infinite light mipmap data has an inconsistent size.",
            ));
        }
        let rgb = values
            .chunks_exact(channels)
            .map(|pixel| {
                let mut rgb = [pixel[0], pixel[1], pixel[2]];
                if matches!(mipmap.encoding, MipmapEncoding::SrgbEncoded) {
                    rgb = rgb.map(inverse_gamma_correct);
                }
                rgb
            })
            .collect::<Vec<_>>();
        Self::from_rgb(level.resolution[0], level.resolution[1], &rgb)
    }

    pub fn from_rgb(width: u32, height: u32, rgb: &[[f32; 3]]) -> Result<Self, PbrtError> {
        if width == 0 || height == 0 {
            return Err(PbrtError::error(
                "Image infinite light distribution has an empty resolution.",
            ));
        }
        let pixel_count = width
            .checked_mul(height)
            .and_then(|count| usize::try_from(count).ok())
            .ok_or_else(|| {
                PbrtError::error("Image infinite light distribution size overflowed.")
            })?;
        if rgb.len() != pixel_count {
            return Err(PbrtError::error(
                "Image infinite light distribution RGB data has an inconsistent size.",
            ));
        }
        if rgb.iter().flatten().any(|value| !value.is_finite()) {
            return Err(PbrtError::error(
                "Image infinite light distribution contains a non-finite value.",
            ));
        }

        let mut weights = rgb
            .iter()
            .map(|pixel| (pixel[0] + pixel[1] + pixel[2]) / 3.0)
            .collect::<Vec<_>>();
        let average = (weights.iter().map(|&value| f64::from(value)).sum::<f64>()
            / weights.len() as f64) as f32;
        for weight in &mut weights {
            *weight = (*weight - average).max(0.0);
        }
        if weights.iter().all(|&weight| weight == 0.0) {
            weights.fill(1.0);
        }
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err(PbrtError::error(
                "Image infinite light sampling weights are non-finite.",
            ));
        }

        let width_usize = width as usize;
        let height_usize = height as usize;
        let mut texels = Vec::with_capacity(pixel_count);
        let mut row_cdf = Vec::with_capacity(height_usize);
        let mut marginal_integral = 0.0f32;
        for row in weights.chunks_exact(width_usize) {
            let mut conditional_cdf = 0.0f32;
            for &weight in row {
                conditional_cdf += weight / width as f32;
                texels.push(ImageInfiniteDistributionTexel {
                    weight,
                    conditional_cdf,
                });
            }
            marginal_integral += conditional_cdf / height as f32;
            row_cdf.push(marginal_integral);
        }
        let total_integral = marginal_integral;
        if !total_integral.is_finite() {
            return Err(PbrtError::error(
                "Image infinite light sampling integral is non-finite.",
            ));
        }

        Ok(Self {
            resolution: [width, height],
            texels,
            row_cdf,
            total_integral,
        })
    }

    pub fn sample(&self, u: [f32; 2]) -> Option<[f32; 2]> {
        if u.iter()
            .any(|value| !value.is_finite() || !(0.0..1.0).contains(value))
        {
            return None;
        }
        let [width, height] = self.resolution;
        if self.total_integral <= 0.0 || self.row_cdf.is_empty() {
            return None;
        }

        let row_total = *self.row_cdf.last()?;
        let row_target = u[1] * row_total;
        let row = cdf_interval(&self.row_cdf, row_target)?;
        let row_start = if row == 0 { 0.0 } else { self.row_cdf[row - 1] };
        let row_mass = self.row_cdf[row] - row_start;
        if row_mass <= 0.0 {
            return None;
        }
        let row_offset = row * width as usize;
        let row_texels = &self.texels[row_offset..row_offset + width as usize];
        let conditional_total = row_texels.last()?.conditional_cdf;
        if conditional_total <= 0.0 {
            return None;
        }
        let conditional_target = u[0] * conditional_total;
        let column = cdf_interval_by(row_texels.len(), conditional_target, |index| {
            row_texels[index].conditional_cdf
        })?;
        let column_start = if column == 0 {
            0.0
        } else {
            row_texels[column - 1].conditional_cdf
        };
        let column_mass = row_texels[column].conditional_cdf - column_start;
        if column_mass <= 0.0 {
            return None;
        }
        let du = ((conditional_target - column_start) / column_mass).clamp(0.0, 1.0);
        let dv = ((row_target - row_start) / row_mass).clamp(0.0, 1.0);
        Some([
            (column as f32 + du) / width as f32,
            (row as f32 + dv) / height as f32,
        ])
    }

    pub fn pdf(&self, uv: [f32; 2]) -> f32 {
        if uv
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            || self.total_integral <= 0.0
        {
            return 0.0;
        }
        let [width, height] = self.resolution;
        let x = ((uv[0] * width as f32) as u32).min(width - 1) as usize;
        let y = ((uv[1] * height as f32) as u32).min(height - 1) as usize;
        self.texels[y * width as usize + x].weight / self.total_integral
    }
}

fn cdf_interval(cdf: &[f32], target: f32) -> Option<usize> {
    cdf_interval_by(cdf.len(), target, |index| cdf[index])
}

fn cdf_interval_by(count: usize, target: f32, cumulative: impl Fn(usize) -> f32) -> Option<usize> {
    if count == 0 {
        return None;
    }
    let mut first = 0;
    let mut last = count;
    while first < last {
        let middle = first + (last - first) / 2;
        if cumulative(middle) <= target {
            first = middle + 1;
        } else {
            last = middle;
        }
    }
    Some(first.min(count - 1))
}
