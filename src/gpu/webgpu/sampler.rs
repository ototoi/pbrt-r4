use std::time::Instant;

use crate::gpu::flat::{HaltonRandomization, RenderSettings, SamplerKind, MAX_GPU_RENDER_DEPTH};
use crate::samplers::{HaltonSampler, RandomizeStrategy};
use crate::util::base::Point2i;
use crate::util::error::PbrtError;
use crate::util::lowdiscrepancy::primes::PRIMES;
use crate::util::lowdiscrepancy::DigitPermutation;

use super::abi::{
    SamplerUniform, HALTON_RANDOMIZATION_NONE, HALTON_RANDOMIZATION_PERMUTE_DIGITS,
    SAMPLER_KIND_HALTON, SAMPLER_KIND_INDEPENDENT,
};

const TABLE_WIDTH: u32 = 256;
const HALTON_DIMENSION_COUNT: u32 = 6 + 7 * (MAX_GPU_RENDER_DEPTH + 1);
const WORDS_PER_TEXEL: usize = 4;
const HEADER_WORDS_PER_DIMENSION: usize = 4;

pub struct SamplerResources {
    pub uniform: SamplerUniform,
    pub table_texture: wgpu::Texture,
    pub table_view: wgpu::TextureView,
}

pub struct SamplerData {
    pub uniform: SamplerUniform,
    pub words: Vec<u32>,
}

impl SamplerResources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: &RenderSettings,
        resolution: [u32; 2],
    ) -> Result<Self, PbrtError> {
        let started = Instant::now();
        let data = build_sampler_data(settings, resolution)?;
        let texel_count = data.words.len().div_ceil(WORDS_PER_TEXEL).max(1);
        let height = u32::try_from(texel_count.div_ceil(TABLE_WIDTH as usize))
            .map_err(|_| PbrtError::error("GPU sampler table height exceeds u32."))?;
        if height > device.limits().max_texture_dimension_2d {
            return Err(PbrtError::error(
                "GPU sampler table exceeds the maximum texture dimension.",
            ));
        }
        let mut texels = data.words;
        texels.resize(TABLE_WIDTH as usize * height as usize * WORDS_PER_TEXEL, 0);
        let table_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pbrt-r4 sampler table"),
            size: wgpu::Extent3d {
                width: TABLE_WIDTH,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            table_texture.as_image_copy(),
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TABLE_WIDTH * 16),
                rows_per_image: Some(height),
            },
            table_texture.size(),
        );
        let table_view = table_texture.create_view(&wgpu::TextureViewDescriptor::default());
        log::info!(
            "GPU sampler resources: kind={:?}, dimensions={}, words={}, build+upload={:?}",
            settings.sampler_kind,
            data.uniform.dimension_count,
            texels.len(),
            started.elapsed()
        );
        Ok(Self {
            uniform: data.uniform,
            table_texture,
            table_view,
        })
    }
}

pub fn build_sampler_data(
    settings: &RenderSettings,
    resolution: [u32; 2],
) -> Result<SamplerData, PbrtError> {
    if settings.samples_per_pixel == 0 {
        return Err(PbrtError::error(
            "GPU samples per pixel must be greater than zero.",
        ));
    }
    let full_resolution = Point2i::new(
        i32::try_from(resolution[0])
            .map_err(|_| PbrtError::error("GPU viewport width exceeds i32."))?,
        i32::try_from(resolution[1])
            .map_err(|_| PbrtError::error("GPU viewport height exceeds i32."))?,
    );
    let halton = HaltonSampler::new(
        settings.samples_per_pixel,
        full_resolution,
        RandomizeStrategy::None,
        settings.seed,
    );
    let sample_stride = u64::try_from(halton.base_scales[0] * halton.base_scales[1])
        .map_err(|_| PbrtError::error("GPU Halton sample stride is negative."))?;
    let largest_index = u64::from(settings.samples_per_pixel - 1)
        .checked_mul(sample_stride)
        .and_then(|index| index.checked_add(sample_stride.saturating_sub(1)))
        .ok_or_else(|| PbrtError::error("GPU Halton sample index overflowed u64."))?;
    if settings.sampler_kind == SamplerKind::Halton && largest_index > u64::from(u32::MAX) {
        return Err(PbrtError::error(
            "GPU Halton sample index exceeds the WebGPU u32 representation.",
        ));
    }
    let dimension_count = if settings.sampler_kind == SamplerKind::Halton {
        HALTON_DIMENSION_COUNT
    } else {
        0
    };
    let words = pack_halton_table(
        dimension_count,
        settings.halton_randomization,
        settings.seed,
    );
    let uniform = SamplerUniform {
        kind: match settings.sampler_kind {
            SamplerKind::Independent => SAMPLER_KIND_INDEPENDENT,
            SamplerKind::Halton => SAMPLER_KIND_HALTON,
        },
        randomization: match settings.halton_randomization {
            HaltonRandomization::None => HALTON_RANDOMIZATION_NONE,
            HaltonRandomization::PermuteDigits => HALTON_RANDOMIZATION_PERMUTE_DIGITS,
        },
        table_width: TABLE_WIDTH,
        dimension_count,
        base_scales: halton.base_scales.map(|value| value as u32),
        base_exponents: halton.base_exponents.map(|value| value as u32),
        mult_inverse: halton.mult_inverse.map(|value| value as u32),
        padding: [0; 2],
    };
    Ok(SamplerData { uniform, words })
}

fn pack_halton_table(
    dimension_count: u32,
    randomization: HaltonRandomization,
    seed: u32,
) -> Vec<u32> {
    let header_words = dimension_count as usize * HEADER_WORDS_PER_DIMENSION;
    let mut headers = Vec::with_capacity(header_words);
    let mut permutations = Vec::new();
    for &base in PRIMES.iter().take(dimension_count as usize) {
        if randomization == HaltonRandomization::PermuteDigits {
            let permutation = DigitPermutation::new(base as u32, seed);
            headers.extend_from_slice(&[
                permutation.base,
                permutation.n_digits,
                (header_words + permutations.len()) as u32,
                0,
            ]);
            permutations.extend(
                permutation
                    .permutations
                    .iter()
                    .map(|&value| u32::from(value)),
            );
        } else {
            headers.extend_from_slice(&[base as u32, 0, 0, 0]);
        }
    }
    headers.extend(permutations);
    headers
}
