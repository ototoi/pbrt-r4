use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::gpu::flat::{RenderSettings, SamplerKind, SamplerRandomization, MAX_GPU_RENDER_DEPTH};
use crate::samplers::pmj02bn::{BLUE_NOISE_BYTES, N_PMJ02BN_SAMPLES, PMJ02BN_BYTES};
use crate::samplers::{HaltonSampler, RandomizeStrategy};
use crate::util::base::Point2i;
use crate::util::error::PbrtError;
use crate::util::lowdiscrepancy::primes::PRIMES;
use crate::util::lowdiscrepancy::sobol::sobolmatrices::{
    SOBOL_MATRICES_32, SOBOL_MATRIX_SIZE, VDC_SOBOL_MATRICES, VDC_SOBOL_MATRICES_INV,
};
use crate::util::lowdiscrepancy::DigitPermutation;

use super::stages::ResourceId;

const TABLE_WIDTH: u32 = 256;
const HALTON_DIMENSION_COUNT: u32 = 6 + 8 * (MAX_GPU_RENDER_DEPTH + 1);
const WORDS_PER_TEXEL: usize = 4;
const HEADER_WORDS_PER_DIMENSION: usize = 4;
const INVALID_OFFSET: u32 = u32::MAX;
const SAMPLER_KIND_INDEPENDENT: u32 = 0;
const SAMPLER_KIND_HALTON: u32 = 1;
const SAMPLER_KIND_SOBOL: u32 = 2;
const SAMPLER_KIND_PADDED_SOBOL: u32 = 3;
const SAMPLER_KIND_Z_SOBOL: u32 = 4;
const SAMPLER_KIND_PMJ02BN: u32 = 5;
const SAMPLER_KIND_STRATIFIED: u32 = 6;
const SAMPLER_RANDOMIZATION_NONE: u32 = 0;
const SAMPLER_RANDOMIZATION_PERMUTE_DIGITS: u32 = 1;
const SAMPLER_RANDOMIZATION_FAST_OWEN: u32 = 2;
const SAMPLER_RANDOMIZATION_OWEN: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SamplerUniform {
    kind: u32,
    randomization: u32,
    table_width: u32,
    dimension_count: u32,
    samples_per_pixel: u32,
    seed: u32,
    padding: [u32; 2],
    variant_words: [[u32; 4]; 2],
}

pub const SAMPLER_UNIFORM_SIZE: usize = std::mem::size_of::<SamplerUniform>();
pub const SAMPLER_UNIFORM_VARIANT_WORDS_OFFSET: usize =
    std::mem::offset_of!(SamplerUniform, variant_words);

#[derive(Clone, Copy, Debug)]
enum SamplerParameters {
    Independent,
    Halton {
        base_scales: [u32; 2],
        base_exponents: [u32; 2],
        mult_inverse: [u32; 2],
    },
    Sobol {
        scale: u32,
        log2_samples_per_pixel: u32,
        n_base4_digits: u32,
        sobol_offset: u32,
        vdc_offset: u32,
        vdc_inverse_offset: u32,
    },
    Pmj02Bn {
        pmj_offset: u32,
        pmj_pixel_offset: u32,
        blue_noise_offset: u32,
        pixel_tile_size: u32,
    },
    Stratified {
        x_samples: u32,
        y_samples: u32,
        jitter: bool,
    },
}

#[derive(Clone, Copy, Debug)]
struct SamplerConfiguration {
    kind: SamplerKind,
    randomization: SamplerRandomization,
    samples_per_pixel: u32,
    seed: u32,
    parameters: SamplerParameters,
}

impl SamplerConfiguration {
    fn encode(self) -> SamplerUniform {
        let mut variant_words = [[0; 4]; 2];
        match self.parameters {
            SamplerParameters::Independent => {}
            SamplerParameters::Halton {
                base_scales,
                base_exponents,
                mult_inverse,
            } => {
                variant_words[0] = [
                    base_scales[0],
                    base_scales[1],
                    base_exponents[0],
                    base_exponents[1],
                ];
                variant_words[1][..2].copy_from_slice(&mult_inverse);
            }
            SamplerParameters::Sobol {
                scale,
                log2_samples_per_pixel,
                n_base4_digits,
                sobol_offset,
                vdc_offset,
                vdc_inverse_offset,
            } => {
                variant_words[0] = [scale, log2_samples_per_pixel, n_base4_digits, sobol_offset];
                variant_words[1] = [vdc_offset, vdc_inverse_offset, 0, 0];
            }
            SamplerParameters::Pmj02Bn {
                pmj_offset,
                pmj_pixel_offset,
                blue_noise_offset,
                pixel_tile_size,
            } => {
                variant_words[0] = [
                    pmj_offset,
                    pmj_pixel_offset,
                    blue_noise_offset,
                    pixel_tile_size,
                ];
            }
            SamplerParameters::Stratified {
                x_samples,
                y_samples,
                jitter,
            } => variant_words[0] = [x_samples, y_samples, u32::from(jitter), 0],
        }
        SamplerUniform {
            kind: match self.kind {
                SamplerKind::Independent => SAMPLER_KIND_INDEPENDENT,
                SamplerKind::Halton => SAMPLER_KIND_HALTON,
                SamplerKind::Sobol => SAMPLER_KIND_SOBOL,
                SamplerKind::PaddedSobol => SAMPLER_KIND_PADDED_SOBOL,
                SamplerKind::ZSobol => SAMPLER_KIND_Z_SOBOL,
                SamplerKind::Pmj02Bn => SAMPLER_KIND_PMJ02BN,
                SamplerKind::Stratified => SAMPLER_KIND_STRATIFIED,
            },
            randomization: match self.randomization {
                SamplerRandomization::None => SAMPLER_RANDOMIZATION_NONE,
                SamplerRandomization::PermuteDigits => SAMPLER_RANDOMIZATION_PERMUTE_DIGITS,
                SamplerRandomization::FastOwen => SAMPLER_RANDOMIZATION_FAST_OWEN,
                SamplerRandomization::Owen => SAMPLER_RANDOMIZATION_OWEN,
            },
            table_width: TABLE_WIDTH,
            dimension_count: HALTON_DIMENSION_COUNT,
            samples_per_pixel: self.samples_per_pixel,
            seed: self.seed,
            padding: [0; 2],
            variant_words,
        }
    }
}

pub struct SamplerResources {
    params_buffer: wgpu::Buffer,
    _table_texture: wgpu::Texture,
    table_view: wgpu::TextureView,
}

pub struct SamplerBindings<'a> {
    params_buffer: &'a wgpu::Buffer,
    table_view: &'a wgpu::TextureView,
}

impl<'a> SamplerBindings<'a> {
    pub fn resource(&self, resource: ResourceId) -> Option<wgpu::BindingResource<'a>> {
        match resource {
            ResourceId::SamplerParams => Some(self.params_buffer.as_entire_binding()),
            ResourceId::SamplerTable => Some(wgpu::BindingResource::TextureView(self.table_view)),
            _ => None,
        }
    }
}

pub struct SamplerData {
    configuration: SamplerConfiguration,
    words: Vec<u32>,
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
        let uniform = data.configuration.encode();
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
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 sampler UBO"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        log::info!(
            "GPU sampler resources: kind={:?}, dimensions={}, words={}, build+upload={:?}",
            settings.sampler_kind,
            HALTON_DIMENSION_COUNT,
            texels.len(),
            started.elapsed()
        );
        Ok(Self {
            params_buffer,
            _table_texture: table_texture,
            table_view,
        })
    }

    pub fn bindings(&self) -> SamplerBindings<'_> {
        SamplerBindings {
            params_buffer: &self.params_buffer,
            table_view: &self.table_view,
        }
    }
}

impl SamplerData {
    pub fn table_words(&self) -> &[u32] {
        &self.words
    }

    pub fn dimension_count(&self) -> u32 {
        HALTON_DIMENSION_COUNT
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
    let mut words = Vec::new();
    let parameters = match settings.sampler_kind {
        SamplerKind::Independent => SamplerParameters::Independent,
        SamplerKind::Halton => {
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
            if largest_index > u64::from(u32::MAX) {
                return Err(PbrtError::error(
                    "GPU Halton sample index exceeds the WebGPU u32 representation.",
                ));
            }
            words = pack_halton_table(
                HALTON_DIMENSION_COUNT,
                settings.randomization,
                settings.seed,
            );
            SamplerParameters::Halton {
                base_scales: halton.base_scales.map(|value| value as u32),
                base_exponents: halton.base_exponents.map(|value| value as u32),
                mult_inverse: halton.mult_inverse.map(|value| value as u32),
            }
        }
        SamplerKind::Sobol | SamplerKind::PaddedSobol | SamplerKind::ZSobol => {
            let max_resolution = resolution[0].max(resolution[1]).max(1);
            let scale = max_resolution
                .checked_next_power_of_two()
                .ok_or_else(|| PbrtError::error("GPU Sobol resolution scale overflowed u32."))?;
            let log2_scale = scale.ilog2();
            let log2_samples_per_pixel = settings.samples_per_pixel.ilog2();
            let n_base4_digits = log2_scale + log2_samples_per_pixel.div_ceil(2);
            let sobol_offset = append_sobol_matrices(&mut words, HALTON_DIMENSION_COUNT)?;
            let (vdc_offset, vdc_inverse_offset) = if settings.sampler_kind == SamplerKind::Sobol {
                if log2_scale > VDC_SOBOL_MATRICES.len() as u32 {
                    return Err(PbrtError::error(
                        "GPU Sobol resolution exceeds the interval matrix table.",
                    ));
                }
                if log2_scale == 0 {
                    (INVALID_OFFSET, INVALID_OFFSET)
                } else {
                    (
                        append_u64_words(&mut words, &VDC_SOBOL_MATRICES[log2_scale as usize - 1])?,
                        append_u64_words(
                            &mut words,
                            &VDC_SOBOL_MATRICES_INV[log2_scale as usize - 1],
                        )?,
                    )
                }
            } else {
                (INVALID_OFFSET, INVALID_OFFSET)
            };
            SamplerParameters::Sobol {
                scale,
                log2_samples_per_pixel,
                n_base4_digits,
                sobol_offset,
                vdc_offset,
                vdc_inverse_offset,
            }
        }
        SamplerKind::Pmj02Bn => {
            if settings.samples_per_pixel > N_PMJ02BN_SAMPLES as u32 {
                return Err(PbrtError::error(
                    "GPU PMJ02BN supports at most 65536 samples per pixel.",
                ));
            }
            let pmj_offset = append_bytes_as_u32(&mut words, PMJ02BN_BYTES)?;
            let (pixel_tile_size, pixel_words) =
                build_pmj_pixel_samples(settings.samples_per_pixel)?;
            let pmj_pixel_offset = append_words(&mut words, pixel_words)?;
            let blue_noise_offset = append_bytes_as_u32(&mut words, BLUE_NOISE_BYTES)?;
            SamplerParameters::Pmj02Bn {
                pmj_offset,
                pmj_pixel_offset,
                blue_noise_offset,
                pixel_tile_size,
            }
        }
        SamplerKind::Stratified => SamplerParameters::Stratified {
            x_samples: settings.x_samples,
            y_samples: settings.y_samples,
            jitter: settings.jitter,
        },
    };
    let configuration = SamplerConfiguration {
        kind: settings.sampler_kind,
        randomization: settings.randomization,
        samples_per_pixel: settings.samples_per_pixel,
        seed: settings.seed,
        parameters,
    };
    Ok(SamplerData {
        configuration,
        words,
    })
}

fn append_words(words: &mut Vec<u32>, values: Vec<u32>) -> Result<u32, PbrtError> {
    let offset = u32::try_from(words.len())
        .map_err(|_| PbrtError::error("GPU sampler table offset exceeds u32."))?;
    words.extend(values);
    Ok(offset)
}

fn append_sobol_matrices(words: &mut Vec<u32>, dimensions: u32) -> Result<u32, PbrtError> {
    let count = dimensions as usize * SOBOL_MATRIX_SIZE;
    append_words(words, SOBOL_MATRICES_32[..count].to_vec())
}

fn append_u64_words(words: &mut Vec<u32>, values: &[u64]) -> Result<u32, PbrtError> {
    let mut packed = Vec::with_capacity(values.len() * 2);
    for &value in values {
        packed.push(value as u32);
        packed.push((value >> 32) as u32);
    }
    append_words(words, packed)
}

fn append_bytes_as_u32(words: &mut Vec<u32>, bytes: &[u8]) -> Result<u32, PbrtError> {
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    append_words(words, values)
}

fn build_pmj_pixel_samples(samples_per_pixel: u32) -> Result<(u32, Vec<u32>), PbrtError> {
    let rounded = samples_per_pixel.next_power_of_two();
    let rounded_power_of_four = if rounded.ilog2() % 2 == 0 {
        rounded
    } else {
        rounded
            .checked_mul(2)
            .ok_or_else(|| PbrtError::error("GPU PMJ02BN rounded sample count overflowed u32."))?
    };
    let tile_size = 1u32 << (8 - rounded_power_of_four.ilog2() / 2);
    let sample_count = tile_size as usize * tile_size as usize * samples_per_pixel as usize;
    let mut samples = vec![[0.0f32; 2]; sample_count];
    let mut stored = vec![0u32; tile_size as usize * tile_size as usize];
    for index in 0..N_PMJ02BN_SAMPLES {
        let byte = index * 8;
        let x = u32::from_le_bytes(PMJ02BN_BYTES[byte..byte + 4].try_into().unwrap());
        let y = u32::from_le_bytes(PMJ02BN_BYTES[byte + 4..byte + 8].try_into().unwrap());
        let px = (x as f64 * (1.0 / 4294967296.0)) as f32 * tile_size as f32;
        let py = (y as f64 * (1.0 / 4294967296.0)) as f32 * tile_size as f32;
        let pixel_offset = px.floor() as usize + py.floor() as usize * tile_size as usize;
        if stored[pixel_offset] == samples_per_pixel {
            continue;
        }
        let sample_offset =
            pixel_offset * samples_per_pixel as usize + stored[pixel_offset] as usize;
        samples[sample_offset] = [px.fract(), py.fract()];
        stored[pixel_offset] += 1;
    }
    if stored.iter().any(|&count| count != samples_per_pixel) {
        return Err(PbrtError::error(
            "GPU PMJ02BN pixel sample table is incomplete.",
        ));
    }
    let packed = samples
        .into_iter()
        .flat_map(|sample| sample.map(f32::to_bits))
        .collect();
    Ok((tile_size, packed))
}

fn pack_halton_table(
    dimension_count: u32,
    randomization: SamplerRandomization,
    seed: u32,
) -> Vec<u32> {
    let header_words = dimension_count as usize * HEADER_WORDS_PER_DIMENSION;
    let mut headers = Vec::with_capacity(header_words);
    let mut permutations = Vec::new();
    for &base in PRIMES.iter().take(dimension_count as usize) {
        if randomization == SamplerRandomization::PermuteDigits {
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
