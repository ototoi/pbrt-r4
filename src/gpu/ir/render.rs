use super::{GpuBounds2i, GpuFloat, GpuIndex, GpuMatrix3x3, GpuMatrix4x4, GpuVector2, TransformId};

impl GpuBounds2i {
    pub fn area(self) -> Option<u64> {
        let width = u64::from(self.max[0]).checked_sub(u64::from(self.min[0]))?;
        let height = u64::from(self.max[1]).checked_sub(u64::from(self.min[1]))?;
        (width > 0 && height > 0).then(|| width.checked_mul(height))?
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuRenderRequestError {
    ZeroSampleCount,
    SampleRangeOverflow,
    SampleRangeExceedsSamplesPerPixel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRenderRequest {
    pub sample_start: u64,
    pub sample_count: u32,
}

impl GpuRenderRequest {
    pub fn new(
        render: &GpuRenderConfig,
        sample_start: u64,
        sample_count: u32,
    ) -> Result<Self, GpuRenderRequestError> {
        if sample_count == 0 {
            return Err(GpuRenderRequestError::ZeroSampleCount);
        }
        let sample_end = sample_start
            .checked_add(u64::from(sample_count))
            .ok_or(GpuRenderRequestError::SampleRangeOverflow)?;
        if sample_end > u64::from(render.sampler.samples_per_pixel) {
            return Err(GpuRenderRequestError::SampleRangeExceedsSamplesPerPixel);
        }
        Ok(Self {
            sample_start,
            sample_count,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuRenderOutput {
    pub pixel_bounds: GpuBounds2i,
    pub rgb: Box<[[f32; 3]]>,
    pub sample_start: u64,
    pub sample_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuRenderOutputError {
    InvalidPixelBounds,
    PixelCountMismatch { expected: usize, actual: usize },
}

impl GpuRenderOutput {
    pub fn new(
        pixel_bounds: GpuBounds2i,
        rgb: Box<[[f32; 3]]>,
        request: GpuRenderRequest,
    ) -> Result<Self, GpuRenderOutputError> {
        let expected = pixel_bounds
            .pixel_count()
            .filter(|count| *count > 0)
            .ok_or(GpuRenderOutputError::InvalidPixelBounds)?;
        if rgb.len() != expected {
            return Err(GpuRenderOutputError::PixelCountMismatch {
                expected,
                actual: rgb.len(),
            });
        }
        Ok(Self {
            pixel_bounds,
            rgb,
            sample_start: request.sample_start,
            sample_count: request.sample_count,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuPerspectiveCamera {
    pub render_from_camera: TransformId,
    pub camera_from_raster: GpuMatrix4x4,
    pub lens_radius: GpuFloat,
    pub focal_distance: GpuFloat,
    pub shutter_open: GpuFloat,
    pub shutter_close: GpuFloat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuIndependentSampler {
    pub samples_per_pixel: u32,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRgbFilm {
    pub full_resolution: [GpuIndex; 2],
    pub pixel_bounds: GpuBounds2i,
    pub diagonal_mm: GpuFloat,
    pub output_rgb_from_xyz: GpuMatrix3x3,
    pub iso: GpuFloat,
    pub max_component_value: GpuFloat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBoxFilter {
    pub radius: GpuVector2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuWavefrontVolPath {
    pub max_depth: u32,
    pub regularize: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuLightSampler {
    Uniform,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRenderConfig {
    pub camera: GpuPerspectiveCamera,
    pub sampler: GpuIndependentSampler,
    pub film: GpuRgbFilm,
    pub filter: GpuBoxFilter,
    pub integrator: GpuWavefrontVolPath,
    pub light_sampler: GpuLightSampler,
}
