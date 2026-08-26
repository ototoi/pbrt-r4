use super::{Bounds2i, Float, Index, Matrix3x3, Matrix4x4, TransformId, Vector2};
use crate::base::film::Film;
use crate::util::spectrum::{Spectrum, SpectrumType};

impl Bounds2i {
    pub fn area(self) -> Option<u64> {
        let width = u64::from(self.max[0]).checked_sub(u64::from(self.min[0]))?;
        let height = u64::from(self.max[1]).checked_sub(u64::from(self.min[1]))?;
        (width > 0 && height > 0).then(|| width.checked_mul(height))?
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderRequestError {
    ZeroSampleCount,
    SampleRangeOverflow,
    SampleRangeExceedsSamplesPerPixel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderRequest {
    pub sample_start: u64,
    pub sample_count: u32,
}

impl RenderRequest {
    pub fn new(
        render: &RenderConfig,
        sample_start: u64,
        sample_count: u32,
    ) -> Result<Self, RenderRequestError> {
        if sample_count == 0 {
            return Err(RenderRequestError::ZeroSampleCount);
        }
        let sample_end = sample_start
            .checked_add(u64::from(sample_count))
            .ok_or(RenderRequestError::SampleRangeOverflow)?;
        if sample_end > u64::from(render.sampler.samples_per_pixel) {
            return Err(RenderRequestError::SampleRangeExceedsSamplesPerPixel);
        }
        Ok(Self {
            sample_start,
            sample_count,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RenderOutput {
    pub pixel_bounds: Bounds2i,
    pub rgb: Box<[[f32; 3]]>,
    pub sample_start: u64,
    pub sample_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderOutputError {
    InvalidPixelBounds,
    PixelCountMismatch { expected: usize, actual: usize },
}

impl RenderOutput {
    pub fn new(
        pixel_bounds: Bounds2i,
        rgb: Box<[[f32; 3]]>,
        request: RenderRequest,
    ) -> Result<Self, RenderOutputError> {
        let expected = pixel_bounds
            .pixel_count()
            .filter(|count| *count > 0)
            .ok_or(RenderOutputError::InvalidPixelBounds)?;
        if rgb.len() != expected {
            return Err(RenderOutputError::PixelCountMismatch {
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

    /// Copies GPU RGB readback into the CPU film accumulator. The GPU path
    /// currently produces unbounded RGB values, so the conversion uses the
    /// same RGB-to-spectrum representation as the CPU film input path.
    pub fn write_to_film(&self, film: &mut Film) {
        let image: Vec<Spectrum> = self
            .rgb
            .iter()
            .map(|rgb| Spectrum::from_rgb(rgb, SpectrumType::Unbounded))
            .collect();
        film.set_image(&image);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerspectiveCamera {
    pub render_from_camera: TransformId,
    pub camera_from_raster: Matrix4x4,
    pub lens_radius: Float,
    pub focal_distance: Float,
    pub shutter_open: Float,
    pub shutter_close: Float,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndependentSampler {
    pub samples_per_pixel: u32,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RgbFilm {
    pub full_resolution: [Index; 2],
    pub pixel_bounds: Bounds2i,
    pub diagonal_mm: Float,
    pub output_rgb_from_xyz: Matrix3x3,
    pub iso: Float,
    pub max_component_value: Float,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxFilter {
    pub radius: Vector2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WavefrontVolPath {
    pub max_depth: u32,
    pub regularize: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightSampler {
    Uniform,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderConfig {
    pub camera: PerspectiveCamera,
    pub sampler: IndependentSampler,
    pub film: RgbFilm,
    pub filter: BoxFilter,
    pub integrator: WavefrontVolPath,
    pub light_sampler: LightSampler,
}
