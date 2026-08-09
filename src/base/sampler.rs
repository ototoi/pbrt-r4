use crate::base::camera::CameraSample;
use crate::options::PbrtOptions;
use crate::paramdict::*;
use crate::samplers::*;
use crate::util::base::*;
use crate::util::error::*;

/// Sampler enum that unifies all sampler types
///
/// This enum-based approach replaces the dynamic trait object pattern.
/// This corresponds to pbrt-v4's `TaggedPointer` sampler dispatch.
#[derive(Clone)]
pub enum Sampler {
    PMJ02BN(PMJ02BNSampler),
    Independent(IndependentSampler),
    Stratified(StratifiedSampler),
    Halton(HaltonSampler),
    Sobol(SobolSampler),
    ZSobol(ZSobolSampler),
    PaddedSobol(PaddedSobolSampler),
    MLT(MLTSampler),
}

impl Sampler {
    /// Create a sampler from name and parameters
    ///
    /// Corresponds to pbrt-v4's Sampler::Create
    ///
    /// # Arguments
    /// * `name` - Sampler type name (zsobol, halton, stratified, etc.)
    /// * `params` - Sampler parameters
    /// * `full_resolution` - Full image resolution
    ///
    /// # Returns
    /// * `Result<Sampler, PbrtError>` - Created sampler
    pub fn create(
        name: &str,
        params: &ParameterDictionary,
        full_resolution: Point2i,
    ) -> Result<Sampler, PbrtError> {
        let sampler = match name {
            "zsobol" => Sampler::ZSobol(ZSobolSampler::create(params, full_resolution)?),
            "paddedsobol" => {
                Sampler::PaddedSobol(PaddedSobolSampler::create(params, full_resolution)?)
            }
            "halton" => Sampler::Halton(HaltonSampler::create(params, full_resolution)?),
            "sobol" => Sampler::Sobol(SobolSampler::create(params, full_resolution)?),
            "pmj02bn" => Sampler::PMJ02BN(PMJ02BNSampler::create(params)?),
            "independent" => Sampler::Independent(IndependentSampler::create(params)?),
            "stratified" => Sampler::Stratified(StratifiedSampler::create(params)?),
            _ => {
                return Err(PbrtError::error(&format!(
                    "{}: sampler type unknown.",
                    name
                )));
            }
        };

        Ok(sampler)
    }

    /// Get number of samples per pixel
    pub fn samples_per_pixel(&self) -> u32 {
        match self {
            Sampler::PMJ02BN(s) => s.get_samples_per_pixel(),
            Sampler::Independent(s) => s.get_samples_per_pixel(),
            Sampler::Stratified(s) => s.get_samples_per_pixel(),
            Sampler::Halton(s) => s.get_samples_per_pixel(),
            Sampler::Sobol(s) => s.get_samples_per_pixel(),
            Sampler::ZSobol(s) => s.get_samples_per_pixel(),
            Sampler::PaddedSobol(s) => s.get_samples_per_pixel(),
            Sampler::MLT(s) => s.get_samples_per_pixel(),
        }
    }

    // MLT-specific methods
    pub fn start_iteration(&mut self) {
        if let Sampler::MLT(s) = self {
            s.start_iteration();
        }
    }

    pub fn accept(&mut self) {
        if let Sampler::MLT(s) = self {
            s.accept();
        }
    }

    pub fn reject(&mut self) {
        if let Sampler::MLT(s) = self {
            s.reject();
        }
    }

    pub fn start_stream(&mut self, index: u64) {
        if let Sampler::MLT(s) = self {
            s.start_stream(index);
        }
    }

    /// Start sampling for a pixel
    pub fn start_pixel(&mut self, p: &Point2i) {
        match self {
            Sampler::PMJ02BN(s) => s.start_pixel(p),
            Sampler::Independent(s) => s.start_pixel(p),
            Sampler::Stratified(s) => s.start_pixel(p),
            Sampler::Halton(s) => s.start_pixel(p),
            Sampler::Sobol(s) => s.start_pixel(p),
            Sampler::ZSobol(s) => s.start_pixel(p),
            Sampler::PaddedSobol(s) => s.start_pixel(p),
            Sampler::MLT(s) => s.start_pixel(p),
        }
    }

    /// pbrt-v4 `Sampler::StartPixelSample(p, sampleIndex, dimension)` --
    /// the canonical entry point for the tile-loop driver. Sets the
    /// current pixel, advances to `sample_index`, and (for LDS
    /// samplers that track an explicit per-sample dimension counter)
    /// seeds at `dimension`. Independent / Stratified / MLT don't carry
    /// a per-sample dimension
    /// state, so `dimension` is silently ignored there -- matches v4
    /// where those samplers' `StartPixelSample` overrides drop the
    /// argument too.
    pub fn start_pixel_sample(&mut self, p: Point2i, sample_index: u32, dimension: u32) {
        match self {
            Sampler::PMJ02BN(s) => {
                s.start_pixel(&p);
                s.set_sample_number(sample_index);
                let _ = dimension;
            }
            Sampler::Independent(s) => {
                s.start_pixel(&p);
                s.start_pixel_sample(sample_index, dimension);
            }
            Sampler::Stratified(s) => {
                s.start_pixel(&p);
                s.set_sample_number(sample_index);
                let _ = dimension;
            }
            Sampler::Halton(s) => {
                s.start_pixel(&p);
                s.start_pixel_sample(sample_index, dimension);
            }
            Sampler::Sobol(s) => {
                s.start_pixel(&p);
                s.start_pixel_sample(sample_index, dimension);
            }
            Sampler::ZSobol(s) => {
                s.start_pixel(&p);
                s.start_pixel_sample(sample_index, dimension);
            }
            Sampler::PaddedSobol(s) => {
                s.start_pixel(&p);
                s.start_pixel_sample(sample_index, dimension);
            }
            Sampler::MLT(s) => {
                // MLT has no per-sample seek; the chain advances itself
                // via start_iteration / accept / reject. The (pixel,
                // sample_index) tuple is bookkeeping only.
                s.start_pixel(&p);
                let _ = (sample_index, dimension);
            }
        }
    }

    /// pbrt-v4 `Sampler::GetPixel2D` -- the dedicated 2D sample for
    /// the film plane. Low-discrepancy samplers (Sobol / Halton /
    /// PaddedSobol / ZSobol) dedicate the first two dimensions to
    /// pixel coordinates so pixel sampling stays well-stratified
    /// across samples. Independent / Stratified / etc. simply forward
    /// to `get_2d`, matching v4 where their `GetPixel2D` is the
    /// default.
    pub fn get_pixel_2d(&mut self) -> Point2f {
        match self {
            Sampler::PMJ02BN(s) => s.get_pixel_2d(),
            Sampler::Independent(s) => s.get_pixel_2d(),
            Sampler::Stratified(s) => s.get_pixel_2d(),
            Sampler::Halton(s) => s.get_pixel_2d(),
            Sampler::Sobol(s) => s.get_pixel_2d(),
            Sampler::ZSobol(s) => s.get_pixel_2d(),
            Sampler::PaddedSobol(s) => s.get_pixel_2d(),
            Sampler::MLT(s) => s.get_pixel_2d(),
        }
    }

    /// Get a 1D sample
    pub fn get_1d(&mut self) -> Float {
        match self {
            Sampler::PMJ02BN(s) => s.get_1d(),
            Sampler::Independent(s) => s.get_1d(),
            Sampler::Stratified(s) => s.get_1d(),
            Sampler::Halton(s) => s.get_1d(),
            Sampler::Sobol(s) => s.get_1d(),
            Sampler::ZSobol(s) => s.get_1d(),
            Sampler::PaddedSobol(s) => s.get_1d(),
            Sampler::MLT(s) => s.get_1d(),
        }
    }

    /// Get a 2D sample
    pub fn get_2d(&mut self) -> Point2f {
        match self {
            Sampler::PMJ02BN(s) => s.get_2d(),
            Sampler::Independent(s) => s.get_2d(),
            Sampler::Stratified(s) => s.get_2d(),
            Sampler::Halton(s) => s.get_2d(),
            Sampler::Sobol(s) => s.get_2d(),
            Sampler::ZSobol(s) => s.get_2d(),
            Sampler::PaddedSobol(s) => s.get_2d(),
            Sampler::MLT(s) => s.get_2d(),
        }
    }

    /// Get a camera sample for the given pixel
    pub fn get_camera_sample(&mut self, p: &Point2i) -> CameraSample {
        if PbrtOptions::get().disable_pixel_jitter {
            return CameraSample {
                p_film: Point2f::new(p.x as Float + 0.5, p.y as Float + 0.5),
                time: 0.5,
                p_lens: Point2f::new(0.5, 0.5),
                filter_weight: 1.0,
            };
        }
        match self {
            Sampler::PMJ02BN(s) => s.get_camera_sample(p),
            Sampler::Independent(s) => s.get_camera_sample(p),
            Sampler::Stratified(s) => s.get_camera_sample(p),
            Sampler::Halton(s) => s.get_camera_sample(p),
            Sampler::Sobol(s) => s.get_camera_sample(p),
            Sampler::ZSobol(s) => s.get_camera_sample(p),
            Sampler::PaddedSobol(s) => s.get_camera_sample(p),
            Sampler::MLT(s) => s.get_camera_sample(p),
        }
    }

    /// Request a 1D sample array
    pub fn request_1d_array(&mut self, n: u32) {
        match self {
            Sampler::PMJ02BN(s) => s.request_1d_array(n),
            Sampler::Independent(s) => s.request_1d_array(n),
            Sampler::Stratified(s) => s.request_1d_array(n),
            Sampler::Halton(s) => s.request_1d_array(n),
            Sampler::Sobol(s) => s.request_1d_array(n),
            Sampler::ZSobol(s) => s.request_1d_array(n),
            Sampler::PaddedSobol(s) => s.request_1d_array(n),
            Sampler::MLT(s) => s.request_1d_array(n),
        }
    }

    /// Request a 2D sample array
    pub fn request_2d_array(&mut self, n: u32) {
        match self {
            Sampler::PMJ02BN(s) => s.request_2d_array(n),
            Sampler::Independent(s) => s.request_2d_array(n),
            Sampler::Stratified(s) => s.request_2d_array(n),
            Sampler::Halton(s) => s.request_2d_array(n),
            Sampler::Sobol(s) => s.request_2d_array(n),
            Sampler::ZSobol(s) => s.request_2d_array(n),
            Sampler::PaddedSobol(s) => s.request_2d_array(n),
            Sampler::MLT(s) => s.request_2d_array(n),
        }
    }

    /// Get a 1D sample array
    pub fn get_1d_array(&mut self, n: u32) -> Option<Vec<Float>> {
        match self {
            Sampler::PMJ02BN(s) => s.get_1d_array(n),
            Sampler::Independent(s) => s.get_1d_array(n),
            Sampler::Stratified(s) => s.get_1d_array(n),
            Sampler::Halton(s) => s.get_1d_array(n),
            Sampler::Sobol(s) => s.get_1d_array(n),
            Sampler::ZSobol(s) => s.get_1d_array(n),
            Sampler::PaddedSobol(s) => s.get_1d_array(n),
            Sampler::MLT(s) => s.get_1d_array(n),
        }
    }

    /// Get a 2D sample array
    pub fn get_2d_array(&mut self, n: u32) -> Option<Vec<Vector2f>> {
        match self {
            Sampler::PMJ02BN(s) => s.get_2d_array(n),
            Sampler::Independent(s) => s.get_2d_array(n),
            Sampler::Stratified(s) => s.get_2d_array(n),
            Sampler::Halton(s) => s.get_2d_array(n),
            Sampler::Sobol(s) => s.get_2d_array(n),
            Sampler::ZSobol(s) => s.get_2d_array(n),
            Sampler::PaddedSobol(s) => s.get_2d_array(n),
            Sampler::MLT(s) => s.get_2d_array(n),
        }
    }

    /// Start the next sample
    pub fn start_next_sample(&mut self) -> bool {
        match self {
            Sampler::PMJ02BN(s) => s.start_next_sample(),
            Sampler::Independent(s) => s.start_next_sample(),
            Sampler::Stratified(s) => s.start_next_sample(),
            Sampler::Halton(s) => s.start_next_sample(),
            Sampler::Sobol(s) => s.start_next_sample(),
            Sampler::ZSobol(s) => s.start_next_sample(),
            Sampler::PaddedSobol(s) => s.start_next_sample(),
            Sampler::MLT(s) => s.start_next_sample(),
        }
    }

    /// Set the sample number
    pub fn set_sample_number(&mut self, sample_num: u32) -> bool {
        match self {
            Sampler::PMJ02BN(s) => s.set_sample_number(sample_num),
            Sampler::Independent(s) => s.set_sample_number(sample_num),
            Sampler::Stratified(s) => s.set_sample_number(sample_num),
            Sampler::Halton(s) => s.set_sample_number(sample_num),
            Sampler::Sobol(s) => s.set_sample_number(sample_num),
            Sampler::ZSobol(s) => s.set_sample_number(sample_num),
            Sampler::PaddedSobol(s) => s.set_sample_number(sample_num),
            Sampler::MLT(_) => false,
        }
    }

    /// Round sample count
    pub fn round_count(&self, n: u32) -> u32 {
        match self {
            Sampler::PMJ02BN(_) => n,
            Sampler::Independent(_) => n,
            Sampler::Stratified(_) => n,
            Sampler::Halton(_) => n,
            Sampler::Sobol(_) => n,
            Sampler::ZSobol(_) => n,
            Sampler::PaddedSobol(_) => n,
            Sampler::MLT(_) => n,
        }
    }

    /// Get number of samples per pixel (alternative name)
    pub fn get_samples_per_pixel(&self) -> u32 {
        self.samples_per_pixel()
    }
}
