use crate::base::camera::CameraSample;
use crate::options::*;
use crate::paramdict::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::lowdiscrepancy::{murmur_hash_64a, permutation_element};
use crate::util::rng::RNG;

#[derive(Debug, PartialEq, Default, Clone)]
pub struct StratifiedSampler {
    x_pixel_samples: u32,
    y_pixel_samples: u32,
    seed: u32,
    jitter_samples: bool,
    pixel: Point2i,
    sample_index: u32,
    dimension: u32,
    rng: RNG,
}

impl StratifiedSampler {
    pub fn new(
        x_pixel_samples: u32,
        y_pixel_samples: u32,
        jitter_samples: bool,
        seed: u32,
    ) -> Self {
        StratifiedSampler {
            x_pixel_samples,
            y_pixel_samples,
            seed,
            jitter_samples,
            pixel: Point2i::zero(),
            sample_index: 0,
            dimension: 0,
            rng: RNG::new(),
        }
    }

    pub fn start_pixel(&mut self, p: &Point2i) {
        self.start_pixel_sample(*p, 0, 0);
    }

    pub fn get_1d(&mut self) -> Float {
        let hash = hash_pixel_dimension_seed(&self.pixel, self.dimension, self.seed);
        let stratum =
            permutation_element(self.sample_index, self.get_samples_per_pixel(), hash as u32);
        self.dimension += 1;
        let delta = if self.jitter_samples {
            self.rng.uniform_float()
        } else {
            0.5
        };
        (stratum as Float + delta) / self.get_samples_per_pixel() as Float
    }

    pub fn get_2d(&mut self) -> Point2f {
        let hash = hash_pixel_dimension_seed(&self.pixel, self.dimension, self.seed);
        let stratum =
            permutation_element(self.sample_index, self.get_samples_per_pixel(), hash as u32);
        self.dimension += 2;
        let delta = if self.jitter_samples {
            Point2f::new(self.rng.uniform_float(), self.rng.uniform_float())
        } else {
            Point2f::new(0.5, 0.5)
        };
        let x = stratum % self.x_pixel_samples;
        let y = stratum / self.x_pixel_samples;
        Point2f::new(
            (x as Float + delta.x) / self.x_pixel_samples as Float,
            (y as Float + delta.y) / self.y_pixel_samples as Float,
        )
    }

    pub fn get_pixel_2d(&mut self) -> Point2f {
        self.get_2d()
    }

    pub fn get_camera_sample(&mut self, p_raster: &Point2i) -> CameraSample {
        CameraSample {
            p_film: Point2f::new(p_raster.x as Float, p_raster.y as Float) + self.get_pixel_2d(),
            time: self.get_1d(),
            p_lens: self.get_2d(),
            filter_weight: 1.0,
        }
    }

    pub fn request_1d_array(&mut self, _n: u32) {}

    pub fn request_2d_array(&mut self, _n: u32) {}

    pub fn get_1d_array(&mut self, _n: u32) -> Option<Vec<Float>> {
        None
    }

    pub fn get_2d_array(&mut self, _n: u32) -> Option<Vec<Vector2f>> {
        None
    }

    pub fn start_next_sample(&mut self) -> bool {
        self.set_sample_number(self.sample_index + 1)
    }

    pub fn set_sample_number(&mut self, sample_num: u32) -> bool {
        self.start_pixel_sample(self.pixel, sample_num, 0);
        sample_num < self.get_samples_per_pixel()
    }

    pub fn get_samples_per_pixel(&self) -> u32 {
        self.x_pixel_samples * self.y_pixel_samples
    }

    pub fn start_pixel_sample(&mut self, pixel: Point2i, sample_index: u32, dimension: u32) {
        self.pixel = pixel;
        self.sample_index = sample_index;
        self.dimension = dimension;
        self.rng.set_sequence(hash_pixel_seed(&pixel, self.seed));
        self.rng
            .advance((u64::from(sample_index) * 65536 + u64::from(dimension)) as i64);
    }
}

impl StratifiedSampler {
    pub fn create(params: &ParameterDictionary) -> Result<StratifiedSampler, PbrtError> {
        let jitter = params.get_one_bool("jitter", true);
        let mut xsamp = params.get_one_int("xsamples", 4) as u32;
        let mut ysamp = params.get_one_int("ysamples", 4) as u32;
        let seed = params.get_one_int("seed", PbrtOptions::get().seed as i32) as u32;
        {
            let options = PbrtOptions::get();
            if options.quick_render {
                xsamp = 1;
                ysamp = 1;
            }
        }
        Ok(StratifiedSampler::new(xsamp, ysamp, jitter, seed))
    }
}

fn hash_pixel_dimension_seed(pixel: &Point2i, dimension: u32, seed: u32) -> u64 {
    let mut buf = [0u8; 16];
    buf[0..4].copy_from_slice(&pixel.x.to_ne_bytes());
    buf[4..8].copy_from_slice(&pixel.y.to_ne_bytes());
    buf[8..12].copy_from_slice(&dimension.to_ne_bytes());
    buf[12..16].copy_from_slice(&seed.to_ne_bytes());
    murmur_hash_64a(&buf, 0)
}

fn hash_pixel_seed(pixel: &Point2i, seed: u32) -> u64 {
    let mut buf = [0u8; 12];
    buf[0..4].copy_from_slice(&pixel.x.to_ne_bytes());
    buf[4..8].copy_from_slice(&pixel.y.to_ne_bytes());
    buf[8..12].copy_from_slice(&seed.to_ne_bytes());
    murmur_hash_64a(&buf, 0)
}
