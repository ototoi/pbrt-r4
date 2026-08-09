use crate::base::camera::CameraSample;
use crate::options::*;
use crate::paramdict::*;
use crate::samplers::*;

use crate::util::base::*;
use crate::util::error::*;
use crate::util::lowdiscrepancy::sobol::sobolmatrices::{SOBOL_MATRICES_32, SOBOL_MATRIX_SIZE};
use crate::util::lowdiscrepancy::*;
use crate::util::profile::*;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct SobolSampler {
    pub base: BaseSampler,
    pub samples_per_pixel: u32,
    pub scale: u32,
    pub seed: u32,
    pub randomize: RandomizeStrategy,
    pub pixel: Point2i,
    pub dimension: u32,
    pub sobol_index: i64,
}

impl SobolSampler {
    pub fn new(
        samples_per_pixel: u32,
        full_resolution: Point2i,
        randomize: RandomizeStrategy,
        seed: u32,
    ) -> Self {
        let scale = round_up_pow2(u32::max(full_resolution.x as u32, full_resolution.y as u32));
        Self {
            base: BaseSampler::new(samples_per_pixel),
            samples_per_pixel,
            scale,
            seed,
            randomize,
            pixel: Point2i::zero(),
            dimension: 2,
            sobol_index: 0,
        }
    }

    pub fn start_pixel(&mut self, p: &Point2i) {
        self.base.start_pixel(p);
        self.pixel = *p;
        self.fill_sample_arrays();
        self.start_pixel_sample(0, 0);
    }

    pub fn get_1d(&mut self) -> Float {
        let _p = ProfilePhase::new(Prof::GetSample);
        if self.dimension >= n_sobol_dimensions() {
            self.dimension = 2;
        }
        let dim = self.dimension;
        self.dimension += 1;
        self.sample_dimension(dim)
    }

    pub fn get_2d(&mut self) -> Point2f {
        let _p = ProfilePhase::new(Prof::GetSample);
        if self.dimension + 1 >= n_sobol_dimensions() {
            self.dimension = 2;
        }
        let u = Point2f::new(
            self.sample_dimension(self.dimension),
            self.sample_dimension(self.dimension + 1),
        );
        self.dimension += 2;
        u
    }

    pub fn get_pixel_2d(&mut self) -> Point2f {
        let mut u = Point2f::new(
            sobol_sample(self.sobol_index, 0, 0),
            sobol_sample(self.sobol_index, 1, 0),
        );
        for dim in 0..2 {
            u[dim] = Float::clamp(
                u[dim] * self.scale as Float - self.pixel[dim] as Float,
                0.0,
                ONE_MINUS_EPSILON,
            );
        }
        u
    }

    pub fn get_camera_sample(&mut self, p_raster: &Point2i) -> CameraSample {
        CameraSample {
            p_film: Point2f::new(p_raster.x as Float, p_raster.y as Float) + self.get_pixel_2d(),
            time: self.get_1d(),
            p_lens: self.get_2d(),
            filter_weight: 1.0,
        }
    }

    pub fn request_1d_array(&mut self, n: u32) {
        self.base.request_1d_array(n);
    }

    pub fn request_2d_array(&mut self, n: u32) {
        self.base.request_2d_array(n);
    }

    pub fn get_1d_array(&mut self, n: u32) -> Option<Vec<Float>> {
        self.base.get_1d_array(n)
    }

    pub fn get_2d_array(&mut self, n: u32) -> Option<Vec<Vector2f>> {
        self.base.get_2d_array(n)
    }

    pub fn start_next_sample(&mut self) -> bool {
        let ok = self.base.start_next_sample();
        if ok {
            self.start_pixel_sample(self.base.current_pixel_sample_index, 0);
        }
        ok
    }

    pub fn set_sample_number(&mut self, sample_num: u32) -> bool {
        let ok = self.base.set_sample_number(sample_num);
        if ok {
            self.start_pixel_sample(sample_num, 0);
        }
        ok
    }

    pub fn get_samples_per_pixel(&self) -> u32 {
        self.samples_per_pixel
    }

    pub fn create(
        params: &ParameterDictionary,
        full_resolution: Point2i,
    ) -> Result<SobolSampler, PbrtError> {
        let mut nsamp = params.get_one_int("pixelsamples", 16) as u32;
        let randomize = parse_randomize_strategy(
            params.get_one_string("randomization", "fastowen"),
            "SobolSampler",
        )?;
        let seed = params.get_one_int("seed", PbrtOptions::get().seed as i32) as u32;
        {
            let options = PbrtOptions::get();
            if options.quick_render {
                nsamp = 1;
            }
        }
        Ok(SobolSampler::new(nsamp, full_resolution, randomize, seed))
    }

    pub fn start_pixel_sample(&mut self, sample_index: u32, dim: u32) {
        self.pixel = self.base.current_pixel;
        self.dimension = u32::max(2, dim);
        self.sobol_index =
            sobol_interval_to_index(log2int(self.scale), sample_index as u64, &self.pixel) as i64;
    }

    fn sample_dimension(&self, dimension: u32) -> Float {
        sample_dimension_with_strategy(self.sobol_index, dimension, self.randomize, self.seed)
    }

    fn fill_sample_arrays(&mut self) {
        let mut dim = 5u32;
        for i in 0..self.base.samples1d_array_sizes.len() {
            let samples = self.samples_per_pixel * self.base.samples1d_array_sizes[i];
            for j in 0..samples {
                let index =
                    sobol_interval_to_index(log2int(self.scale), j as u64, &self.pixel) as i64;
                self.base.sample_array1d[i][j as usize] =
                    sample_dimension_with_strategy(index, dim, self.randomize, self.seed);
            }
            dim += 1;
        }
        for i in 0..self.base.samples2d_array_sizes.len() {
            let samples = self.samples_per_pixel * self.base.samples2d_array_sizes[i];
            for j in 0..samples {
                let index =
                    sobol_interval_to_index(log2int(self.scale), j as u64, &self.pixel) as i64;
                self.base.sample_array2d[i][j as usize] = Point2f::new(
                    sample_dimension_with_strategy(index, dim, self.randomize, self.seed),
                    sample_dimension_with_strategy(index, dim + 1, self.randomize, self.seed),
                );
            }
            dim += 2;
        }
    }
}

fn parse_randomize_strategy(
    value: String,
    sampler_name: &str,
) -> Result<RandomizeStrategy, PbrtError> {
    match value.as_str() {
        "none" => Ok(RandomizeStrategy::None),
        "permutedigits" => Ok(RandomizeStrategy::PermuteDigits),
        "fastowen" => Ok(RandomizeStrategy::FastOwen),
        "owen" => Ok(RandomizeStrategy::Owen),
        _ => Err(PbrtError::error(&format!(
            "{}: unknown randomization strategy {}",
            sampler_name, value
        ))),
    }
}

fn n_sobol_dimensions() -> u32 {
    (SOBOL_MATRICES_32.len() / SOBOL_MATRIX_SIZE) as u32
}

fn sample_dimension_with_strategy(
    index: i64,
    dimension: u32,
    randomize: RandomizeStrategy,
    seed: u32,
) -> Float {
    match randomize {
        RandomizeStrategy::None => sobol_sample(index, dimension, 0),
        RandomizeStrategy::PermuteDigits => {
            let hash = mix_bits(((dimension as u64) << 32) | seed as u64) as u32;
            sobol_sample(index, dimension, hash as u64)
        }
        RandomizeStrategy::FastOwen => randomized_sobol_sample(
            index,
            dimension,
            fast_owen_scramble,
            mix_bits(((dimension as u64) << 32) | seed as u64) as u32,
        ),
        RandomizeStrategy::Owen => randomized_sobol_sample(
            index,
            dimension,
            owen_scramble,
            mix_bits(((dimension as u64) << 32) | seed as u64) as u32,
        ),
    }
}

fn randomized_sobol_sample(
    a: i64,
    dimension: u32,
    randomizer: fn(u32, u32) -> u32,
    seed: u32,
) -> Float {
    let mut a = a;
    let mut v: u32 = 0;
    let mut i = usize::min(
        dimension as usize * SOBOL_MATRIX_SIZE,
        SOBOL_MATRICES_32.len() - 1,
    );
    while a != 0 {
        if (a & 1) != 0 {
            v ^= SOBOL_MATRICES_32[i] as u32;
        }
        a >>= 1;
        i += 1;
        i %= SOBOL_MATRICES_32.len();
    }
    let v = randomizer(v, seed);
    Float::min(
        (v as f64 * 2.3283064365386963e-10f64) as Float,
        FLOAT_ONE_MINUS_EPSILON as Float,
    )
}

fn fast_owen_scramble(mut v: u32, seed: u32) -> u32 {
    v = reverse_bits32(v);
    v ^= v.wrapping_mul(0x3d20adea);
    v = v.wrapping_add(seed);
    v = v.wrapping_mul((seed >> 16) | 1);
    v ^= v.wrapping_mul(0x05526c56);
    v ^= v.wrapping_mul(0x53a22864);
    reverse_bits32(v)
}

fn owen_scramble(mut v: u32, seed: u32) -> u32 {
    if (seed & 1) != 0 {
        v ^= 1u32 << 31;
    }
    for b in 1..32 {
        let mask = (!0u32) << (32 - b);
        if (mix_bits(((v & mask) ^ seed) as u64) as u32 & (1u32 << b)) != 0 {
            v ^= 1u32 << (31 - b);
        }
    }
    v
}

fn mix_bits(mut v: u64) -> u64 {
    v ^= v >> 31;
    v = v.wrapping_mul(0x7fb5d329728ea185);
    v ^= v >> 27;
    v = v.wrapping_mul(0x81dadef4bc2dd44d);
    v ^= v >> 33;
    v
}
