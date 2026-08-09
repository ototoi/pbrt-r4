use crate::base::camera::CameraSample;
use crate::options::*;
use crate::paramdict::*;
use crate::samplers::*;

use crate::util::base::*;
use crate::util::error::*;
use crate::util::lowdiscrepancy::primes::PRIMES;
use crate::util::lowdiscrepancy::DigitPermutation;
use crate::util::lowdiscrepancy::*;
use crate::util::profile::*;
use crate::util::sampling::*;

use std::sync::Arc;

const K_MAX_RESOLUTION: i32 = 128;

/// `digit_permutations` is ~7 MB (`Vec<u16>` with one entry per
/// prime in `PRIMES` summed); cloning a `HaltonSampler` per rayon
/// work-item deep-copied that table, peaking at gigabytes of RSS.
/// Wrap it in `Arc` so clones share the storage.
#[derive(Debug, Default, PartialEq, Clone)]
pub struct HaltonSampler {
    pub base: BaseSampler,
    pub samples_per_pixel: u32,
    pub randomize: RandomizeStrategy,
    pub digit_permutations: Arc<Vec<DigitPermutation>>,
    pub base_scales: [i32; 2],
    pub base_exponents: [i32; 2],
    pub mult_inverse: [i32; 2],
    pub halton_index: i64,
    pub dimension: u32,
}

impl HaltonSampler {
    pub fn new(
        samples_per_pixel: u32,
        full_resolution: Point2i,
        randomize: RandomizeStrategy,
        seed: u32,
    ) -> Self {
        let digit_permutations =
            Arc::new(if matches!(randomize, RandomizeStrategy::PermuteDigits) {
                compute_radical_inverse_permutations_with_seed(seed)
            } else {
                Vec::new()
            });

        let mut base_scales = [1, 1];
        let mut base_exponents = [0, 0];
        for i in 0..2 {
            let base = if i == 0 { 2 } else { 3 };
            let mut scale = 1;
            let mut exp = 0;
            while scale < i32::min(full_resolution[i], K_MAX_RESOLUTION) {
                scale *= base;
                exp += 1;
            }
            base_scales[i] = scale;
            base_exponents[i] = exp;
        }

        let mult_inverse = [
            multiplicative_inverse(base_scales[1] as i64, base_scales[0] as i64) as i32,
            multiplicative_inverse(base_scales[0] as i64, base_scales[1] as i64) as i32,
        ];

        Self {
            base: BaseSampler::new(samples_per_pixel),
            samples_per_pixel,
            randomize,
            digit_permutations,
            base_scales,
            base_exponents,
            mult_inverse,
            halton_index: 0,
            dimension: 2,
        }
    }

    pub fn start_pixel(&mut self, p: &Point2i) {
        self.base.start_pixel(p);
        self.fill_sample_arrays();
        self.start_pixel_sample(0, 0);
    }

    pub fn get_1d(&mut self) -> Float {
        let _p = ProfilePhase::new(Prof::GetSample);
        if self.dimension >= PRIMES.len() as u32 {
            self.dimension = 2;
        }
        let dim = self.dimension;
        self.dimension += 1;
        self.sample_dimension(dim)
    }

    pub fn get_2d(&mut self) -> Point2f {
        let _p = ProfilePhase::new(Prof::GetSample);
        if self.dimension + 1 >= PRIMES.len() as u32 {
            self.dimension = 2;
        }
        let dim = self.dimension;
        self.dimension += 2;
        Point2f::new(self.sample_dimension(dim), self.sample_dimension(dim + 1))
    }

    pub fn get_pixel_2d(&mut self) -> Point2f {
        Point2f::new(
            radical_inverse(0, (self.halton_index >> self.base_exponents[0]) as u64),
            radical_inverse(1, (self.halton_index / self.base_scales[1] as i64) as u64),
        )
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
    ) -> Result<HaltonSampler, PbrtError> {
        let mut nsamp = params.get_one_int("pixelsamples", 16) as u32;
        let randomize = parse_halton_randomize_strategy(
            params.get_one_string("randomization", "permutedigits"),
        )?;
        let seed = params.get_one_int("seed", PbrtOptions::get().seed as i32) as u32;
        {
            let options = PbrtOptions::get();
            if options.quick_render {
                nsamp = 1;
            }
        }
        Ok(HaltonSampler::new(nsamp, full_resolution, randomize, seed))
    }

    pub fn start_pixel_sample(&mut self, sample_index: u32, dim: u32) {
        self.halton_index = 0;
        let sample_stride = (self.base_scales[0] * self.base_scales[1]) as i64;
        if sample_stride > 1 {
            let pm = Point2i::new(
                math_mod(self.base.current_pixel[0] as i64, K_MAX_RESOLUTION as i64) as i32,
                math_mod(self.base.current_pixel[1] as i64, K_MAX_RESOLUTION as i64) as i32,
            );
            for i in 0..2 {
                let dim_offset = if i == 0 {
                    inverse_radical_inverse(2, pm[i] as u64, self.base_exponents[i] as usize)
                } else {
                    inverse_radical_inverse(3, pm[i] as u64, self.base_exponents[i] as usize)
                };
                self.halton_index += (dim_offset
                    * ((sample_stride as i32 / self.base_scales[i]) * self.mult_inverse[i]) as u64)
                    as i64;
            }
            self.halton_index %= sample_stride;
        }

        self.halton_index += sample_index as i64 * sample_stride;
        self.dimension = u32::max(2, dim);
    }

    fn sample_dimension(&self, dimension: u32) -> Float {
        match self.randomize {
            RandomizeStrategy::None => radical_inverse(dimension, self.halton_index as u64),
            RandomizeStrategy::PermuteDigits => {
                if dimension == 0 {
                    radical_inverse(
                        dimension,
                        (self.halton_index >> self.base_exponents[0]) as u64,
                    )
                } else if dimension == 1 {
                    radical_inverse(
                        dimension,
                        (self.halton_index / self.base_scales[1] as i64) as u64,
                    )
                } else {
                    let wrapped_dim = wrap_dimension(dimension);
                    scrambled_radical_inverse(
                        wrapped_dim,
                        self.halton_index as u64,
                        permutation_for_dimension(wrapped_dim, &self.digit_permutations),
                    )
                }
            }
            RandomizeStrategy::Owen => owen_scrambled_radical_inverse(
                dimension,
                self.halton_index as u64,
                mix_bits(1 + ((dimension as u64) << 4)) as u32,
            ),
            RandomizeStrategy::FastOwen => owen_scrambled_radical_inverse(
                dimension,
                self.halton_index as u64,
                mix_bits(1 + ((dimension as u64) << 4)) as u32,
            ),
        }
    }

    fn fill_sample_arrays(&mut self) {
        let mut dim = 5u32;
        for i in 0..self.base.samples1d_array_sizes.len() {
            let samples = self.samples_per_pixel * self.base.samples1d_array_sizes[i];
            for j in 0..samples {
                self.start_pixel_sample(j, dim);
                self.base.sample_array1d[i][j as usize] = self.sample_dimension(dim);
            }
            dim += 1;
        }
        for i in 0..self.base.samples2d_array_sizes.len() {
            let samples = self.samples_per_pixel * self.base.samples2d_array_sizes[i];
            for j in 0..samples {
                self.start_pixel_sample(j, dim);
                self.base.sample_array2d[i][j as usize] =
                    Point2f::new(self.sample_dimension(dim), self.sample_dimension(dim + 1));
            }
            dim += 2;
        }
    }
}

fn parse_halton_randomize_strategy(value: String) -> Result<RandomizeStrategy, PbrtError> {
    match value.as_str() {
        "none" => Ok(RandomizeStrategy::None),
        "permutedigits" => Ok(RandomizeStrategy::PermuteDigits),
        "owen" => Ok(RandomizeStrategy::Owen),
        "fastowen" => Err(PbrtError::error(
            "\"fastowen\" randomization not supported by Halton sampler.",
        )),
        _ => Err(PbrtError::error(&format!(
            "{}: unknown randomization strategy given to HaltonSampler",
            value
        ))),
    }
}

fn compute_radical_inverse_permutations_with_seed(seed: u32) -> Vec<DigitPermutation> {
    // pbrt-v4 `ComputeRadicalInversePermutations(seed)` (lowdiscrepancy.cpp:47).
    compute_radical_inverse_permutations(seed)
}

fn wrap_dimension(dim: u32) -> u32 {
    (dim as usize % PRIMES.len()) as u32
}

pub fn permutation_for_dimension(dim: u32, perms: &[DigitPermutation]) -> &DigitPermutation {
    &perms[wrap_dimension(dim) as usize]
}

fn math_mod(a: i64, b: i64) -> u64 {
    let result = a - (a / b) * b;
    if result < 0 {
        (result + b) as u64
    } else {
        result as u64
    }
}

fn extended_gcd(a: u64, b: u64) -> (i64, i64) {
    if b == 0 {
        return (1, 0);
    }
    let d = (a / b) as i64;
    let (xp, yp) = extended_gcd(b, a % b);
    (yp, xp - (d * yp))
}

fn multiplicative_inverse(a: i64, n: i64) -> u64 {
    let (x, _) = extended_gcd(a as u64, n as u64);
    math_mod(x, n)
}

fn owen_scrambled_radical_inverse(base_index: u32, a: u64, hash: u32) -> Float {
    let base = PRIMES[base_index as usize] as u64;
    let mut a = a;
    let inv_base = 1.0 / base as Float;
    let mut inv_base_n = 1.0;
    let mut reversed_digits = 0u64;
    while 1.0 - inv_base_n < 1.0 {
        let next = a / base;
        let mut digit_value = (a - next * base) as u32;
        let digit_hash = mix_bits((hash ^ reversed_digits as u32) as u64) as u32;
        digit_value = permutation_element(digit_value, base as u32, digit_hash);
        reversed_digits = reversed_digits * base + digit_value as u64;
        inv_base_n *= inv_base;
        a = next;
    }
    Float::min(inv_base_n * reversed_digits as Float, ONE_MINUS_EPSILON)
}

fn permutation_element(mut i: u32, l: u32, p: u32) -> u32 {
    let mut w = l - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;
    loop {
        i ^= p;
        i = i.wrapping_mul(0xe170893d);
        i ^= p >> 16;
        i ^= (i & w) >> 4;
        i ^= p >> 8;
        i = i.wrapping_mul(0x0929eb3f);
        i ^= p >> 23;
        i ^= (i & w) >> 1;
        i = i.wrapping_mul(1 | (p >> 27));
        i = i.wrapping_mul(0x6935fa69);
        i ^= (i & w) >> 11;
        i = i.wrapping_mul(0x74dcb303);
        i ^= (i & w) >> 2;
        i = i.wrapping_mul(0x9e501cc3);
        i ^= (i & w) >> 2;
        i = i.wrapping_mul(0xc860a3df);
        i &= w;
        i ^= i >> 5;
        if i < l {
            break;
        }
    }
    (i + p) % l
}

fn mix_bits(mut v: u64) -> u64 {
    v ^= v >> 31;
    v = v.wrapping_mul(0x7fb5d329728ea185);
    v ^= v >> 27;
    v = v.wrapping_mul(0x81dadef4bc2dd44d);
    v ^= v >> 33;
    v
}
