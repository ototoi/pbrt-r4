use crate::base::camera::CameraSample;
use crate::options::PbrtOptions;
use crate::samplers::*;

use crate::util::base::*;
use crate::util::profile::*;
use crate::util::sampling::*;

#[derive(Debug, PartialEq, Default, Clone)]
pub struct PrimarySample {
    pub value: Float,
    pub last_modification_iteration: i64,
    pub value_backup: Float,
    pub modify_backup: i64,
}

impl PrimarySample {
    pub fn backup(&mut self) {
        self.value_backup = self.value;
        self.modify_backup = self.last_modification_iteration;
    }
    pub fn restore(&mut self) {
        self.value = self.value_backup;
        self.last_modification_iteration = self.modify_backup;
    }
}

#[derive(Debug, PartialEq, Default, Clone)]
pub struct MLTSampler {
    base: BaseSampler,
    rng: RNG,
    sigma: Float,
    large_step_probability: Float,
    stream_count: u64,

    x: Vec<PrimarySample>,
    current_iteration: i64,
    large_step: bool,
    last_large_step_iteration: i64,
    stream_index: u64,
    sample_index: u64,
}

impl MLTSampler {
    pub fn new(
        mutation_per_pixel: u32,
        rng_sequence_index: u64,
        sigma: Float,
        large_step_probability: Float,
        stream_count: u64,
    ) -> Self {
        let x = Vec::new();
        // pbrt-v4 (samplers.h:640) initializes the internal RNG with
        // `rng(MixBits(rngSequenceIndex) ^ MixBits(Options->seed))`, where
        // the single-arg ctor expands to `SetSequence(s, MixBits(s))`.
        // Use the scene-level seed in the same sequence hash as pbrt-v4.
        let seed = PbrtOptions::get().seed as u64;
        let mut rng = RNG::new();
        rng.set_sequence(mix_bits(rng_sequence_index) ^ mix_bits(seed));
        MLTSampler {
            base: BaseSampler::new(mutation_per_pixel),
            rng,
            sigma,
            large_step_probability,
            stream_count: stream_count,
            x,
            current_iteration: 0,
            large_step: true,
            last_large_step_iteration: 0,
            stream_index: 0,
            sample_index: 0,
        }
    }

    pub fn start_iteration(&mut self) {
        self.current_iteration += 1;
        self.large_step = self.rng.uniform_float() < self.large_step_probability;
    }

    pub fn accept(&mut self) {
        if self.large_step {
            self.last_large_step_iteration = self.current_iteration;
        }
    }

    pub fn reject(&mut self) {
        for xi in &mut self.x {
            if xi.last_modification_iteration == self.current_iteration {
                xi.restore();
            }
        }
        self.current_iteration -= 1;
    }

    pub fn start_stream(&mut self, index: u64) {
        assert!(index < self.stream_count);
        self.stream_index = index;
        self.sample_index = 0;
    }

    pub fn get_next_index(&mut self) -> u64 {
        let index = self.stream_index + self.stream_count * self.sample_index;
        self.sample_index += 1;
        return index;
    }

    fn ensure_ready(&mut self, index: usize) {
        // Enlarge _MLTSampler::X_ if necessary and get current $\VEC{X}_i$
        if index >= self.x.len() {
            self.x.resize(index + 1, PrimarySample::default());
        }
        let xi = &mut self.x[index];

        // Reset $\VEC{X}_i$ if a large step took place in the meantime
        if xi.last_modification_iteration < self.last_large_step_iteration {
            xi.value = self.rng.uniform_float();
            xi.last_modification_iteration = self.last_large_step_iteration;
        }

        // Apply remaining sequence of mutations to _xi_
        xi.backup();
        if self.large_step {
            xi.value = self.rng.uniform_float();
        } else {
            let n_small = self.current_iteration - xi.last_modification_iteration;
            // Apply _n_small_ small step mutations

            // Sample the standard normal distribution $N(0, 1)$
            let normal_sample = SQRT_2 * erf_inv(2.0 * self.rng.uniform_float() - 1.0);

            // Compute the effective standard deviation and apply perturbation to $\VEC{X}_i$
            // $\VEC{X}_i$
            let eff_sigma = self.sigma * Float::sqrt(n_small as Float);
            xi.value += normal_sample * eff_sigma;
            xi.value -= Float::floor(xi.value);
        }
        xi.last_modification_iteration = self.current_iteration;
    }

    pub fn start_pixel(&mut self, p: &Point2i) {
        self.base.start_pixel(p);
    }

    pub fn get_1d(&mut self) -> Float {
        let _p = ProfilePhase::new(Prof::GetSample);
        let index = self.get_next_index() as usize;
        self.ensure_ready(index);
        return self.x[index].value;
    }

    pub fn get_2d(&mut self) -> Point2f {
        let x = self.get_1d();
        let y = self.get_1d();
        return Point2f { x, y };
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

    pub fn request_1d_array(&mut self, n: u32) {
        self.base.request_1d_array(n);
    }

    pub fn request_2d_array(&mut self, n: u32) {
        self.base.request_2d_array(n);
    }

    pub fn get_1d_array(&mut self, n: u32) -> Option<Vec<Float>> {
        return self.base.get_1d_array(n);
    }

    pub fn get_2d_array(&mut self, n: u32) -> Option<Vec<Vector2f>> {
        return self.base.get_2d_array(n);
    }

    pub fn start_next_sample(&mut self) -> bool {
        return self.base.start_next_sample();
    }

    pub fn clone_with_seed(&self, seed: u32) -> MLTSampler {
        MLTSampler {
            base: self.base.clone(),
            rng: RNG::new_sequence(seed as u64),
            sigma: self.sigma,
            large_step_probability: self.large_step_probability,
            stream_count: self.stream_count,
            x: self.x.clone(),
            current_iteration: self.current_iteration,
            large_step: self.large_step,
            last_large_step_iteration: self.last_large_step_iteration,
            stream_index: self.stream_index,
            sample_index: self.sample_index,
        }
    }

    pub fn get_samples_per_pixel(&self) -> u32 {
        return self.base.get_samples_per_pixel();
    }
}
