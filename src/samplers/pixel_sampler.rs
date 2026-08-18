use super::base_sampler::BaseSampler;
use crate::util::base::*;
use crate::util::rng::*;

#[derive(Debug, PartialEq, Default, Clone)]
pub struct BasePixelSampler {
    pub base: BaseSampler,
    pub samples1d: Vec<Vec<Float>>,
    pub samples2d: Vec<Vec<Point2f>>,
    pub current1d_dimension: u32,
    pub current2d_dimension: u32,
    pub rng: RNG,
}

impl BasePixelSampler {
    pub fn new(samples_per_pixel: u32, n_sampled_dimensions: u32) -> Self {
        let dim = n_sampled_dimensions as usize;
        let mut samples1d = Vec::with_capacity(dim);
        let mut samples2d = Vec::with_capacity(dim);
        for _ in 0..dim {
            samples1d.push(vec![0.0; samples_per_pixel as usize]);
            samples2d.push(vec![Vector2f::zero(); samples_per_pixel as usize]);
        }
        BasePixelSampler {
            base: BaseSampler::new(samples_per_pixel),
            samples1d,
            samples2d,
            current1d_dimension: 0,
            current2d_dimension: 0,
            rng: RNG::new(),
        }
    }

    pub fn start_pixel(&mut self, p: &Point2i) {
        self.base.start_pixel(p);
    }

    pub fn start_next_sample(&mut self) -> bool {
        self.current1d_dimension = 0;
        self.current2d_dimension = 0;
        return self.base.start_next_sample();
    }

    pub fn set_sample_number(&mut self, sample_num: u32) -> bool {
        self.current1d_dimension = 0;
        self.current2d_dimension = 0;
        return self.base.set_sample_number(sample_num);
    }

    pub fn get_1d(&mut self) -> Float {
        let dim = self.current1d_dimension as usize;
        if dim < self.samples1d.len() {
            self.current1d_dimension += 1;
            let index = self.base.current_pixel_sample_index as usize;
            return self.samples1d[dim][index];
        } else {
            return self.rng.uniform_float();
        }
    }

    pub fn get_2d(&mut self) -> Point2f {
        let dim = self.current2d_dimension as usize;
        if dim < self.samples2d.len() {
            self.current2d_dimension += 1;
            let index = self.base.current_pixel_sample_index as usize;
            return self.samples2d[dim][index];
        } else {
            return Point2f::new(self.rng.uniform_float(), self.rng.uniform_float());
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
}
