use super::base_sampler::BaseSampler;
use crate::util::base::*;

#[derive(Debug, PartialEq, Default, Clone)]
pub struct BaseGlobalSampler {
    pub base: BaseSampler,
    pub dimension: u32,
    pub interval_sample_index: i64,
    pub array_end_dim: u32,
}

pub trait GlobalSampler {
    fn get_index_for_sample(&self, sample_num: i64) -> i64;
    fn sample_dimension(&self, index: i64, dimension: u32) -> Float;
    fn get_base(&mut self) -> &mut BaseGlobalSampler;
}

impl BaseGlobalSampler {
    #[allow(non_upper_case_globals)]
    pub const array_start_dim: u32 = 5;

    pub fn new(samples_per_pixel: u32) -> Self {
        BaseGlobalSampler {
            base: BaseSampler::new(samples_per_pixel),
            dimension: 0,
            interval_sample_index: 0,
            array_end_dim: Self::array_start_dim + 1,
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
