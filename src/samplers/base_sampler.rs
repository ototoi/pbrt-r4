use crate::util::base::*;
use crate::util::profile::*;

#[derive(Debug, PartialEq, Default, Clone)]
pub struct BaseSampler {
    pub samples_per_pixel: u32,
    pub current_pixel: Point2i,
    pub current_pixel_sample_index: u32,
    pub samples1d_array_sizes: Vec<u32>,
    pub samples2d_array_sizes: Vec<u32>,
    pub sample_array1d: Vec<Vec<Float>>,
    pub sample_array2d: Vec<Vec<Point2f>>,
    pub array1d_offset: u64,
    pub array2d_offset: u64,
}

impl BaseSampler {
    pub fn new(samples_per_pixel: u32) -> Self {
        BaseSampler {
            samples_per_pixel,
            current_pixel: Point2i::zero(),
            current_pixel_sample_index: 0,
            samples1d_array_sizes: Vec::new(),
            samples2d_array_sizes: Vec::new(),
            sample_array1d: Vec::new(),
            sample_array2d: Vec::new(),
            array1d_offset: 0,
            array2d_offset: 0,
        }
    }

    pub fn request_1d_array(&mut self, n: u32) {
        self.samples1d_array_sizes.push(n);
        self.sample_array1d
            .push(vec![0.0; (n * self.samples_per_pixel) as usize]);
    }

    pub fn request_2d_array(&mut self, n: u32) {
        self.samples2d_array_sizes.push(n);
        self.sample_array2d.push(vec![
            Vector2f::zero();
            (n * self.samples_per_pixel) as usize
        ]);
    }

    pub fn get_1d_array(&mut self, n: u32) -> Option<Vec<Float>> {
        let _p = ProfilePhase::new(Prof::GetSample);

        let n = n as usize;
        if self.array1d_offset == self.sample_array1d.len() as u64 {
            return None;
        }
        let o1 = self.array1d_offset as usize;
        let o2 = n * self.current_pixel_sample_index as usize;
        self.array1d_offset += 1;
        return Some(self.sample_array1d[o1][o2..(o2 + n)].to_vec());
    }

    pub fn get_2d_array(&mut self, n: u32) -> Option<Vec<Vector2f>> {
        let _p = ProfilePhase::new(Prof::GetSample);

        let n = n as usize;
        if self.array2d_offset == self.sample_array2d.len() as u64 {
            return None;
        }
        let o1 = self.array2d_offset as usize;
        let o2 = n * self.current_pixel_sample_index as usize;
        self.array2d_offset += 1;
        return Some(self.sample_array2d[o1][o2..(o2 + n)].to_vec());
    }

    pub fn start_pixel(&mut self, p: &Point2i) {
        let _p = ProfilePhase::new(Prof::StartPixel);

        self.current_pixel = *p;
        self.current_pixel_sample_index = 0;
        self.array1d_offset = 0;
        self.array2d_offset = 0;
    }

    pub fn start_next_sample(&mut self) -> bool {
        self.array1d_offset = 0;
        self.array2d_offset = 0;
        self.current_pixel_sample_index += 1;
        return self.current_pixel_sample_index < self.samples_per_pixel;
    }

    pub fn set_sample_number(&mut self, sample_num: u32) -> bool {
        self.array1d_offset = 0;
        self.array2d_offset = 0;
        self.current_pixel_sample_index = sample_num;
        return self.current_pixel_sample_index < self.samples_per_pixel;
    }

    pub fn get_samples_per_pixel(&self) -> u32 {
        return self.samples_per_pixel;
    }
}
