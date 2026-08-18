use crate::base::camera::CameraSample;
use crate::options::*;
use crate::paramdict::*;
use crate::samplers::*;

use crate::util::base::*;
use crate::util::error::*;
use crate::util::sampling::*;

#[derive(Debug, Clone, PartialEq)]
pub struct IndependentSampler {
    base: BaseSampler,
    rng: RNG,
    samples_per_pixel: u32,
    seed: u32,
    pixel: Point2i,
}

impl IndependentSampler {
    pub fn new(samples_per_pixel: u32, seed: u32) -> Self {
        Self {
            base: BaseSampler::new(samples_per_pixel),
            rng: RNG::new(),
            samples_per_pixel,
            seed,
            pixel: Point2i::zero(),
        }
    }

    pub fn start_pixel(&mut self, p: &Point2i) {
        self.base.start_pixel(p);
        self.pixel = *p;
        self.fill_sample_arrays();
        self.start_pixel_sample(0, 0);
    }

    pub fn get_1d(&mut self) -> Float {
        self.rng.uniform_float()
    }

    pub fn get_2d(&mut self) -> Point2f {
        Point2f::new(self.rng.uniform_float(), self.rng.uniform_float())
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

    pub fn create(params: &ParameterDictionary) -> Result<IndependentSampler, PbrtError> {
        let ns = params.get_one_int("pixelsamples", 4) as u32;
        let seed = params.get_one_int("seed", PbrtOptions::get().seed as i32) as u32;
        Ok(Self::new(ns, seed))
    }

    pub fn start_pixel_sample(&mut self, sample_index: u32, dimension: u32) {
        self.pixel = self.base.current_pixel;
        self.rng
            .set_sequence(hash_pixel_seed(&self.pixel, self.seed));
        let offset = u64::from(sample_index) * 65536 + u64::from(dimension);
        self.rng.advance(offset as i64);
    }

    fn fill_sample_arrays(&mut self) {
        let mut dim = 5u32;
        for i in 0..self.base.samples1d_array_sizes.len() {
            let total = self.samples_per_pixel * self.base.samples1d_array_sizes[i];
            for j in 0..total {
                let mut rng = RNG::new();
                rng.set_sequence(hash_pixel_seed(&self.pixel, self.seed));
                let offset = u64::from(j) * 65536 + u64::from(dim);
                rng.advance(offset as i64);
                self.base.sample_array1d[i][j as usize] = rng.uniform_float();
            }
            dim += 1;
        }
        for i in 0..self.base.samples2d_array_sizes.len() {
            let total = self.samples_per_pixel * self.base.samples2d_array_sizes[i];
            for j in 0..total {
                let mut rng = RNG::new();
                rng.set_sequence(hash_pixel_seed(&self.pixel, self.seed));
                let offset = u64::from(j) * 65536 + u64::from(dim);
                rng.advance(offset as i64);
                self.base.sample_array2d[i][j as usize] =
                    Point2f::new(rng.uniform_float(), rng.uniform_float());
            }
            dim += 2;
        }
    }
}

fn hash_pixel_seed(pixel: &Point2i, seed: u32) -> u64 {
    let mut buf = [0u8; 12];
    buf[0..4].copy_from_slice(&pixel.x.to_ne_bytes());
    buf[4..8].copy_from_slice(&pixel.y.to_ne_bytes());
    buf[8..12].copy_from_slice(&seed.to_ne_bytes());
    murmur_hash_64a(&buf, 0)
}

fn murmur_hash_64a(key: &[u8], seed: u64) -> u64 {
    let m = 0xc6a4a7935bd1e995u64;
    let r = 47u32;

    let len = key.len() as u64;
    let mut h = seed ^ len.wrapping_mul(m);

    let nblocks = key.len() / 8;
    for i in 0..nblocks {
        let start = i * 8;
        let mut k = u64::from_ne_bytes(key[start..start + 8].try_into().unwrap());
        k = k.wrapping_mul(m);
        k ^= k >> r;
        k = k.wrapping_mul(m);

        h ^= k;
        h = h.wrapping_mul(m);
    }

    let tail = &key[nblocks * 8..];
    match tail.len() {
        7 => {
            h ^= (tail[6] as u64) << 48;
            h ^= (tail[5] as u64) << 40;
            h ^= (tail[4] as u64) << 32;
            h ^= (tail[3] as u64) << 24;
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        6 => {
            h ^= (tail[5] as u64) << 40;
            h ^= (tail[4] as u64) << 32;
            h ^= (tail[3] as u64) << 24;
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        5 => {
            h ^= (tail[4] as u64) << 32;
            h ^= (tail[3] as u64) << 24;
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        4 => {
            h ^= (tail[3] as u64) << 24;
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        3 => {
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        2 => {
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        1 => {
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        _ => {}
    }

    h ^= h >> r;
    h = h.wrapping_mul(m);
    h ^= h >> r;
    h
}
