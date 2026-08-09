use crate::base::camera::CameraSample;
use crate::options::*;
use crate::paramdict::*;
use crate::samplers::*;

use crate::util::base::*;
use crate::util::error::*;
use crate::util::sampling::*;

#[derive(Debug, PartialEq, Default, Clone)]
pub struct StratifiedSampler {
    base: BasePixelSampler,
    x_pixel_samples: u32,
    y_pixel_samples: u32,
    seed: u32,
    jitter_samples: bool,
}

impl StratifiedSampler {
    pub fn new(
        x_pixel_samples: u32,
        y_pixel_samples: u32,
        jitter_samples: bool,
        seed: u32,
        n_sampled_dimensions: u32,
    ) -> Self {
        let samples_per_pixel = x_pixel_samples * y_pixel_samples;
        StratifiedSampler {
            base: BasePixelSampler::new(samples_per_pixel, n_sampled_dimensions),
            x_pixel_samples,
            y_pixel_samples,
            seed,
            jitter_samples,
        }
    }

    pub fn start_pixel(&mut self, p: &Point2i) {
        self.base.start_pixel(p);
        self.base.rng.set_sequence(hash_pixel_seed(p, self.seed));
        for i in 0..self.base.samples1d.len() {
            let nsamples = self.base.samples1d[i].len();
            stratified_sample_1d(
                &mut self.base.samples1d[i],
                nsamples,
                &mut self.base.rng,
                self.jitter_samples,
            );
            shuffle_array(&mut self.base.samples1d[i], nsamples, 1, &mut self.base.rng);
        }
        for i in 0..self.base.samples2d.len() {
            stratified_sample_2d(
                &mut self.base.samples2d[i],
                self.x_pixel_samples as usize,
                self.y_pixel_samples as usize,
                &mut self.base.rng,
                self.jitter_samples,
            );
            let nsamples = self.base.samples2d[i].len();
            shuffle_array(&mut self.base.samples2d[i], nsamples, 1, &mut self.base.rng);
        }

        for i in 0..self.base.base.samples1d_array_sizes.len() {
            for j in 0..self.base.base.samples_per_pixel {
                let count = self.base.base.samples1d_array_sizes[i] as usize;
                let start = (j * count as u32) as usize;
                let end = ((j + 1) * count as u32) as usize;
                stratified_sample_1d(
                    &mut self.base.base.sample_array1d[i][start..end],
                    count,
                    &mut self.base.rng,
                    self.jitter_samples,
                );
                shuffle_array(
                    &mut self.base.base.sample_array1d[i][start..end],
                    count,
                    1,
                    &mut self.base.rng,
                );
            }
        }
        for i in 0..self.base.base.samples2d_array_sizes.len() {
            for j in 0..self.base.base.samples_per_pixel {
                let count = self.base.base.samples2d_array_sizes[i] as usize;
                let start = (j * count as u32) as usize;
                let end = ((j + 1) * count as u32) as usize;
                latin_hypercube_2d(
                    &mut self.base.base.sample_array2d[i][start..end],
                    count,
                    &mut self.base.rng,
                );
            }
        }
    }

    pub fn get_1d(&mut self) -> Float {
        self.base.get_1d()
    }

    pub fn get_2d(&mut self) -> Point2f {
        self.base.get_2d()
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
        self.base.start_next_sample()
    }

    pub fn set_sample_number(&mut self, sample_num: u32) -> bool {
        self.base.set_sample_number(sample_num)
    }

    pub fn get_samples_per_pixel(&self) -> u32 {
        self.base.base.samples_per_pixel
    }
}

impl StratifiedSampler {
    pub fn create(params: &ParameterDictionary) -> Result<StratifiedSampler, PbrtError> {
        let jitter = params.get_one_bool("jitter", true);
        let mut xsamp = params.get_one_int("xsamples", 4) as u32;
        let mut ysamp = params.get_one_int("ysamples", 4) as u32;
        let sd = params.get_one_int("dimensions", 4) as u32;
        let seed = params.get_one_int("seed", PbrtOptions::get().seed as i32) as u32;
        {
            let options = PbrtOptions::get();
            if options.quick_render {
                xsamp = 1;
                ysamp = 1;
            }
        }
        Ok(StratifiedSampler::new(xsamp, ysamp, jitter, seed, sd))
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
