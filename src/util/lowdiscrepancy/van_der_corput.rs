use crate::util::base::*;
use crate::util::rng::*;
use crate::util::sampling::*;

// Define _CVanDerCorput_ Generator Matrix
#[rustfmt::skip]
const CVAN_DER_CORPUT:[u32;32] = [
    0b10000000000000000000000000000000,
    0b1000000000000000000000000000000,
    0b100000000000000000000000000000,
    0b10000000000000000000000000000,
    // Remainder of Van Der Corput generator matrix entries
    0b1000000000000000000000000000,
    0b100000000000000000000000000,
    0b10000000000000000000000000,
    0b1000000000000000000000000,
    0b100000000000000000000000,
    0b10000000000000000000000,
    0b1000000000000000000000,
    0b100000000000000000000,
    0b10000000000000000000,
    0b1000000000000000000,
    0b100000000000000000,
    0b10000000000000000,
    0b1000000000000000,
    0b100000000000000,
    0b10000000000000,
    0b1000000000000,
    0b100000000000,
    0b10000000000,
    0b1000000000,
    0b100000000,
    0b10000000,
    0b1000000,
    0b100000,
    0b10000,
    0b1000,
    0b100,
    0b10,
    0b1,
];

const CSOBOL: [[u32; 32]; 2] = [
    [
        0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x8000000, 0x4000000, 0x2000000, 0x1000000,
        0x800000, 0x400000, 0x200000, 0x100000, 0x80000, 0x40000, 0x20000, 0x10000, 0x8000, 0x4000,
        0x2000, 0x1000, 0x800, 0x400, 0x200, 0x100, 0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1,
    ],
    [
        0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000, 0xaa000000,
        0xff000000, 0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000, 0x88880000, 0xcccc0000,
        0xaaaa0000, 0xffff0000, 0x80008000, 0xc000c000, 0xa000a000, 0xf000f000, 0x88008800,
        0xcc00cc00, 0xaa00aa00, 0xff00ff00, 0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0,
        0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,
    ],
];

/// pbrt-v4 `MultiplyGenerator`: multiply a binary index by a generator
/// matrix using XOR over the selected columns.
#[inline]
pub fn multiply_generator(c: &[u32], mut a: u32) -> u32 {
    let mut v = 0;
    let mut i = 0;
    while a != 0 {
        if a & 1 != 0 {
            v ^= c[i];
        }
        i += 1;
        a >>= 1;
    }
    v
}

#[inline]
pub fn sample_generator_matrix(c: &[u32], a: u32, scramble: u32) -> Float {
    Float::min(
        (multiply_generator(c, a) ^ scramble) as Float * 2.3283064365386963e-10,
        ONE_MINUS_EPSILON,
    )
}

#[inline]
pub fn gray_code_sample_1d(c: &[u32], n: usize, scramble: u32, p: &mut [Float]) {
    let mut v = scramble;
    for i in 0..n {
        p[i] = Float::min(
            v as Float * 2.3283064365386963e-10, /* 1/2^32 */
            ONE_MINUS_EPSILON,
        );
        v ^= c[count_trailing_zeros((i + 1) as u32) as usize];
    }
}

#[inline]
pub fn gray_code_sample_2d(
    c0: &[u32],
    c1: &[u32],
    n: usize,
    scramble: &[u32; 2],
    p: &mut [Point2f],
) {
    let mut v = scramble.clone();
    for i in 0..n {
        p[i].x = Float::min(
            v[0] as Float * 2.3283064365386963e-10, /* 1/2^32 */
            ONE_MINUS_EPSILON,
        );
        p[i].y = Float::min(
            v[1] as Float * 2.3283064365386963e-10, /* 1/2^32 */
            ONE_MINUS_EPSILON,
        );
        v[0] ^= c0[count_trailing_zeros((i + 1) as u32) as usize];
        v[1] ^= c1[count_trailing_zeros((i + 1) as u32) as usize];
    }
}

#[inline]
pub fn van_der_corput(
    n_samples_per_pixel_sample: u32,
    n_pixel_samples: u32,
    samples: &mut [Float],
    rng: &mut RNG,
) {
    let n_samples_per_pixel_sample = n_samples_per_pixel_sample as usize;
    let n_pixel_samples = n_pixel_samples as usize;
    let scramble = rng.uniform_uint32();
    let total_samples = n_samples_per_pixel_sample * n_pixel_samples;
    gray_code_sample_1d(&CVAN_DER_CORPUT, total_samples as usize, scramble, samples);
    for i in 0..n_pixel_samples {
        let offset = i * n_samples_per_pixel_sample;
        shuffle_array(&mut samples[offset..], n_samples_per_pixel_sample, 1, rng);
    }
    shuffle_array(
        samples,
        n_pixel_samples,
        n_samples_per_pixel_sample as u32,
        rng,
    );
}

#[inline]
pub fn sobol_2d(
    n_samples_per_pixel_sample: u32,
    n_pixel_samples: u32,
    samples: &mut [Point2f],
    rng: &mut RNG,
) {
    let n_samples_per_pixel_sample = n_samples_per_pixel_sample as usize;
    let n_pixel_samples = n_pixel_samples as usize;

    let scramble = [rng.uniform_uint32(), rng.uniform_uint32()];
    gray_code_sample_2d(
        &CSOBOL[0],
        &CSOBOL[1],
        n_samples_per_pixel_sample * n_pixel_samples,
        &scramble,
        samples,
    );
    for i in 0..n_pixel_samples {
        let offset = i * n_samples_per_pixel_sample;
        shuffle_array(&mut samples[offset..], n_samples_per_pixel_sample, 1, rng);
    }
    shuffle_array(
        samples,
        n_pixel_samples,
        n_samples_per_pixel_sample as u32,
        rng,
    );
}
