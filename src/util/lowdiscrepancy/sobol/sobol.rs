use super::sobolmatrices::*;
use crate::util::base::*;

#[inline]
pub fn sobol_interval_to_index(m: u32, frame: u64, p: &Point2i) -> u64 {
    let mut frame = frame;
    if m == 0 {
        return frame;
    }
    let m2 = m.wrapping_shl(1);
    let mut index = frame.wrapping_shl(m2);
    let mut delta: u64 = 0;
    let mut c = 0;
    while frame != 0 {
        if (frame & 1) != 0 {
            delta ^= VDC_SOBOL_MATRICES[(m - 1) as usize][c];
        }
        c += 1;
        frame = frame.wrapping_shr(1);
    }
    let mut b = ((((p.x as u32) as u64).wrapping_shl(m)) | (p.y as u64)) ^ delta;
    let mut c = 0;
    while b != 0 {
        if (b & 1) != 0 {
            index ^= VDC_SOBOL_MATRICES_INV[(m - 1) as usize][c];
        }
        c += 1;
        b = b.wrapping_shr(1);
    }
    return index;
}

#[inline]
pub fn sobol_sample(a: i64, dimension: u32, scramble: u64) -> Float {
    sobol_sample_float(a, dimension, scramble as u32) as Float
}

pub fn sobol_sample_float(a: i64, dimension: u32, scramble: u32) -> f32 {
    debug_assert!(a >= 0 && a < (1_i64 << SOBOL_MATRIX_SIZE));
    debug_assert!((dimension as usize) < NUM_SOBOL_DIMENSIONS);
    let mut a = a;
    let mut v = scramble;
    let mut i = dimension as usize * SOBOL_MATRIX_SIZE;
    while a != 0 {
        if (a & 1) != 0 {
            v ^= SOBOL_MATRICES_32[i] as u32;
        }
        a >>= 1;
        i += 1;
    }
    let fv = ((v as f64) * 2.3283064365386963e-10) as f32;
    return f32::min(fv, FLOAT_ONE_MINUS_EPSILON);
}

pub fn sobol_sample_double(a: i64, dimension: u32, scramble: u64) -> f64 {
    debug_assert!(a >= 0 && a < (1_i64 << SOBOL_MATRIX_SIZE));
    debug_assert!((dimension as usize) < NUM_SOBOL_DIMENSIONS);
    let mut a = a;
    let mut v = scramble;
    let mut i = dimension as usize * SOBOL_MATRIX_SIZE;
    while a != 0 {
        if (a & 1) != 0 {
            v ^= SOBOL_MATRICES_32[i] as u64;
        }
        a >>= 1;
        i += 1;
    }
    let fv = ((v as f64) * 2.3283064365386963e-10) as f64;
    return f64::min(fv, ONE_MINUS_EPSILON as f64);
}
