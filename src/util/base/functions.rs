use crate::util::base::Float;
use crate::util::sampling::difference_of_products;

pub const MACHINE_EPSILON: Float = Float::EPSILON * 0.5;

#[inline]
pub fn gamma(n: Float) -> Float {
    let e = n * MACHINE_EPSILON;
    return e / (1.0 - e);
}

#[inline]
pub fn gamma_correct(value: Float) -> Float {
    if value <= 0.0031308 {
        return 12.92 * value;
    } else {
        return 1.055 * Float::powf(value, 1.0 / 2.4) - 0.055;
    }
}

#[inline]
pub fn inverse_gamma_correct(value: Float) -> Float {
    if value <= 0.04045 {
        return value * 1.0 / 12.92;
    } else {
        return Float::powf((value + 0.055) * 1.0 / 1.055, 2.4);
    }
}

#[inline]
pub fn lerp(t: Float, v1: Float, v2: Float) -> Float {
    return (1.0 - t) * v1 + t * v2;
}

#[inline]
pub fn quadratic(a: Float, b: Float, c: Float) -> Option<(Float, Float)> {
    if a == 0.0 {
        if b == 0.0 {
            return None;
        }
        let root = -c / b;
        return Some((root, root));
    }

    let discrim = difference_of_products(b, b, 4.0 * a, c);
    if discrim < 0.0 {
        return None;
    }
    let root_discrim = discrim.sqrt();
    let q = -0.5
        * (b + if b >= 0.0 {
            root_discrim
        } else {
            -root_discrim
        });
    let mut t0 = q / a;
    let mut t1 = if q == 0.0 { -b / (2.0 * a) } else { c / q };
    if t0 > t1 {
        std::mem::swap(&mut t0, &mut t1);
    }
    Some((t0, t1))
}

#[inline]
pub fn is_power_of_2(v: u32) -> bool {
    return (v != 0) && ((v & (v - 1)) == 0);
}

#[inline]
pub fn round_up_pow2(v: u32) -> u32 {
    let mut v = v;
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

#[inline]
pub fn is_power_of_2_64(v: u64) -> bool {
    return (v != 0) && ((v & (v - 1)) == 0);
}

#[inline]
pub fn round_up_pow2_64(v: u64) -> u64 {
    let mut v = v;
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

#[inline]
pub fn count_trailing_zeros(v: u32) -> u32 {
    return v.trailing_zeros();
}

//, const Predicate &pred
#[inline]
pub fn find_interval(v: &[Float], pred: &dyn Fn(&[Float], usize) -> bool) -> usize {
    let mut first = 0;
    let mut len = v.len() as i64;
    while len > 0 {
        let half = len.wrapping_shr(1);
        let middle = first + half;
        // Bisect range based on value of _pred_ at _middle_
        if pred(v, middle as usize) {
            first = middle + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    return i64::clamp(first - 1, 0, v.len() as i64 - 2) as usize;
}

#[inline]
pub fn find_interval_range(
    range: (usize, usize),
    v: &[Float],
    pred: &dyn Fn(&[Float], usize) -> bool,
) -> usize {
    let mut first = range.0;
    let mut len = range.1;
    while len > 0 {
        let half = len.wrapping_shr(1);
        let middle = first + half;
        // Bisect range based on value of _pred_ at _middle_
        if pred(v, middle) {
            first = middle + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    return usize::clamp(first - 1, 0, range.1 - 2);
}

#[inline]
pub fn log2int(v: u32) -> u32 {
    debug_assert!(v > 0);
    return 31 - v.leading_zeros();
}

#[inline]
pub fn log2_int_64(v: u64) -> u32 {
    debug_assert!(v > 0);
    return 63 - v.leading_zeros();
}

#[inline]
pub fn log4int(v: u32) -> u32 {
    log2int(v) / 2
}

#[inline]
pub fn log4_int_64(v: u64) -> u32 {
    log2_int_64(v) / 2
}

#[inline]
pub fn erf_inv(x: Float) -> Float {
    let x = Float::clamp(x, -0.99999, 0.99999);
    let mut w = -Float::ln((1.0 - x) * (1.0 + x));
    let mut p;
    if w < 5.0 {
        w -= 2.5;
        p = 2.81022636e-08;
        p = 3.43273939e-07 + p * w;
        p = -3.5233877e-06 + p * w;
        p = -4.39150654e-06 + p * w;
        p = 0.00021858087 + p * w;
        p = -0.00125372503 + p * w;
        p = -0.00417768164 + p * w;
        p = 0.246640727 + p * w;
        p = 1.50140941 + p * w;
    } else {
        w = Float::sqrt(w) - 3.0;
        p = -0.000200214257;
        p = 0.000100950558 + p * w;
        p = 0.00134934322 + p * w;
        p = -0.00367342844 + p * w;
        p = 0.00573950773 + p * w;
        p = -0.0076224613 + p * w;
        p = 0.00943887047 + p * w;
        p = 1.00167406 + p * w;
        p = 2.83297682 + p * w;
    }
    return p * x;
}

#[inline]
pub fn erf(x: Float) -> Float {
    const A1: Float = 0.254829592;
    const A2: Float = -0.284496736;
    const A3: Float = 1.421413741;
    const A4: Float = -1.453152027;
    const A5: Float = 1.061405429;
    const P: Float = 0.3275911;

    // Save the sign of x
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = Float::abs(x);

    // A&S formula 7.1.26
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * Float::exp(-x * x);

    return sign * y;
}
