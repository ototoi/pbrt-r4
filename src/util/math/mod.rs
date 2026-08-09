use crate::util::base::*;

/// Returns floor(log2(v)), matching pbrt-v4's `Log2Int` for positive values.
#[inline]
pub fn log2_int(v: u64) -> u32 {
    debug_assert!(v > 0);
    63 - v.leading_zeros()
}

/// Returns floor(log4(v)), matching pbrt-v4's `Log4Int` for positive values.
#[inline]
pub fn log4_int(v: u64) -> u32 {
    log2_int(v) / 2
}

#[inline]
pub fn is_power_of_four(v: u64) -> bool {
    v > 0 && v.is_power_of_two() && (log2_int(v) & 1) == 0
}

#[inline]
pub fn round_up_power_of_four(v: u64) -> u64 {
    debug_assert!(v > 0);
    if is_power_of_four(v) {
        v
    } else {
        1u64 << (2 * (1 + log4_int(v)))
    }
}

/// pbrt-v4's CPU `FastExp` approximation for the f32 build.
#[inline]
pub fn fast_exp(x: Float) -> Float {
    #[cfg(not(feature = "float-as-double"))]
    {
        let xp = x * 1.442695041;
        let fxp = xp.floor();
        let f = xp - fxp;
        let exponent = fxp as i32;
        let two_to_f = evaluate_polynomial(f, &[1.0, 0.695556856, 0.226173572, 0.0781455737]);
        let two_to_f_bits = two_to_f.to_bits();
        let current_exponent = ((two_to_f_bits >> 23) & 0xff) as i32 - 127;
        let final_exponent = current_exponent + exponent;
        if final_exponent < -126 {
            0.0
        } else if final_exponent > 127 {
            Float::INFINITY
        } else {
            let bits = (two_to_f_bits & 0x807f_ffff) | (((final_exponent + 127) as u32) << 23);
            Float::from_bits(bits)
        }
    }
    #[cfg(feature = "float-as-double")]
    {
        x.exp()
    }
}

#[inline]
pub fn sinc(x: Float) -> Float {
    let x = Float::abs(x);
    if x < 1e-5 {
        1.0
    } else {
        Float::sin(PI * x) / (PI * x)
    }
}

#[inline]
pub fn windowed_sinc(x: Float, radius: Float, tau: Float) -> Float {
    if Float::abs(x) > radius {
        0.0
    } else {
        sinc(x) * sinc(x / tau)
    }
}

#[inline]
pub fn safe_asin(x: Float) -> Float {
    debug_assert!((-1.0001..=1.0001).contains(&x));
    x.clamp(-1.0, 1.0).asin()
}

#[inline]
pub fn safe_acos(x: Float) -> Float {
    debug_assert!((-1.0001..=1.0001).contains(&x));
    x.clamp(-1.0, 1.0).acos()
}

#[inline]
pub fn log2(x: Float) -> Float {
    x.ln() * std::f64::consts::LOG2_E as Float
}

#[inline]
pub fn evaluate_polynomial(x: Float, coefficients: &[Float]) -> Float {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |value, coefficient| value * x + coefficient)
}

#[inline]
pub fn i0(x: Float) -> Float {
    let mut value = 0.0;
    let mut x2i = 1.0;
    let mut factorial = 1;
    let mut power = 1;
    for i in 0..10 {
        if i > 1 {
            factorial *= i;
        }
        value += x2i / (power as Float * (factorial as Float).powi(2));
        x2i *= x * x;
        power *= 4;
    }
    value
}

#[inline]
pub fn log_i0(x: Float) -> Float {
    if x > 12.0 {
        x + 0.5 * (-(2.0 * PI).ln() + (1.0 / x).ln() + 1.0 / (8.0 * x))
    } else {
        i0(x).ln()
    }
}

pub fn newton_bisection<F>(mut x0: Float, mut x1: Float, f: F) -> Float
where
    F: Fn(Float) -> (Float, Float),
{
    debug_assert!(x0 < x1);
    let (f0, f1) = (f(x0).0, f(x1).0);
    if f0.abs() < 1e-6 {
        return x0;
    }
    if f1.abs() < 1e-6 {
        return x1;
    }
    let start_negative = f0 < 0.0;
    let mut x = x0 + (x1 - x0) * -f0 / (f1 - f0);
    for _ in 0..128 {
        if !(x0 < x && x < x1) {
            x = (x0 + x1) / 2.0;
        }
        let (fx, derivative) = f(x);
        if start_negative == (fx < 0.0) {
            x0 = x;
        } else {
            x1 = x;
        }
        if x1 - x0 < 1e-6 || fx.abs() < 1e-6 {
            return x;
        }
        x -= fx / derivative;
    }
    (x0 + x1) / 2.0
}

#[inline]
pub fn gaussian(x: Float, mu: Float, sigma: Float) -> Float {
    if sigma <= 0.0 {
        return 0.0;
    }
    let sigma2 = sigma * sigma;
    (1.0 / Float::sqrt(2.0 * PI * sigma2)) * Float::exp(-((x - mu) * (x - mu)) / (2.0 * sigma2))
}

#[inline]
pub fn gaussian_integral(x0: Float, x1: Float, mu: Float, sigma: Float) -> Float {
    if sigma <= 0.0 {
        return 0.0;
    }
    let sigma_root2 = sigma * Float::sqrt(2.0);
    0.5 * (erf((mu - x0) / sigma_root2) - erf((mu - x1) / sigma_root2))
}
