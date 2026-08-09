use super::microfacet::*;

use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::scattering::*;

fn beckmann_sample_11(cos_theta_i: Float, u1: Float, u2: Float) -> (Float, Float) {
    /* Special case (normal incidence) */
    if cos_theta_i > 0.9999 {
        let r = Float::sqrt(-Float::ln(1.0 - u1));
        let sin_phi = Float::sin(2.0 * PI * u2);
        let cos_phi = Float::cos(2.0 * PI * u2);
        let slopex = r * cos_phi;
        let slopey = r * sin_phi;
        return (slopex, slopey);
    }

    /* The original inversion routine from the paper contained
    discontinuities, which causes issues for QMC integration
    and techniques like Kelemen-style MLT. The following code
    performs a numerical inversion with better behavior */
    let sin_theta_i = Float::sqrt(Float::max(0.0, 1.0 - cos_theta_i * cos_theta_i));
    let tan_theta_i = sin_theta_i / cos_theta_i;
    let cot_theta_i = 1.0 / tan_theta_i;

    /* Search interval -- everything is parameterized
    in the Erf() domain */
    let mut a = -1.0;
    let mut c = erf(cot_theta_i);
    let sample_x = Float::max(u1, 1e-6);

    /* Start with a good initial guess */

    /* We can do better (inverse of an approximation computed in
     * Mathematica) */
    let theta_i = Float::acos(cos_theta_i);
    let fit = 1.0 + theta_i * (-0.876 + theta_i * (0.4265 - 0.0594 * theta_i));
    let mut b = c - (1.0 + c) * Float::powf(1.0 - sample_x, fit);

    /* Normalization factor for the CDF */
    const SQRT_PI_INV: Float = INV_SQRT_PI;
    let normalization =
        1.0 / (1.0 + c + SQRT_PI_INV * tan_theta_i * Float::exp(-cot_theta_i * cot_theta_i));

    for _ in 0..10 {
        /* Bisection criterion -- the oddly-looking
        Boolean expression are intentional to check
        for NaNs at little additional cost */
        if !(b >= a && b <= c) {
            b = 0.5 * (a + c);
        }

        /* Evaluate the CDF and its derivative
        (i.e. the density function) */
        let inv_erf = erf_inv(b);
        let value = normalization
            * (1.0 + b + SQRT_PI_INV * tan_theta_i * Float::exp(-inv_erf * inv_erf))
            - sample_x;
        let derivative = normalization * (1.0 - inv_erf * tan_theta_i);

        if Float::abs(value) < 1e-5 {
            break;
        }

        /* Update bisection intervals */
        if value > 0.0 {
            c = b;
        } else {
            a = b;
        }
        b -= value / derivative;
    }

    /* Now convert back into a slope value */
    let slope_x = erf_inv(b);

    /* Simulate Y component */
    let slope_y = erf_inv(2.0 * Float::max(u2, 1e-6) - 1.0);

    assert!(!Float::is_infinite(slope_x));
    assert!(!Float::is_nan(slope_x));
    assert!(!Float::is_infinite(slope_y));
    assert!(!Float::is_nan(slope_y));

    return (slope_x, slope_y);
}

pub fn beckmann_sample(
    wi: &Vector3f,
    alpha_x: Float,
    alpha_y: Float,
    u1: Float,
    u2: Float,
) -> Vector3f {
    // 1. stretch wi
    let wi_stretched = Vector3f::new(alpha_x * wi.x, alpha_y * wi.y, wi.z).normalize();

    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    let (mut slope_x, mut slope_y) = beckmann_sample_11(cos_theta(&wi_stretched), u1, u2);

    // 3. rotate
    let tmp = cos_phi(&wi_stretched) * slope_x - sin_phi(&wi_stretched) * slope_y;
    slope_y = sin_phi(&wi_stretched) * slope_x + cos_phi(&wi_stretched) * slope_y;
    slope_x = tmp;

    // 4. unstretch
    slope_x *= alpha_x;
    slope_y *= alpha_y;

    // 5. compute normal
    return Vector3f::new(-slope_x, -slope_y, 1.0).normalize();
}

pub struct BeckmannDistribution {
    alphax: Float,
    alphay: Float,
    samplevis: bool,
}

impl BeckmannDistribution {
    pub fn new(alphax: Float, alphay: Float, samplevis: bool) -> Self {
        let alphax = Float::max(0.001, alphax);
        let alphay = Float::max(0.001, alphay);
        BeckmannDistribution {
            alphax,
            alphay,
            samplevis,
        }
    }
    pub fn roughness_to_alpha(roughness: Float) -> Float {
        let roughness = Float::max(roughness, 1e-3);
        let x = Float::ln(roughness);
        return 1.62142
            + 0.819955 * x
            + 0.1734 * x * x
            + 0.0171201 * x * x * x
            + 0.000640711 * x * x * x * x;
    }
}

fn sample_wh_helper(alphax: Float, alphay: Float, u0: Float, u1: Float) -> (Float, Float) {
    if alphax == alphay {
        let log_sample = Float::ln(1.0 - u0);
        assert!(!Float::is_infinite(log_sample));
        let tan2_theta = -alphax * alphax * log_sample;
        let phi = u1 * 2.0 * PI;
        return (tan2_theta, phi);
    } else {
        // Compute _tan2Theta_ and _phi_ for anisotropic Beckmann
        // distribution
        let log_sample = Float::ln(1.0 - u0);
        assert!(!Float::is_infinite(log_sample));
        let mut phi = Float::atan(alphay / alphax * Float::tan(2.0 * PI * u1 + 0.5 * PI));
        if u1 > 0.5 {
            phi += PI;
        }
        let sin_phi = Float::sin(phi);
        let cos_phi = Float::cos(phi);
        let alphax2 = alphax * alphax;
        let alphay2 = alphay * alphay;
        let tan2_theta = -log_sample / (cos_phi * cos_phi / alphax2 + sin_phi * sin_phi / alphay2);
        return (tan2_theta, phi);
    }
}

impl MicrofacetDistribution for BeckmannDistribution {
    fn d(&self, wh: &Vector3f) -> Float {
        let alphax = self.alphax;
        let alphay = self.alphay;
        let tan_2_theta = tan_2_theta(wh);
        if Float::is_infinite(tan_2_theta) {
            return 0.0;
        }
        let cos_2_theta = cos_2_theta(wh);
        let cos_4_theta = cos_2_theta * cos_2_theta;
        return Float::exp(
            -tan_2_theta * (cos_2_phi(wh) / (alphax * alphax) + sin_2_phi(wh) / (alphay * alphay)),
        ) / (PI * alphax * alphay * cos_4_theta);
    }

    fn lambda(&self, w: &Vector3f) -> Float {
        let alphax = self.alphax;
        let alphay = self.alphay;

        let abs_tan_theta = Float::abs(tan_theta(w));
        if Float::is_infinite(abs_tan_theta) {
            return 0.0;
        }
        // Compute _alpha_ for direction _w_
        let alpha = Float::sqrt(cos_2_phi(w) * alphax * alphax + sin_2_phi(w) * alphay * alphay);
        let a = 1.0 / (alpha * abs_tan_theta);
        if a >= 1.6 {
            return 0.0;
        }
        return (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a);
    }

    fn sample_wh(&self, wo: &Vector3f, u: &Vector2f) -> Vector3f {
        if !self.samplevis {
            // Sample full distribution of normals for Beckmann distribution
            let alphax = self.alphax;
            let alphay = self.alphay;

            let (tan2_theta, phi) = sample_wh_helper(alphax, alphay, u[0], u[1]);

            assert!(Float::is_finite(tan2_theta));
            assert!(Float::is_finite(phi));

            // Map sampled Beckmann angles to normal direction _wh_
            let cos_theta = 1.0 / Float::sqrt(1.0 + tan_2_theta(wo));
            let sin_theta = Float::sqrt(Float::max(0.0, 1.0 - cos_theta * cos_theta));
            let mut wh = spherical_direction(sin_theta, cos_theta, phi).normalize();
            if !same_hemisphere(wo, &wh) {
                wh = -wh;
            }
            return wh;
        } else {
            assert!(Float::is_finite(wo.x));
            assert!(Float::is_finite(wo.y));
            assert!(Float::is_finite(wo.z));

            // Sample visible area of normals for Beckmann distribution
            let flip = wo.z < 0.0;
            let wo = if flip { -*wo } else { *wo };
            let wi = if flip { -wo } else { wo };
            let mut wh = beckmann_sample(&wi, self.alphax, self.alphay, u[0], u[1]);
            if flip {
                wh = -wh;
            }
            return wh;
        }
    }

    fn pdf(&self, wo: &Vector3f, wh: &Vector3f) -> Float {
        if self.samplevis {
            return self.d(wh) * self.g1(wo) * Vector3f::abs_dot(wo, wh) / abs_cos_theta(wo);
        } else {
            return self.d(wh) * abs_cos_theta(wh);
        }
    }
}
