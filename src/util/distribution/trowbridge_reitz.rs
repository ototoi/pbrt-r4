use super::microfacet::*;

use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::scattering::*; // Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
                                // For scattering functions

fn uniform_sample_disk_polar(u: &Point2f) -> Point2f {
    let r = Float::sqrt(u[0]);
    let phi = 2.0 * PI * u[1];
    Point2f::new(r * Float::cos(phi), r * Float::sin(phi))
}

#[derive(Debug, Clone)]
pub struct TrowbridgeReitzDistribution {
    alphax: Float,
    alphay: Float,
    samplevis: bool,
}

impl TrowbridgeReitzDistribution {
    pub fn new(alphax: Float, alphay: Float, samplevis: bool) -> Self {
        let mut distrib = TrowbridgeReitzDistribution {
            alphax,
            alphay,
            samplevis,
        };
        if !distrib.effectively_smooth() {
            distrib.alphax = Float::max(1e-4, distrib.alphax);
            distrib.alphay = Float::max(1e-4, distrib.alphay);
        }
        distrib
    }

    pub fn roughness_to_alpha(roughness: Float) -> Float {
        Float::sqrt(Float::max(roughness, 0.0))
    }

    pub fn effectively_smooth(&self) -> bool {
        Float::max(self.alphax, self.alphay) < 1e-3
    }

    pub fn regularize(&mut self) {
        if self.alphax < 0.3 {
            self.alphax = Float::clamp(2.0 * self.alphax, 0.1, 0.3);
        }
        if self.alphay < 0.3 {
            self.alphay = Float::clamp(2.0 * self.alphay, 0.1, 0.3);
        }
    }
}

fn sample_wh_helper(alphax: Float, alphay: Float, u1: Float, u2: Float) -> (Float, Float, Float) {
    let u1 = u1.clamp(1e-6, 1.0 - 1e-6);
    let u2 = u2.clamp(1e-6, 1.0 - 1e-6);
    //
    if alphax == alphay {
        let tan_theta_2 = alphax * alphax * u1 / (1.0 - u1);
        let phi = 2.0 * PI * u2;
        let cos_theta = 1.0 / Float::sqrt(1.0 + tan_theta_2);
        let sin_theta = Float::sqrt(Float::max(0.0, 1.0 - cos_theta * cos_theta));
        return (sin_theta, cos_theta, phi);
    } else {
        let mut phi = Float::atan(alphay / alphax * Float::tan(2.0 * PI * u2 + 0.5 * PI));
        if u2 > 0.5 {
            phi += PI;
        }
        let sin_phi = Float::sin(phi);
        let cos_phi = Float::cos(phi);
        let alphax2 = alphax * alphax;
        let alphay2 = alphay * alphay;
        let alpha2 = 1.0 / (cos_phi * cos_phi / alphax2 + sin_phi * sin_phi / alphay2);
        let tan_theta_2 = alpha2 * u1 / (1.0 - u1);
        let cos_theta = 1.0 / Float::sqrt(1.0 + tan_theta_2);
        let sin_theta = Float::sqrt(Float::max(0.0, 1.0 - cos_theta * cos_theta));
        return (sin_theta, cos_theta, phi);
    }
}

impl MicrofacetDistribution for TrowbridgeReitzDistribution {
    fn d(&self, wh: &Vector3f) -> Float {
        let alphax = self.alphax;
        let alphay = self.alphay;
        let tan_2_theta = tan_2_theta(wh);
        if Float::is_infinite(tan_2_theta) {
            return 0.0;
        }
        let cos_4_theta = cos_2_theta(wh) * cos_2_theta(wh);
        if cos_4_theta < 1e-16 {
            return 0.0;
        }
        let e =
            (cos_2_phi(wh) / (alphax * alphax) + sin_2_phi(wh) / (alphay * alphay)) * tan_2_theta;
        1.0 / (PI * alphax * alphay * cos_4_theta * (1.0 + e) * (1.0 + e))
    }

    fn lambda(&self, w: &Vector3f) -> Float {
        let tan_2_theta = tan_2_theta(w);
        if Float::is_infinite(tan_2_theta) {
            return 0.0;
        }
        let alphax = self.alphax;
        let alphay = self.alphay;
        let alpha_2 = (cos_phi(w) * alphax) * (cos_phi(w) * alphax)
            + (sin_phi(w) * alphay) * (sin_phi(w) * alphay);
        (Float::sqrt(1.0 + alpha_2 * tan_2_theta) - 1.0) / 2.0
    }

    fn sample_wh(&self, wo: &Vector3f, u: &Vector2f) -> Vector3f {
        if !self.samplevis {
            let (sin_theta, cos_theta, phi) =
                sample_wh_helper(self.alphax, self.alphay, u[0], u[1]);

            assert!(!Float::is_infinite(sin_theta));
            assert!(!Float::is_nan(sin_theta));
            assert!(!Float::is_infinite(cos_theta));
            assert!(!Float::is_nan(cos_theta));
            assert!(!Float::is_infinite(phi));
            assert!(!Float::is_nan(phi));

            let mut wh = spherical_direction(sin_theta, cos_theta, phi).normalize();
            if !same_hemisphere(wo, &wh) {
                wh = -wh;
            }
            wh
        } else {
            let mut wh = Vector3f::new(self.alphax * wo.x, self.alphay * wo.y, wo.z).normalize();
            if wh.z < 0.0 {
                wh = -wh;
            }

            let t1 = if wh.z < 0.99999 {
                Vector3f::cross(&Vector3f::new(0.0, 0.0, 1.0), &wh).normalize()
            } else {
                Vector3f::new(1.0, 0.0, 0.0)
            };
            let t2 = Vector3f::cross(&wh, &t1);

            let mut p = uniform_sample_disk_polar(u);
            let h = Float::sqrt(Float::max(0.0, 1.0 - p.x * p.x));
            p.y = lerp((1.0 + wh.z) / 2.0, h, p.y);

            let pz = Float::sqrt(Float::max(0.0, 1.0 - p.x * p.x - p.y * p.y));
            let nh = p.x * t1 + p.y * t2 + pz * wh;
            Vector3f::new(
                self.alphax * nh.x,
                self.alphay * nh.y,
                Float::max(1e-6, nh.z),
            )
            .normalize()
        }
    }

    fn pdf(&self, wo: &Vector3f, wh: &Vector3f) -> Float {
        if self.samplevis {
            if abs_cos_theta(wo) == 0.0 {
                return 0.0;
            }
            self.d(wh) * self.g1(wo) * Vector3f::abs_dot(wo, wh) / abs_cos_theta(wo)
        } else {
            self.d(wh) * abs_cos_theta(wh)
        }
    }
}
