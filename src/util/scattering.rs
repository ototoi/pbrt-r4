use crate::util::base::*;

#[inline]
pub fn cos_theta(w: &Vector3f) -> Float {
    w.z
}

#[inline]
pub fn cos_2_theta(w: &Vector3f) -> Float {
    w.z * w.z
}

#[inline]
pub fn abs_cos_theta(w: &Vector3f) -> Float {
    Float::abs(w.z)
}

#[inline]
pub fn sin_2_theta(w: &Vector3f) -> Float {
    Float::max(0.0, 1.0 - cos_2_theta(w))
}

#[inline]
pub fn sin_theta(w: &Vector3f) -> Float {
    Float::sqrt(sin_2_theta(w))
}

#[inline]
pub fn tan_theta(w: &Vector3f) -> Float {
    sin_theta(w) / cos_theta(w)
}

#[inline]
pub fn tan_2_theta(w: &Vector3f) -> Float {
    sin_2_theta(w) / cos_2_theta(w)
}

#[inline]
pub fn cos_phi(w: &Vector3f) -> Float {
    let sin = sin_theta(w);
    if sin == 0.0 {
        1.0
    } else {
        Float::clamp(w.x / sin, -1.0, 1.0)
    }
}

#[inline]
pub fn sin_phi(w: &Vector3f) -> Float {
    let sin = sin_theta(w);
    if sin == 0.0 {
        0.0
    } else {
        Float::clamp(w.y / sin, -1.0, 1.0)
    }
}

#[inline]
pub fn cos_2_phi(w: &Vector3f) -> Float {
    cos_phi(w) * cos_phi(w)
}

#[inline]
pub fn sin_2_phi(w: &Vector3f) -> Float {
    sin_phi(w) * sin_phi(w)
}

#[inline]
pub fn cos_d_phi(wa: &Vector3f, wb: &Vector3f) -> Float {
    let waxy = wa.x * wa.x + wa.y * wa.y;
    let wbxy = wb.x * wb.x + wb.y * wb.y;
    if waxy <= 0.0 || wbxy <= 0.0 {
        return 1.0;
    }
    Float::clamp(
        (wa.x * wb.x + wa.y * wb.y) / Float::sqrt(waxy * wbxy),
        -1.0,
        1.0,
    )
}

#[inline]
pub fn reflect(wo: &Vector3f, n: &Vector3f) -> Vector3f {
    let a = 2.0 * Vector3f::dot(wo, n) * *n;
    let b = -*wo;
    a + b
}

#[inline]
pub fn refract(wi: &Vector3f, n: &Vector3f, eta: Float) -> Option<Vector3f> {
    // Compute $\cos \theta_\roman{t}$ using Snell's law
    let cos_theta_i = Vector3f::dot(n, wi);
    let sin2_theta_i = Float::max(0.0, 1.0 - cos_theta_i * cos_theta_i);
    let sin2_theta_t = eta * eta * sin2_theta_i;
    // Handle total internal reflection for transmission
    if sin2_theta_t >= 1.0 {
        None
    } else {
        let cos_theta_t = Float::sqrt(1.0 - sin2_theta_t);
        let a = eta * -*wi;
        let b = (eta * cos_theta_i - cos_theta_t) * *n;
        let wt = a + b;
        Some(wt)
    }
}

#[inline]
pub fn same_hemisphere(w: &Vector3f, wp: &Vector3f) -> bool {
    w.z * wp.z > 0.0
}
