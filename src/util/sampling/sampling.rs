pub use crate::util::base::*;
use crate::util::geometry::spherical_direction_axes;
pub use crate::util::math::gaussian;
use crate::util::misc::next_float_down;
pub use crate::util::rng::*; // Import RNG type
use crate::util::spectrum::cie::lerp as cie_lerp;
use crate::util::vecmath::coordinate_system;
use crate::util::vecmath::frame::Frame;

/// pbrt-v4 thresholds (`Triangle::MinSphericalSampleArea` /
/// `MaxSphericalSampleArea`): triangles whose subtended solid angle
/// falls in this range use spherical-triangle sampling instead of the
/// uniform-area sampling.
pub const MIN_SPHERICAL_SAMPLE_AREA: Float = 3e-4;
pub const MAX_SPHERICAL_SAMPLE_AREA: Float = 6.22;

/// pbrt-v4 `SphericalTriangleArea` — solid angle of the spherical
/// triangle formed by three unit-length direction vectors.
#[inline]
pub fn spherical_triangle_area(a: &Vector3f, b: &Vector3f, c: &Vector3f) -> Float {
    Float::abs(
        2.0 * Float::atan2(
            a.dot(&Vector3f::cross(b, c)),
            1.0 + a.dot(b) + a.dot(c) + b.dot(c),
        ),
    )
}

/// pbrt-v4 `SphericalQuadArea` — solid angle of the spherical quadrilateral
/// formed by four unit-length direction vectors.
#[inline]
pub fn spherical_quad_area(a: &Vector3f, b: &Vector3f, c: &Vector3f, d: &Vector3f) -> Float {
    let mut axb = Vector3f::cross(a, b);
    let mut bxc = Vector3f::cross(b, c);
    let mut cxd = Vector3f::cross(c, d);
    let mut dxa = Vector3f::cross(d, a);
    if axb.length_squared() == 0.0
        || bxc.length_squared() == 0.0
        || cxd.length_squared() == 0.0
        || dxa.length_squared() == 0.0
    {
        return 0.0;
    }
    axb = axb.normalize();
    bxc = bxc.normalize();
    cxd = cxd.normalize();
    dxa = dxa.normalize();

    let alpha = angle_between(&dxa, &-axb);
    let beta = angle_between(&axb, &-bxc);
    let gamma = angle_between(&bxc, &-cxd);
    let delta = angle_between(&cxd, &-dxa);

    Float::abs(alpha + beta + gamma + delta - 2.0 * PI)
}

// pbrt-v4 helpers (math.h) used by SampleSphericalTriangle and its
// bilinear cosine-warp PDF; defined here so all of r4's spherical
// triangle sampling lives next to its other Monte Carlo helpers.

#[inline]
pub fn safe_sqrt(x: Float) -> Float {
    if x <= 0.0 {
        0.0
    } else {
        x.sqrt()
    }
}

#[inline]
pub fn difference_of_products(a: Float, b: Float, c: Float, d: Float) -> Float {
    let cd = c * d;
    let err = (-c).mul_add(d, cd);
    let dop = a.mul_add(b, -cd);
    dop + err
}

#[inline]
pub fn sum_of_products(a: Float, b: Float, c: Float, d: Float) -> Float {
    let cd = c * d;
    let err = c.mul_add(d, -cd);
    let sop = a.mul_add(b, cd);
    sop + err
}

#[inline]
pub fn angle_between(a: &Vector3f, b: &Vector3f) -> Float {
    // pbrt-v4 `AngleBetween` — numerically stable for tiny / near-pi angles.
    if a.dot(b) < 0.0 {
        PI - 2.0 * Float::asin((*a + *b).length() / 2.0)
    } else {
        2.0 * Float::asin((*b - *a).length() / 2.0)
    }
}

#[inline]
pub fn gram_schmidt(v: &Vector3f, w: &Vector3f) -> Vector3f {
    *v - v.dot(w) * *w
}

#[inline]
pub fn linear_pdf(x: Float, a: Float, b: Float) -> Float {
    if !(0.0..=1.0).contains(&x) || a + b == 0.0 {
        return 0.0;
    }
    2.0 * lerp(x, a, b) / (a + b)
}

#[inline]
pub fn sample_linear(u: Float, a: Float, b: Float) -> Float {
    if u == 0.0 && a == 0.0 {
        return 0.0;
    }
    let x = u * (a + b) / (a + Float::sqrt(cie_lerp(u, a * a, b * b)));
    Float::min(x, ONE_MINUS_EPSILON)
}

#[inline]
pub fn invert_linear_sample(x: Float, a: Float, b: Float) -> Float {
    x * (a * (2.0 - x) + b * x) / (a + b)
}

#[inline]
pub fn sample_tent(u: Float, radius: Float) -> Float {
    let (index, _, remapped) = sample_discrete(&[0.5, 0.5], u).unwrap();
    if index == 0 {
        -radius + radius * sample_linear(remapped, 0.0, 1.0)
    } else {
        radius * sample_linear(remapped, 1.0, 0.0)
    }
}

#[inline]
pub fn tent_pdf(x: Float, radius: Float) -> Float {
    if radius <= 0.0 || Float::abs(x) >= radius {
        0.0
    } else {
        1.0 / radius - Float::abs(x) / (radius * radius)
    }
}

#[inline]
pub fn invert_tent_sample(x: Float, radius: Float) -> Float {
    if x <= 0.0 {
        (1.0 - invert_linear_sample(-x / radius, 1.0, 0.0)) / 2.0
    } else {
        0.5 + invert_linear_sample(x / radius, 1.0, 0.0) / 2.0
    }
}

#[inline]
pub fn sample_discrete(weights: &[Float], u: Float) -> Option<(usize, Float, Float)> {
    if weights.is_empty() {
        return None;
    }
    let sum: Float = weights.iter().sum();
    if sum <= 0.0 {
        return None;
    }
    let mut up = u.clamp(0.0, ONE_MINUS_EPSILON) * sum;
    if up == sum {
        up = next_float_down(up);
    }
    let mut offset = 0usize;
    let mut accumulated = 0.0;
    while accumulated + weights[offset] <= up {
        accumulated += weights[offset];
        offset += 1;
        if offset == weights.len() {
            offset -= 1;
            break;
        }
    }
    let weight = weights[offset];
    let remapped = if weight > 0.0 {
        ((up - accumulated) / weight).min(ONE_MINUS_EPSILON)
    } else {
        0.0
    };
    Some((offset, weight / sum, remapped))
}

#[inline]
pub fn exponential_pdf(x: Float, a: Float) -> Float {
    debug_assert!(a > 0.0);
    a * (-a * x).exp()
}

#[inline]
pub fn sample_exponential(u: Float, a: Float) -> Float {
    debug_assert!(a > 0.0);
    -(1.0 - u).ln() / a
}

#[inline]
pub fn invert_exponential_sample(x: Float, a: Float) -> Float {
    debug_assert!(a > 0.0);
    1.0 - (-a * x).exp()
}

#[inline]
pub fn normal_pdf(x: Float, mu: Float, sigma: Float) -> Float {
    gaussian(x, mu, sigma)
}

#[inline]
pub fn sample_normal(u: Float, mu: Float, sigma: Float) -> Float {
    mu + SQRT_2 * sigma * erf_inv(2.0 * u - 1.0)
}

#[inline]
pub fn invert_normal_sample(x: Float, mu: Float, sigma: Float) -> Float {
    0.5 * (1.0 + erf((x - mu) / (sigma * SQRT_2)))
}

#[inline]
pub fn logistic_pdf(x: Float, s: Float) -> Float {
    let x = x.abs();
    (-x / s).exp() / (s * (1.0 + (-x / s).exp()).powi(2))
}

#[inline]
pub fn sample_logistic(u: Float, s: Float) -> Float {
    -s * (1.0 / u - 1.0).ln()
}

#[inline]
pub fn invert_logistic_sample(x: Float, s: Float) -> Float {
    1.0 / (1.0 + (-x / s).exp())
}

#[inline]
pub fn sample_two_normal(u: Point2f, mu: Float, sigma: Float) -> Point2f {
    let r2 = -2.0 * (1.0 - u.x).ln();
    let theta = 2.0 * PI * u.y;
    let r = r2.sqrt();
    Point2f::new(mu + sigma * r * theta.cos(), mu + sigma * r * theta.sin())
}

#[inline]
pub fn sample_trimmed_exponential(u: Float, c: Float, x_max: Float) -> Float {
    (1.0 - u * (1.0 - (-c * x_max).exp())).ln() / -c
}

#[inline]
pub fn trimmed_exponential_pdf(x: Float, c: Float, x_max: Float) -> Float {
    if x < 0.0 || x > x_max {
        return 0.0;
    }
    c * (-c * x).exp() / (1.0 - (-c * x_max).exp())
}

#[inline]
pub fn invert_trimmed_exponential_sample(x: Float, c: Float, x_max: Float) -> Float {
    debug_assert!(x >= 0.0 && x <= x_max);
    (1.0 - (-c * x).exp()) / (1.0 - (-c * x_max).exp())
}

#[inline]
pub fn smooth_step_pdf(x: Float, a: Float, b: Float) -> Float {
    if x < a || x > b {
        return 0.0;
    }
    let t = Float::clamp((x - a) / (b - a), 0.0, 1.0);
    (2.0 / (b - a)) * t * t * (3.0 - 2.0 * t)
}

#[inline]
pub fn sample_smooth_step(u: Float, a: Float, b: Float) -> Float {
    debug_assert!(a < b);
    let mut x = a + u * (b - a);
    for _ in 0..16 {
        let t = (x - a) / (b - a);
        let p = 2.0 * t * t * t - t * t * t * t;
        let derivative = smooth_step_pdf(x, a, b);
        if derivative == 0.0 {
            break;
        }
        let step = (p - u) / derivative;
        x = Float::clamp(x - step, a, b);
        if step.abs() < 1e-6 * (b - a) {
            break;
        }
    }
    x
}

#[inline]
pub fn invert_smooth_step_sample(x: Float, a: Float, b: Float) -> Float {
    let t = (x - a) / (b - a);
    2.0 * t * t * t - t * t * t * t
}

#[inline]
pub fn trimmed_logistic_pdf(x: Float, s: Float, a: Float, b: Float) -> Float {
    if x < a || x > b {
        return 0.0;
    }
    logistic_pdf(x, s) / (invert_logistic_sample(b, s) - invert_logistic_sample(a, s))
}

#[inline]
pub fn sample_trimmed_logistic(u: Float, s: Float, a: Float, b: Float) -> Float {
    debug_assert!(a < b);
    let pa = invert_logistic_sample(a, s);
    let pb = invert_logistic_sample(b, s);
    let x = sample_logistic(pa + u * (pb - pa), s);
    Float::clamp(x, a, b)
}

#[inline]
pub fn invert_trimmed_logistic_sample(x: Float, s: Float, a: Float, b: Float) -> Float {
    debug_assert!(a <= x && x <= b);
    let pa = invert_logistic_sample(a, s);
    (invert_logistic_sample(x, s) - pa) / (invert_logistic_sample(b, s) - pa)
}

#[inline]
pub fn sample_henyey_greenstein(wo: Vector3f, g: Float, u: Point2f) -> (Vector3f, Float) {
    let g = g.clamp(-0.99, 0.99);
    let cos_theta = if g.abs() < 1e-3 {
        1.0 - 2.0 * u.x
    } else {
        let sqr_term = (1.0 - g * g) / (1.0 + g - 2.0 * g * u.x);
        -(1.0 + g * g - sqr_term * sqr_term) / (2.0 * g)
    };
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * u.y;
    let (v1, v2) = coordinate_system(&wo);
    let wi = spherical_direction_axes(sin_theta, cos_theta, phi, &v1, &v2, &wo);
    let denom = 1.0 + g * g + 2.0 * g * cos_theta;
    let pdf = INV_4_PI * (1.0 - g * g) / (denom * denom.sqrt());
    (wi, pdf)
}

#[inline]
pub fn sample_spherical_rectangle(
    p_ref: Point3f,
    s: Point3f,
    ex: Vector3f,
    ey: Vector3f,
    u: Point2f,
) -> (Point3f, Float) {
    let ex_length = ex.length();
    let ey_length = ey.length();
    let mut frame = Frame::from_xy(ex / ex_length, ey / ey_length);
    let d_local = frame.to_local(s - p_ref);
    let mut z0 = d_local.z;
    if z0 > 0.0 {
        frame.z = -frame.z;
        z0 = -z0;
    }
    let x0 = d_local.x;
    let y0 = d_local.y;
    let x1 = x0 + ex_length;
    let y1 = y0 + ey_length;
    let v00 = Vector3f::new(x0, y0, z0);
    let v01 = Vector3f::new(x0, y1, z0);
    let v10 = Vector3f::new(x1, y0, z0);
    let v11 = Vector3f::new(x1, y1, z0);
    let n0 = Vector3f::cross(&v00, &v10).normalize();
    let n1 = Vector3f::cross(&v10, &v11).normalize();
    let n2 = Vector3f::cross(&v11, &v01).normalize();
    let n3 = Vector3f::cross(&v01, &v00).normalize();
    let g0 = angle_between(&-n0, &n1);
    let g1 = angle_between(&-n1, &n2);
    let g2 = angle_between(&-n2, &n3);
    let g3 = angle_between(&-n3, &n0);
    let solid_angle = g0 + g1 + g2 + g3 - 2.0 * PI;
    if solid_angle <= 0.0 || solid_angle < 1e-3 {
        let pdf = if solid_angle > 0.0 {
            1.0 / solid_angle
        } else {
            0.0
        };
        return (s + u.x * ex + u.y * ey, pdf);
    }
    let b0 = n0.z;
    let b1 = n2.z;
    let au = u.x * (g0 + g1 - 2.0 * PI) + (u.x - 1.0) * (g2 + g3);
    let fu = (au.cos() * b0 - b1) / au.sin();
    let mut cu = (1.0 as Float).copysign(fu) / (fu * fu + b0 * b0).sqrt();
    cu = cu.clamp(-(ONE_MINUS_EPSILON as Float), ONE_MINUS_EPSILON as Float);
    let xu = (-(cu * z0) / (1.0 - cu * cu).max(0.0).sqrt()).clamp(x0, x1);
    let dd = (xu * xu + z0 * z0).sqrt();
    let h0 = y0 / (dd * dd + y0 * y0).sqrt();
    let h1 = y1 / (dd * dd + y1 * y1).sqrt();
    let hv = h0 + u.y * (h1 - h0);
    let yv = if hv * hv < 1.0 - 1e-6 {
        hv * dd / (1.0 - hv * hv).sqrt()
    } else {
        y1
    };
    (
        p_ref + frame.from_local(Vector3f::new(xu, yv, z0)),
        1.0 / solid_angle,
    )
}

#[inline]
pub fn invert_spherical_rectangle_sample(
    p_ref: Point3f,
    s: Point3f,
    ex: Vector3f,
    ey: Vector3f,
    p_rect: Point3f,
) -> Point2f {
    let ex_length = ex.length();
    let ey_length = ey.length();
    let mut frame = Frame::from_xy(ex / ex_length, ey / ey_length);
    let d_local = frame.to_local(s - p_ref);
    let mut z0 = d_local.z;
    if z0 > 0.0 {
        frame.z = -frame.z;
        z0 = -z0;
    }
    let x0 = d_local.x;
    let y0 = d_local.y;
    let x1 = x0 + ex_length;
    let y1 = y0 + ey_length;
    let v00 = Vector3f::new(x0, y0, z0);
    let v01 = Vector3f::new(x0, y1, z0);
    let v10 = Vector3f::new(x1, y0, z0);
    let v11 = Vector3f::new(x1, y1, z0);
    let n0 = Vector3f::cross(&v00, &v10).normalize();
    let n1 = Vector3f::cross(&v10, &v11).normalize();
    let n2 = Vector3f::cross(&v11, &v01).normalize();
    let n3 = Vector3f::cross(&v01, &v00).normalize();
    let g0 = angle_between(&-n0, &n1);
    let g1 = angle_between(&-n1, &n2);
    let g2 = angle_between(&-n2, &n3);
    let g3 = angle_between(&-n3, &n0);
    let solid_angle = g0 + g1 + g2 + g3 - 2.0 * PI;
    if solid_angle < 1e-3 {
        let pq = p_rect - s;
        return Point2f::new(
            Vector3f::dot(&pq, &ex) / ex.length_squared(),
            Vector3f::dot(&pq, &ey) / ey.length_squared(),
        );
    }
    let local = frame.to_local(p_rect - p_ref);
    let mut xu = local.x.clamp(x0, x1);
    if xu == 0.0 {
        xu = 1e-10;
    }
    let yv = local.y;
    let z0_squared = z0 * z0;
    let b0 = n0.z;
    let b1 = n2.z;
    let inverse_cu_squared = 1.0 + z0_squared / (xu * xu);
    let fu_squared = inverse_cu_squared - b0 * b0;
    let fu = fu_squared.max(0.0).sqrt().copysign(xu);
    let root = (difference_of_products(b0, b0, b1, b1) + fu_squared)
        .max(0.0)
        .sqrt();
    let mut au = (-(b1 * fu) - (b0 * root).copysign(fu * b0)).atan2(b0 * b1 - root * fu.abs());
    if au > 0.0 {
        au -= 2.0 * PI;
    }
    let u0 = (au + g2 + g3) / solid_angle;
    let dd_squared = xu * xu + z0_squared;
    let h0 = y0 / (dd_squared + y0 * y0).sqrt();
    let h1 = y1 / (dd_squared + y1 * y1).sqrt();
    let yv_squared = yv * yv;
    let delta_h = h0 - h1;
    let term =
        delta_h.abs() * (yv_squared * (dd_squared + yv_squared)).sqrt() / (dd_squared + yv_squared);
    let denom = delta_h * delta_h;
    let candidates = [
        (difference_of_products(h0, h0, h0, h1) - term) / denom,
        (difference_of_products(h0, h0, h0, h1) + term) / denom,
    ];
    let mut best = candidates[0];
    let mut best_error = Float::INFINITY;
    for candidate in candidates {
        let hv = h0 + candidate * (h1 - h0);
        let y = hv * dd_squared.sqrt() / (1.0 - hv * hv).max(1e-12).sqrt();
        let error = (y - yv).abs();
        if error < best_error {
            best_error = error;
            best = candidate;
        }
    }
    Point2f::new(u0.clamp(0.0, 1.0), best.clamp(0.0, 1.0))
}

#[inline]
pub fn sample_catmull_rom(
    nodes: &[Float],
    values: &[Float],
    cdf: &[Float],
    sample: Float,
) -> Option<(Float, Float, Float)> {
    if nodes.len() < 2 || nodes.len() != values.len() || values.len() != cdf.len() {
        return None;
    }
    let mut u = sample * cdf[cdf.len() - 1];
    let index = find_interval(cdf, &|v, i| v[i] <= u);
    let x0 = nodes[index];
    let x1 = nodes[index + 1];
    let width = x1 - x0;
    let f0 = values[index];
    let f1 = values[index + 1];
    let d0 = if index > 0 {
        width * (f1 - values[index - 1]) / (x1 - nodes[index - 1])
    } else {
        f1 - f0
    };
    let d1 = if index + 2 < nodes.len() {
        width * (values[index + 2] - f0) / (nodes[index + 2] - x0)
    } else {
        f1 - f0
    };
    u = (u - cdf[index]) / width;
    let mut a = 0.0;
    let mut b = 1.0;
    let mut t = if f0 != f1 {
        (f0 - (f0 * f0 + 2.0 * u * (f1 - f0)).max(0.0).sqrt()) / (f0 - f1)
    } else {
        u / f0
    };
    let mut fhat = f0;
    for _ in 0..32 {
        if !(a..=b).contains(&t) {
            t = 0.5 * (a + b);
        }
        let integral = t
            * (f0
                + t * (0.5 * d0
                    + t * ((1.0 / 3.0) * (-2.0 * d0 - d1) + f1 - f0
                        + t * (0.25 * (d0 + d1) + 0.5 * (f0 - f1)))));
        fhat = f0
            + t * (d0 + t * (-2.0 * d0 - d1 + 3.0 * (f1 - f0) + t * (d0 + d1 + 2.0 * (f0 - f1))));
        if (integral - u).abs() < 1e-6 || b - a < 1e-6 {
            break;
        }
        if integral < u {
            a = t;
        } else {
            b = t;
        }
        if fhat == 0.0 {
            t = 0.5 * (a + b);
        } else {
            t -= (integral - u) / fhat;
        }
    }
    Some((x0 + width * t, fhat, fhat / cdf[cdf.len() - 1]))
}

pub fn bilinear_pdf(p: Point2f, w: &[Float; 4]) -> Float {
    if p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 {
        return 0.0;
    }
    let wsum = w[0] + w[1] + w[2] + w[3];
    if wsum == 0.0 {
        return 1.0;
    }
    4.0 * ((1.0 - p.x) * (1.0 - p.y) * w[0]
        + p.x * (1.0 - p.y) * w[1]
        + (1.0 - p.x) * p.y * w[2]
        + p.x * p.y * w[3])
        / wsum
}

pub fn sample_bilinear(u: Point2f, w: &[Float; 4]) -> Point2f {
    let py = sample_linear(u.y, w[0] + w[1], w[2] + w[3]);
    let lerp = cie_lerp;
    let px = sample_linear(u.x, lerp(py, w[0], w[2]), lerp(py, w[1], w[3]));
    Point2f::new(px, py)
}

pub fn invert_bilinear_sample(p: Point2f, w: &[Float; 4]) -> Point2f {
    let lerp = cie_lerp;
    Point2f::new(
        invert_linear_sample(p.x, lerp(p.y, w[0], w[2]), lerp(p.y, w[1], w[3])),
        invert_linear_sample(p.y, w[0] + w[1], w[2] + w[3]),
    )
}

/// pbrt-v4 `SampleSphericalTriangle` — uniformly sample the spherical
/// triangle subtended at `p` by the three vertices `v` and return the
/// barycentric coordinates of the sampled point plus, optionally, the
/// solid-angle PDF (= `1 / area`).
pub fn sample_spherical_triangle(
    v: &[Point3f; 3],
    p: Point3f,
    u: Point2f,
) -> (Option<[Float; 3]>, Float) {
    let mut a = v[0] - p;
    let mut b = v[1] - p;
    let mut c = v[2] - p;
    if a.length_squared() <= 0.0 || b.length_squared() <= 0.0 || c.length_squared() <= 0.0 {
        return (None, 0.0);
    }
    a = a.normalize();
    b = b.normalize();
    c = c.normalize();

    let mut n_ab = Vector3f::cross(&a, &b);
    let mut n_bc = Vector3f::cross(&b, &c);
    let mut n_ca = Vector3f::cross(&c, &a);
    if n_ab.length_squared() == 0.0 || n_bc.length_squared() == 0.0 || n_ca.length_squared() == 0.0
    {
        return (None, 0.0);
    }
    n_ab = n_ab.normalize();
    n_bc = n_bc.normalize();
    n_ca = n_ca.normalize();

    let alpha = angle_between(&n_ab, &(-n_ca));
    let beta = angle_between(&n_bc, &(-n_ab));
    let gamma = angle_between(&n_ca, &(-n_bc));

    let a_pi = alpha + beta + gamma;
    let area = a_pi - PI;
    let pdf = if area <= 0.0 { 0.0 } else { 1.0 / area };

    let lerp = cie_lerp;
    let ap_pi = lerp(u.x, PI, a_pi);

    let cos_alpha = alpha.cos();
    let sin_alpha = alpha.sin();
    let sin_phi = ap_pi.sin() * cos_alpha - ap_pi.cos() * sin_alpha;
    let cos_phi = ap_pi.cos() * cos_alpha + ap_pi.sin() * sin_alpha;
    let k1 = cos_phi + cos_alpha;
    let k2 = sin_phi - sin_alpha * a.dot(&b);
    let denom = sum_of_products(k2, sin_phi, k1, cos_phi) * sin_alpha;
    let mut cos_bp = if denom == 0.0 {
        0.0
    } else {
        (k2 + difference_of_products(k2, cos_phi, k1, sin_phi) * cos_alpha) / denom
    };
    if cos_bp.is_nan() {
        cos_bp = 0.0;
    }
    cos_bp = cos_bp.clamp(-1.0, 1.0);

    let sin_bp = safe_sqrt(1.0 - cos_bp * cos_bp);
    let cp = cos_bp * a + sin_bp * gram_schmidt(&c, &a).normalize();

    let cos_theta = 1.0 - u.y * (1.0 - cp.dot(&b));
    let sin_theta = safe_sqrt(1.0 - cos_theta * cos_theta);
    let w = cos_theta * b + sin_theta * gram_schmidt(&cp, &b).normalize();

    let e1 = v[1] - v[0];
    let e2 = v[2] - v[0];
    let s1 = Vector3f::cross(&w, &e2);
    let divisor = s1.dot(&e1);
    if divisor == 0.0 {
        return (Some([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), pdf);
    }
    let inv_divisor = 1.0 / divisor;
    let s = p - v[0];
    let mut b1 = s.dot(&s1) * inv_divisor;
    let mut b2 = w.dot(&Vector3f::cross(&s, &e1)) * inv_divisor;
    b1 = b1.clamp(0.0, 1.0);
    b2 = b2.clamp(0.0, 1.0);
    if b1 + b2 > 1.0 {
        let s = b1 + b2;
        b1 /= s;
        b2 /= s;
    }
    (Some([1.0 - b1 - b2, b1, b2]), pdf)
}

/// pbrt-v4 `InvertSphericalTriangleSample` — given a direction `w`
/// known to point into the spherical triangle at `p`, recover the
/// `(u0, u1)` Point2f that `sample_spherical_triangle` would consume to
/// produce that same direction. Used inside the cosine-warp PDF path
/// of `Triangle::pdf_from` to map an `wi` back to bilinear domain.
pub fn invert_spherical_triangle_sample(v: &[Point3f; 3], p: Point3f, w: Vector3f) -> Point2f {
    let mut a = v[0] - p;
    let mut b = v[1] - p;
    let mut c = v[2] - p;
    if a.length_squared() <= 0.0 || b.length_squared() <= 0.0 || c.length_squared() <= 0.0 {
        return Point2f::new(0.5, 0.5);
    }
    a = a.normalize();
    b = b.normalize();
    c = c.normalize();

    let mut n_ab = Vector3f::cross(&a, &b);
    let mut n_bc = Vector3f::cross(&b, &c);
    let mut n_ca = Vector3f::cross(&c, &a);
    if n_ab.length_squared() == 0.0 || n_bc.length_squared() == 0.0 || n_ca.length_squared() == 0.0
    {
        return Point2f::new(0.5, 0.5);
    }
    n_ab = n_ab.normalize();
    n_bc = n_bc.normalize();
    n_ca = n_ca.normalize();

    let alpha = angle_between(&n_ab, &(-n_ca));
    let beta = angle_between(&n_bc, &(-n_ab));
    let gamma = angle_between(&n_ca, &(-n_bc));

    let bw = Vector3f::cross(&b, &w);
    let ca = Vector3f::cross(&c, &a);
    let mut cp = Vector3f::cross(&bw, &ca);
    if cp.length_squared() == 0.0 {
        return Point2f::new(0.5, 0.5);
    }
    cp = cp.normalize();
    if cp.dot(&(a + c)) < 0.0 {
        cp = -cp;
    }

    let u0 = if a.dot(&cp) > 0.9999984769 {
        0.0
    } else {
        let mut n_cpb = Vector3f::cross(&cp, &b);
        let mut n_acp = Vector3f::cross(&a, &cp);
        if n_cpb.length_squared() == 0.0 || n_acp.length_squared() == 0.0 {
            return Point2f::new(0.5, 0.5);
        }
        n_cpb = n_cpb.normalize();
        n_acp = n_acp.normalize();
        let area_p = alpha + angle_between(&n_ab, &n_cpb) + angle_between(&n_acp, &(-n_cpb)) - PI;
        let area = alpha + beta + gamma - PI;
        if area == 0.0 {
            0.0
        } else {
            area_p / area
        }
    };
    let denom = 1.0 - cp.dot(&b);
    let u1 = if denom == 0.0 {
        0.0
    } else {
        (1.0 - w.dot(&b)) / denom
    };
    Point2f::new(u0.clamp(0.0, 1.0), u1.clamp(0.0, 1.0))
}

pub fn shuffle_array<T: Copy>(samp: &mut [T], count: usize, dim: u32, rng: &mut RNG) {
    let dim = dim as usize;
    for i in 0..count {
        let other = i + rng.uniform_uint32_threshold((count - i) as u32) as usize;
        for j in 0..dim {
            let a = dim * i + j;
            let b = dim * other + j;
            samp.swap(a, b);
            //std::mem::swap(&mut samp[a], &mut samp[b]);
        }
    }
}

pub fn stratified_sample_1d(samples: &mut [Float], nsamples: usize, rng: &mut RNG, jitter: bool) {
    let inv_nsamples = 1.0 / (nsamples as Float);
    for i in 0..nsamples {
        let delta = if jitter { rng.uniform_float() } else { 0.5 };
        samples[i] = Float::min(((i as Float) + delta) * inv_nsamples, ONE_MINUS_EPSILON);
    }
}

pub fn stratified_sample_2d(
    samples: &mut [Point2f],
    nx: usize,
    ny: usize,
    rng: &mut RNG,
    jitter: bool,
) {
    let dx = 1.0 / (nx as Float);
    let dy = 1.0 / (ny as Float);
    for y in 0..ny {
        for x in 0..nx {
            let i = y * nx + x;
            let jx = if jitter { rng.uniform_float() } else { 0.5 };
            let jy = if jitter { rng.uniform_float() } else { 0.5 };
            let xx = Float::min(((x as Float) + jx) * dx, ONE_MINUS_EPSILON);
            let yy = Float::min(((y as Float) + jy) * dy, ONE_MINUS_EPSILON);
            samples[i] = Point2f::new(xx, yy);
        }
    }
}

pub fn latin_hypercube(samples: &mut [Float], nsamples: usize, ndim: usize, rng: &mut RNG) {
    // Generate LHS samples along diagonal
    let inv_nsamples = 1.0 / (nsamples as Float);
    for i in 0..nsamples {
        for j in 0..ndim {
            let sj = (i as Float + (rng.uniform_float())) * inv_nsamples;
            samples[ndim * i + j] = Float::min(sj, ONE_MINUS_EPSILON);
        }
    }
    // Permute LHS samples in each dimension
    for i in 0..ndim {
        for j in 0..nsamples {
            let other = j + rng.uniform_uint32_threshold((nsamples - j) as u32) as usize;
            let a = ndim * j + i;
            let b = ndim * other + i;
            samples.swap(a, b);
        }
    }
}

pub fn latin_hypercube_1d(samples: &mut [Float], nsamples: usize, rng: &mut RNG) {
    latin_hypercube(samples, nsamples, 1, rng);
}

pub fn latin_hypercube_2d(samples: &mut [Point2f], nsamples: usize, rng: &mut RNG) {
    let mut v = vec![0.0; samples.len() * 2];
    for i in 0..samples.len() {
        v[2 * i + 0] = samples[i].x;
        v[2 * i + 1] = samples[i].y;
    }
    latin_hypercube(&mut v, nsamples, 2, rng);
    for i in 0..samples.len() {
        samples[i].x = v[2 * i + 0];
        samples[i].y = v[2 * i + 1];
    }
}

pub fn uniform_sample_hemisphere(u: &Point2f) -> Vector3f {
    let z = u[0];
    let r = Float::sqrt(Float::max(0.0, 1.0 - z * z));
    let phi = 2.0 * PI * u[1];
    return Vector3f::new(r * Float::cos(phi), r * Float::sin(phi), z);
}

pub fn uniform_hemisphere_pdf() -> Float {
    return INV_2_PI;
}

#[inline]
pub fn uniform_sample_sphere(u: &Point2f) -> Vector3f {
    let z = 1.0 - 2.0 * u[0];
    let r = Float::sqrt(Float::max(0.0, 1.0 - z * z));
    let phi = 2.0 * PI * u[1];
    return Vector3f::new(r * Float::cos(phi), r * Float::sin(phi), z);
}

#[inline]
pub fn uniform_sphere_pdf() -> Float {
    return INV_4_PI;
}

pub fn uniform_sample_triangle(u: &Point2f) -> Point2f {
    let su0 = Float::sqrt(u[0]);
    return Point2f::new(1.0 - su0, u[1] * su0);
}

pub fn concentric_sample_disk(u: &Point2f) -> Point2f {
    // Map uniform random numbers to $[-1,1]^2$
    let u_offset = *u * 2.0 - Vector2f::new(1.0, 1.0);

    // Handle degeneracy at the origin
    if u_offset.x == 0.0 && u_offset.y == 0.0 {
        return Point2f::zero();
    }

    // Apply concentric mapping to point
    if Float::abs(u_offset.x) > Float::abs(u_offset.y) {
        let r = u_offset.x;
        let theta = PI_OVER_4 * (u_offset.y / u_offset.x);
        return Point2f::new(r * Float::cos(theta), r * Float::sin(theta));
    } else {
        let r = u_offset.y;
        let theta = PI_OVER_2 - PI_OVER_4 * (u_offset.x / u_offset.y);
        return Point2f::new(r * Float::cos(theta), r * Float::sin(theta));
    }
}

#[inline]
pub fn uniform_sample_disk_polar(u: &Point2f) -> Point2f {
    let r = Float::sqrt(u[0]);
    let theta = 2.0 * PI * u[1];
    Point2f::new(r * Float::cos(theta), r * Float::sin(theta))
}

#[inline]
pub fn invert_uniform_disk_polar_sample(p: &Point2f) -> Point2f {
    let mut phi = Float::atan2(p.y, p.x);
    if phi < 0.0 {
        phi += 2.0 * PI;
    }
    Point2f::new(p.x * p.x + p.y * p.y, phi / (2.0 * PI))
}

#[inline]
pub fn invert_uniform_disk_concentric_sample(p: &Point2f) -> Point2f {
    let theta = Float::atan2(p.y, p.x);
    let r = Float::sqrt(p.x * p.x + p.y * p.y);
    let (x, y) = if Float::abs(theta) < PI_OVER_4 || Float::abs(theta) > 3.0 * PI_OVER_4 {
        let signed_r = if p.x < 0.0 { -r } else { r };
        let x = signed_r;
        let y = if p.x < 0.0 {
            if p.y < 0.0 {
                (PI + theta) * signed_r / PI_OVER_4
            } else {
                (theta - PI) * signed_r / PI_OVER_4
            }
        } else {
            theta * signed_r / PI_OVER_4
        };
        (x, y)
    } else {
        let signed_r = if p.y < 0.0 { -r } else { r };
        let y = signed_r;
        let x = if p.y < 0.0 {
            -(PI_OVER_2 + theta) * signed_r / PI_OVER_4
        } else {
            (PI_OVER_2 - theta) * signed_r / PI_OVER_4
        };
        (x, y)
    };
    Point2f::new((x + 1.0) / 2.0, (y + 1.0) / 2.0)
}

#[inline]
pub fn invert_uniform_hemisphere_sample(w: &Vector3f) -> Point2f {
    let mut phi = Float::atan2(w.y, w.x);
    if phi < 0.0 {
        phi += 2.0 * PI;
    }
    Point2f::new(w.z, phi / (2.0 * PI))
}

#[inline]
pub fn invert_uniform_sphere_sample(w: &Vector3f) -> Point2f {
    let mut phi = Float::atan2(w.y, w.x);
    if phi < 0.0 {
        phi += 2.0 * PI;
    }
    Point2f::new((1.0 - w.z) / 2.0, phi / (2.0 * PI))
}

#[inline]
pub fn invert_cosine_hemisphere_sample(w: &Vector3f) -> Point2f {
    invert_uniform_disk_concentric_sample(&Point2f::new(w.x, w.y))
}

#[inline]
pub fn uniform_cone_pdf(cos_theta_max: Float) -> Float {
    return 1.0 / (2.0 * PI * (1.0 - cos_theta_max));
}

pub fn uniform_sample_cone(u: &Point2f, cos_theta_max: Float) -> Vector3f {
    let cos_theta = (1.0 - u[0]) + u[0] * cos_theta_max;
    let sin_theta = Float::sqrt(1.0 - cos_theta * cos_theta);
    let phi = u[1] * 2.0 * PI;
    return Vector3f::new(
        Float::cos(phi) * sin_theta,
        Float::sin(phi) * sin_theta,
        cos_theta,
    );
}

#[inline]
pub fn invert_uniform_cone_sample(w: &Vector3f, cos_theta_max: Float) -> Point2f {
    let mut phi = Float::atan2(w.y, w.x);
    if phi < 0.0 {
        phi += 2.0 * PI;
    }
    Point2f::new((w.z - 1.0) / (cos_theta_max - 1.0), phi / (2.0 * PI))
}

#[inline]
pub fn uniform_sample_hemisphere_concentric(u: &Point2f) -> Vector3f {
    let d = concentric_sample_disk(u);
    let z = 1.0 - d.x * d.x - d.y * d.y;
    let scale = Float::sqrt(Float::max(0.0, 2.0 - d.x * d.x - d.y * d.y));
    Vector3f::new(d.x * scale, d.y * scale, z)
}

pub fn cosine_sample_hemisphere(u: &Point2f) -> Vector3f {
    let d = concentric_sample_disk(u);
    let z = Float::sqrt(Float::max(0.0, 1.0 - d.x * d.x - d.y * d.y));
    return Vector3f::new(d.x, d.y, z);
}

pub fn cosine_hemisphere_pdf(cos_theta: Float) -> Float {
    return cos_theta * INV_PI;
}

#[inline]
pub fn balance_heuristic(n_f: i32, f_pdf: Float, n_g: i32, g_pdf: Float) -> Float {
    let f = n_f as Float * f_pdf;
    let g = n_g as Float * g_pdf;
    if f + g == 0.0 {
        return 0.0;
    }
    f / (f + g)
}

#[inline]
pub fn power_heuristic(n_f: i32, f_pdf: Float, n_g: i32, g_pdf: Float) -> Float {
    let f = n_f as Float * f_pdf;
    let g = n_g as Float * g_pdf;
    let denom = f * f + g * g;
    if denom == 0.0 {
        return 0.0;
    }
    (f * f) / denom
}
