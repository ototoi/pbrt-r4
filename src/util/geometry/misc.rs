use crate::util::base::*;
use crate::util::misc::*;

#[inline]
pub fn offset_ray_origin(p: &Point3f, p_error: &Vector3f, n: &Normal3f, w: &Vector3f) -> Point3f {
    // r4 carries explicit absolute error bounds instead of v4's Point3fi
    // intervals; use the same first-order bound as v4.
    let d = Vector3f::dot(&n.abs(), p_error);

    let mut offset = d * n.clone();
    if Vector3f::dot(w, n) < 0.0 {
        offset = -offset;
    }
    let po = *p + offset;
    let mut a = [po.x, po.y, po.z];

    for i in 0..3 {
        if offset[i] > 0.0 {
            a[i] = next_float_up(a[i]);
        } else if offset[i] < 0.0 {
            a[i] = next_float_down(a[i]);
        }
    }
    return Point3f::new(a[0], a[1], a[2]);
}

#[inline]
pub fn face_forward(n: &Vector3f, v: &Vector3f) -> Vector3f {
    if Vector3f::dot(n, v) < 0.0 {
        return *n * -1.0;
    } else {
        return *n;
    }
}

#[inline]
pub fn max_dimension(v: &Vector3f) -> i32 {
    if v.x > v.y {
        if v.x > v.z {
            return 0;
        } else {
            return 2;
        }
    } else {
        if v.y > v.z {
            return 1;
        } else {
            return 2;
        }
    }
}

#[inline]
pub fn permute(v: &Vector3f, x: usize, y: usize, z: usize) -> Vector3f {
    return Vector3f::new(v[x], v[y], v[z]);
}

#[inline]
pub fn max_component(v: &Vector3f) -> Float {
    return Float::max(v.x, Float::max(v.y, v.z));
}

#[inline]
pub fn coordinate_system(v1: &Vector3f) -> (Vector3f, Vector3f) {
    let v2 = if Float::abs(v1.x) > Float::abs(v1.y) {
        Vector3f::new(-v1.z, 0.0, v1.x) / Float::sqrt(v1.x * v1.x + v1.z * v1.z)
    } else {
        Vector3f::new(0.0, v1.z, -v1.y) / Float::sqrt(v1.y * v1.y + v1.z * v1.z)
    };
    let v3 = Vector3f::cross(v1, &v2).normalize();
    return (v2, v3);
}

#[inline]
pub fn spherical_direction(sin_theta: Float, cos_theta: Float, phi: Float) -> Vector3f {
    return Vector3f::new(
        sin_theta * Float::cos(phi),
        sin_theta * Float::sin(phi),
        cos_theta,
    );
}

#[inline]
pub fn spherical_direction_axes(
    sin_theta: Float,
    cos_theta: Float,
    phi: Float,
    x: &Vector3f,
    y: &Vector3f,
    z: &Vector3f,
) -> Vector3f {
    return (sin_theta * Float::cos(phi) * *x)
        + (sin_theta * Float::sin(phi) * *y)
        + (cos_theta * *z);
}

#[inline]
pub fn spherical_theta(v: &Vector3f) -> Float {
    return Float::acos(Float::clamp(v.z, -1.0, 1.0));
}

#[inline]
pub fn spherical_phi(v: &Vector3f) -> Float {
    let p = Float::atan2(v.y, v.x);
    return if p < 0.0 { p + 2.0 * PI } else { p };
}

/// pbrt-v4 `EqualAreaSquareToSphere` (util/math.cpp:292-314):
/// the inverse of [`equal_area_sphere_to_square`]. Maps a point in
/// `[0, 1]^2` to a unit direction on the sphere such that the
/// determinant of the Jacobian is constant (`4π` over the whole
/// square), so a uniform sampler on the square induces a uniform
/// sampler on the sphere.
pub fn equal_area_square_to_sphere(p: &Point2f) -> Vector3f {
    debug_assert!((0.0..=1.0).contains(&p.x) && (0.0..=1.0).contains(&p.y));
    let u = 2.0 * p.x - 1.0;
    let v = 2.0 * p.y - 1.0;
    let up = u.abs();
    let vp = v.abs();

    let signed_distance = 1.0 - (up + vp);
    let d = signed_distance.abs();
    let r = 1.0 - d;

    let phi = if r == 0.0 { 1.0 } else { (vp - up) / r + 1.0 } * (PI / 4.0);
    let z = (1.0 - r * r).copysign(signed_distance);

    let cos_phi = phi.cos().copysign(u);
    let sin_phi = phi.sin().copysign(v);
    let r_safe = (2.0 - r * r).max(0.0).sqrt();
    Vector3f::new(cos_phi * r * r_safe, sin_phi * r * r_safe, z)
}

/// pbrt-v4 `EqualAreaSphereToSquare` (util/math.cpp:317-361):
/// maps a unit direction back to `[0, 1]^2`. Inverse of
/// [`equal_area_square_to_sphere`].
pub fn equal_area_sphere_to_square(d: &Vector3f) -> Point2f {
    let x = d.x.abs();
    let y = d.y.abs();
    let z = d.z.abs();

    let r = (1.0 - z).max(0.0).sqrt();
    let a = x.max(y);
    let b = x.min(y);
    let b_over_a = if a == 0.0 { 0.0 } else { b / a };

    // 6th-degree minimax polynomial approximation of atan(b/a) * 2/π
    // over b/a ∈ [0, 1] (Clarberg "Fast Equal-Area Mapping of the
    // (Hemi)Sphere using SIMD"). Coefficients match v4 verbatim.
    const T1: Float = 0.406_758_566_246_788_5e-5;
    const T2: Float = 0.636_226_545_274_016_1;
    const T3: Float = 0.615_720_178_982_802_2e-2;
    const T4: Float = -0.247_333_733_281_268_94;
    const T5: Float = 0.881_770_664_775_316_3e-1;
    const T6: Float = 0.419_038_818_029_165_75e-1;
    const T7: Float = -0.251_390_972_343_483_53e-1;
    let bb = b_over_a;
    let phi = ((((((T7 * bb + T6) * bb + T5) * bb + T4) * bb + T3) * bb + T2) * bb + T1);

    let phi = if x < y { 1.0 - phi } else { phi };

    let mut v = phi * r;
    let mut u = r - v;

    if d.z < 0.0 {
        std::mem::swap(&mut u, &mut v);
        u = 1.0 - u;
        v = 1.0 - v;
    }

    u = u.copysign(d.x);
    v = v.copysign(d.y);

    Point2f::new(0.5 * (u + 1.0), 0.5 * (v + 1.0))
}
