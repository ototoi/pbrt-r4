use crate::util::base::{Float, Point3f, Vector3f};
use crate::util::geometry::Bounds3f;
use crate::util::sampling::sampling::safe_sqrt;
use crate::util::transform::Transform;

/// pbrt-v4 `DirectionCone` (util/vecmath.h:1099). A cone of directions
/// expressed by an axis `w` and `cosTheta` = cosine of the half-angle.
/// Used by the light-BVH sampler to bound the set of emission
/// directions of a (possibly aggregate) light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionCone {
    pub w: Vector3f,
    pub cos_theta: Float,
}

impl DirectionCone {
    pub fn new(w: Vector3f, cos_theta: Float) -> Self {
        Self {
            w: w.normalize(),
            cos_theta,
        }
    }

    pub fn from_direction(w: Vector3f) -> Self {
        Self::new(w, 1.0)
    }

    pub fn empty() -> Self {
        // pbrt-v4 marks the empty cone with `cosTheta = Infinity`.
        Self {
            w: Vector3f::new(0.0, 0.0, 1.0),
            cos_theta: Float::INFINITY,
        }
    }

    pub fn entire_sphere() -> Self {
        Self {
            w: Vector3f::new(0.0, 0.0, 1.0),
            cos_theta: -1.0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cos_theta == Float::INFINITY
    }
}

/// pbrt-v4 `DirectionCone::Union(a, b)` (util/vecmath.cpp).
pub fn union_direction_cones(a: &DirectionCone, b: &DirectionCone) -> DirectionCone {
    if a.is_empty() {
        return *b;
    }
    if b.is_empty() {
        return *a;
    }

    let theta_a = safe_acos(a.cos_theta);
    let theta_b = safe_acos(b.cos_theta);
    let theta_d = angle_between(a.w, b.w);

    if Float::min(theta_d + theta_b, std::f32::consts::PI as Float) <= theta_a {
        return *a;
    }
    if Float::min(theta_d + theta_a, std::f32::consts::PI as Float) <= theta_b {
        return *b;
    }

    let theta_o = (theta_a + theta_d + theta_b) * 0.5;
    if theta_o >= std::f32::consts::PI as Float {
        return DirectionCone::entire_sphere();
    }

    let theta_r = theta_o - theta_a;
    let wr = Vector3f::cross(&a.w, &b.w);
    if wr.length_squared() == 0.0 {
        return DirectionCone::entire_sphere();
    }
    let wr = wr.normalize();
    // Rotate `a.w` around `wr` by `theta_r` (`Transform::rotate` takes
    // degrees, matching v4's `Rotate(Degrees(theta_r), wr)(a.w)`).
    let rot = Transform::rotate(theta_r.to_degrees(), wr.x, wr.y, wr.z);
    let w = rot.transform_vector(&a.w);
    DirectionCone::new(w, theta_o.cos())
}

/// pbrt-v4 `BoundSubtendedDirections(b, p)` (util/vecmath.h:1185).
/// Cone bounding the directions from `p` to points in the bounding box.
pub fn bound_subtended_directions(b: &Bounds3f, p: &Point3f) -> DirectionCone {
    let (center_v, radius) = b.bounding_sphere();
    let p_center = Point3f::new(center_v.x, center_v.y, center_v.z);
    let d2 = (p_center - *p).length_squared();
    // At or inside the bounding sphere there is no finite cone around the
    // center direction that conservatively contains the box.  The equality
    // case matters for degenerate bounds: normalizing center - p would
    // otherwise normalize the zero vector and return a cone full of NaNs.
    if d2 <= radius * radius {
        return DirectionCone::entire_sphere();
    }
    let w = (p_center - *p).normalize();
    let sin2 = radius * radius / d2;
    let cos_theta_max = safe_sqrt(1.0 - sin2);
    DirectionCone::new(w, cos_theta_max)
}

fn safe_acos(x: Float) -> Float {
    Float::acos(x.clamp(-1.0, 1.0))
}

fn angle_between(a: Vector3f, b: Vector3f) -> Float {
    // pbrt-v4 `AngleBetween` (vecmath.h:1023): numerically stable variant
    // that picks a different branch depending on the sign of `a·b`.
    if a.dot(&b) < 0.0 {
        std::f32::consts::PI as Float - 2.0 * Float::asin(((a + b).length() * 0.5).min(1.0))
    } else {
        2.0 * Float::asin(((b - a).length() * 0.5).min(1.0))
    }
}
