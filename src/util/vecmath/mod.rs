pub mod frame;

pub use frame::*;

use crate::util::base::{Float, Vector3f, PI};
use crate::util::math::safe_asin;

/// pbrt-v4 `AngleBetween` (`util/vecmath.h:986`): numerically stable angle
/// between two directions. The `dot < 0` branch keeps obtuse angles up to
/// 180 degrees, which a naive `acos(dot)` loses to rounding and a clamped
/// dot product caps at 90 degrees.
pub fn angle_between(a: Vector3f, b: Vector3f) -> Float {
    if a.dot(&b) < 0.0 {
        PI - 2.0 * safe_asin((a + b).length() / 2.0)
    } else {
        2.0 * safe_asin((b - a).length() / 2.0)
    }
}
