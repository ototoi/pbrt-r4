use crate::util::base::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

#[derive(Debug, Copy, Clone)]
pub struct FilterSample {
    pub p: Point2f,
    pub weight: Float,
}
