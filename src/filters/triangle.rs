use crate::filters::*;
use crate::paramdict::*;

use crate::util::base::*;
use crate::util::error::*;
use crate::util::math::*;
use crate::util::sampling::sample_tent;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

pub struct TriangleFilter {
    pub base: BaseFilter,
}

impl TriangleFilter {
    pub fn new(radius: &Vector2f) -> Self {
        TriangleFilter {
            base: BaseFilter::new(radius),
        }
    }

    pub fn create(params: &ParameterDictionary) -> Result<TriangleFilter, PbrtError> {
        let xw = params.get_one_float("xradius", 2.0);
        let yw = params.get_one_float("yradius", 2.0);
        Ok(TriangleFilter::new(&Vector2f::new(xw, yw)))
    }

    pub fn evaluate(&self, p: &Point2f) -> Float {
        Float::max(0.0, self.base.radius.x - Float::abs(p.x))
            * Float::max(0.0, self.base.radius.y - Float::abs(p.y))
    }

    pub fn integral(&self) -> Float {
        self.base.radius.x * self.base.radius.x * self.base.radius.y * self.base.radius.y
    }

    pub fn sample(&self, u: &Point2f) -> FilterSample {
        let p = Point2f::new(
            sample_tent(u.x, self.base.radius.x),
            sample_tent(u.y, self.base.radius.y),
        );
        FilterSample { p, weight: 1.0 }
    }
}

impl Clone for TriangleFilter {
    fn clone(&self) -> Self {
        TriangleFilter { base: self.base }
    }
}
