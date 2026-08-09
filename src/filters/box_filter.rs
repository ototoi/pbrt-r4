use crate::filters::*;
use crate::paramdict::*;

use crate::util::base::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

pub struct BoxFilter {
    pub base: BaseFilter,
}

impl BoxFilter {
    pub fn new(radius: &Vector2f) -> Self {
        BoxFilter {
            base: BaseFilter::new(radius),
        }
    }

    pub fn create(params: &ParameterDictionary) -> Result<BoxFilter, PbrtError> {
        let xw = params.get_one_float("xradius", 0.5);
        let yw = params.get_one_float("yradius", 0.5);
        Ok(BoxFilter::new(&Vector2f::new(xw, yw)))
    }

    pub fn evaluate(&self, _p: &Point2f) -> Float {
        1.0
    }

    pub fn integral(&self) -> Float {
        4.0 * self.base.radius.x * self.base.radius.y
    }

    pub fn sample(&self, u: &Point2f) -> FilterSample {
        let p = Point2f::new(
            lerp(u.x, -self.base.radius.x, self.base.radius.x),
            lerp(u.y, -self.base.radius.y, self.base.radius.y),
        );
        FilterSample { p, weight: 1.0 }
    }
}

impl Clone for BoxFilter {
    fn clone(&self) -> Self {
        BoxFilter { base: self.base }
    }
}
