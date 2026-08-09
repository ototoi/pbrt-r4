use crate::filters::*;
use crate::paramdict::*;

use crate::util::base::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

pub struct MitchellFilter {
    pub base: BaseFilter,
    pub b: Float,
    pub c: Float,
    pub sampler: FilterSampler,
}

impl MitchellFilter {
    pub fn new(radius: &Vector2f, b: Float, c: Float) -> Self {
        let base = BaseFilter::new(radius);
        let eval = MitchellSampledFilter {
            radius: base.radius,
            b,
            c,
        };
        MitchellFilter {
            base,
            b,
            c,
            sampler: FilterSampler::new(&eval),
        }
    }

    pub fn create(params: &ParameterDictionary) -> Result<MitchellFilter, PbrtError> {
        let xw = params.get_one_float("xradius", 2.0);
        let yw = params.get_one_float("yradius", 2.0);
        let b = params.get_one_float("B", 1.0 / 3.0);
        let c = params.get_one_float("C", 1.0 / 3.0);
        Ok(MitchellFilter::new(&Vector2f::new(xw, yw), b, c))
    }

    fn mitchell_1d(&self, x: Float) -> Float {
        let b = self.b;
        let c = self.c;
        let x = Float::abs(x);
        if x <= 1.0 {
            ((12.0 - 9.0 * b - 6.0 * c) * x * x * x
                + (-18.0 + 12.0 * b + 6.0 * c) * x * x
                + (6.0 - 2.0 * b))
                * (1.0 / 6.0)
        } else if x <= 2.0 {
            ((-b - 6.0 * c) * x * x * x
                + (6.0 * b + 30.0 * c) * x * x
                + (-12.0 * b - 48.0 * c) * x
                + (8.0 * b + 24.0 * c))
                * (1.0 / 6.0)
        } else {
            0.0
        }
    }

    pub fn evaluate(&self, p: &Point2f) -> Float {
        self.mitchell_1d(2.0 * p.x * self.base.inv_radius.x)
            * self.mitchell_1d(2.0 * p.y * self.base.inv_radius.y)
    }

    pub fn integral(&self) -> Float {
        (self.base.radius.x * self.base.radius.y) * 0.25
    }

    pub fn sample(&self, u: &Point2f) -> FilterSample {
        self.sampler.sample(u)
    }
}

struct MitchellSampledFilter {
    radius: Vector2f,
    b: Float,
    c: Float,
}

impl MitchellSampledFilter {
    fn mitchell_1d(&self, x: Float) -> Float {
        let b = self.b;
        let c = self.c;
        let x = Float::abs(x);
        if x > 1.0 {
            ((-b - 6.0 * c) * x * x * x
                + (6.0 * b + 30.0 * c) * x * x
                + (-12.0 * b - 48.0 * c) * x
                + (8.0 * b + 24.0 * c))
                * (1.0 / 6.0)
        } else {
            ((12.0 - 9.0 * b - 6.0 * c) * x * x * x
                + (-18.0 + 12.0 * b + 6.0 * c) * x * x
                + (6.0 - 2.0 * b))
                * (1.0 / 6.0)
        }
    }
}

impl SampledFilter for MitchellSampledFilter {
    fn radius(&self) -> Vector2f {
        self.radius
    }

    fn evaluate(&self, p: &Point2f) -> Float {
        self.mitchell_1d(2.0 * p.x / self.radius.x) * self.mitchell_1d(2.0 * p.y / self.radius.y)
    }
}

impl SampledFilter for MitchellFilter {
    fn radius(&self) -> Vector2f {
        self.base.radius
    }

    fn evaluate(&self, p: &Point2f) -> Float {
        MitchellFilter::evaluate(self, p)
    }
}

impl Clone for MitchellFilter {
    fn clone(&self) -> Self {
        MitchellFilter {
            base: self.base,
            b: self.b,
            c: self.c,
            sampler: self.sampler.clone(),
        }
    }
}
