use crate::filters::*;
use crate::paramdict::*;

use crate::util::base::*;
use crate::util::error::*;
use crate::util::math::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

pub struct GaussianFilter {
    pub base: BaseFilter,
    pub sigma: Float,
    pub exp_x: Float,
    pub exp_y: Float,
    pub sampler: FilterSampler,
}

impl GaussianFilter {
    pub fn new(radius: &Vector2f, sigma: Float) -> Self {
        let base = BaseFilter::new(radius);
        let exp_x = gaussian(radius.x, 0.0, sigma);
        let exp_y = gaussian(radius.y, 0.0, sigma);
        let eval = GaussianSampledFilter {
            radius: base.radius,
            sigma,
            exp_x,
            exp_y,
        };
        GaussianFilter {
            base,
            sigma,
            exp_x,
            exp_y,
            sampler: FilterSampler::new(&eval),
        }
    }

    pub fn create(params: &ParameterDictionary) -> Result<GaussianFilter, PbrtError> {
        let xw = params.get_one_float("xradius", 1.5);
        let yw = params.get_one_float("yradius", 1.5);
        let sigma = params.get_one_float("sigma", 0.5);
        Ok(GaussianFilter::new(&Vector2f::new(xw, yw), sigma))
    }

    pub fn evaluate(&self, p: &Point2f) -> Float {
        Float::max(0.0, gaussian(p.x, 0.0, self.sigma) - self.exp_x)
            * Float::max(0.0, gaussian(p.y, 0.0, self.sigma) - self.exp_y)
    }

    pub fn integral(&self) -> Float {
        let gx = gaussian_integral(-self.base.radius.x, self.base.radius.x, 0.0, self.sigma)
            - 2.0 * self.base.radius.x * self.exp_x;
        let gy = gaussian_integral(-self.base.radius.y, self.base.radius.y, 0.0, self.sigma)
            - 2.0 * self.base.radius.y * self.exp_y;
        gx * gy
    }

    pub fn sample(&self, u: &Point2f) -> FilterSample {
        self.sampler.sample(u)
    }
}

struct GaussianSampledFilter {
    radius: Vector2f,
    sigma: Float,
    exp_x: Float,
    exp_y: Float,
}

impl SampledFilter for GaussianSampledFilter {
    fn radius(&self) -> Vector2f {
        self.radius
    }

    fn evaluate(&self, p: &Point2f) -> Float {
        Float::max(0.0, gaussian(p.x, 0.0, self.sigma) - self.exp_x)
            * Float::max(0.0, gaussian(p.y, 0.0, self.sigma) - self.exp_y)
    }
}

impl SampledFilter for GaussianFilter {
    fn radius(&self) -> Vector2f {
        self.base.radius
    }

    fn evaluate(&self, p: &Point2f) -> Float {
        GaussianFilter::evaluate(self, p)
    }
}

impl Clone for GaussianFilter {
    fn clone(&self) -> Self {
        GaussianFilter {
            base: self.base,
            sigma: self.sigma,
            exp_x: self.exp_x,
            exp_y: self.exp_y,
            sampler: self.sampler.clone(),
        }
    }
}
