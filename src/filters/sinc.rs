use crate::filters::*;
use crate::paramdict::*;

use crate::util::base::*;
use crate::util::error::*;
use crate::util::math::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

pub struct LanczosSincFilter {
    pub base: BaseFilter,
    pub tau: Float,
    pub sampler: FilterSampler,
}

impl LanczosSincFilter {
    pub fn new(radius: &Vector2f, tau: Float) -> Self {
        let base = BaseFilter::new(radius);
        let eval = LanczosSincSampledFilter {
            radius: base.radius,
            tau,
        };
        LanczosSincFilter {
            base,
            tau,
            sampler: FilterSampler::new(&eval),
        }
    }

    pub fn create(params: &ParameterDictionary) -> Result<LanczosSincFilter, PbrtError> {
        let xw = params.get_one_float("xradius", 4.0);
        let yw = params.get_one_float("yradius", 4.0);
        let tau = params.get_one_float("tau", 3.0);
        Ok(LanczosSincFilter::new(&Vector2f::new(xw, yw), tau))
    }

    pub fn evaluate(&self, p: &Point2f) -> Float {
        windowed_sinc(p.x, self.base.radius.x, self.tau)
            * windowed_sinc(p.y, self.base.radius.y, self.tau)
    }

    pub fn integral(&self) -> Float {
        sinc_integral_2d(self.base.radius, self.tau)
    }

    pub fn sample(&self, u: &Point2f) -> FilterSample {
        self.sampler.sample(u)
    }
}

struct LanczosSincSampledFilter {
    radius: Vector2f,
    tau: Float,
}

impl SampledFilter for LanczosSincSampledFilter {
    fn radius(&self) -> Vector2f {
        self.radius
    }

    fn evaluate(&self, p: &Point2f) -> Float {
        windowed_sinc(p.x, self.radius.x, self.tau) * windowed_sinc(p.y, self.radius.y, self.tau)
    }
}

impl SampledFilter for LanczosSincFilter {
    fn radius(&self) -> Vector2f {
        self.base.radius
    }

    fn evaluate(&self, p: &Point2f) -> Float {
        LanczosSincFilter::evaluate(self, p)
    }
}

fn integrate_1d_simpson<F: Fn(Float) -> Float>(f: F, min: Float, max: Float, n: usize) -> Float {
    let n = if n % 2 == 0 { n } else { n + 1 };
    let h = (max - min) / n as Float;
    let mut sum = f(min) + f(max);
    for i in 1..n {
        let x = min + (i as Float) * h;
        sum += if i % 2 == 0 { 2.0 } else { 4.0 } * f(x);
    }
    (h / 3.0) * sum
}

fn sinc_integral_1d(radius: Float, tau: Float) -> Float {
    integrate_1d_simpson(|x| windowed_sinc(x, radius, tau), -radius, radius, 1024)
}

fn sinc_integral_2d(radius: Vector2f, tau: Float) -> Float {
    sinc_integral_1d(radius.x, tau) * sinc_integral_1d(radius.y, tau)
}

impl Clone for LanczosSincFilter {
    fn clone(&self) -> Self {
        LanczosSincFilter {
            base: self.base,
            tau: self.tau,
            sampler: self.sampler.clone(),
        }
    }
}
