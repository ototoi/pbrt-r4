// Filter enum - base interface for all filters
// Moved from core::filter to match pbrt-v4 structure

use crate::filters::*;
use crate::paramdict::*;

use crate::util::base::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

// Re-export filter implementations from the filters module
pub use crate::filters::{
    BoxFilter, GaussianFilter, LanczosSincFilter, MitchellFilter, TriangleFilter,
};

#[derive(Clone)]
pub enum Filter {
    Box(BoxFilter),
    Gaussian(GaussianFilter),
    Mitchell(MitchellFilter),
    LanczosSinc(LanczosSincFilter),
    Triangle(TriangleFilter),
}

impl Filter {
    /// Create a Filter from a name and ParameterDictionary (ParameterDictionary in Rust)
    /// Matches pbrt-v4's Filter::Create API
    pub fn create(name: &str, params: &ParameterDictionary) -> Result<Filter, PbrtError> {
        let filter = match name {
            "box" => Filter::Box(BoxFilter::create(params)?),
            "gaussian" => Filter::Gaussian(GaussianFilter::create(params)?),
            "mitchell" => Filter::Mitchell(MitchellFilter::create(params)?),
            "sinc" => Filter::LanczosSinc(LanczosSincFilter::create(params)?),
            "triangle" => Filter::Triangle(TriangleFilter::create(params)?),
            _ => {
                let msg = format!("Filter \"{}\" unknown.", name);
                return Err(PbrtError::error(&msg));
            }
        };

        Ok(filter)
    }

    pub fn evaluate(&self, p: &Point2f) -> Float {
        match self {
            Filter::Box(f) => f.evaluate(p),
            Filter::Gaussian(f) => f.evaluate(p),
            Filter::Mitchell(f) => f.evaluate(p),
            Filter::LanczosSinc(f) => f.evaluate(p),
            Filter::Triangle(f) => f.evaluate(p),
        }
    }

    pub fn radius(&self) -> Vector2f {
        match self {
            Filter::Box(f) => f.base.radius,
            Filter::Gaussian(f) => f.base.radius,
            Filter::Mitchell(f) => f.base.radius,
            Filter::LanczosSinc(f) => f.base.radius,
            Filter::Triangle(f) => f.base.radius,
        }
    }

    pub fn integral(&self) -> Float {
        match self {
            Filter::Box(f) => f.integral(),
            Filter::Gaussian(f) => f.integral(),
            Filter::Mitchell(f) => f.integral(),
            Filter::LanczosSinc(f) => f.integral(),
            Filter::Triangle(f) => f.integral(),
        }
    }

    pub fn sample(&self, u: &Point2f) -> FilterSample {
        match self {
            Filter::Box(f) => f.sample(u),
            Filter::Gaussian(f) => f.sample(u),
            Filter::Mitchell(f) => f.sample(u),
            Filter::LanczosSinc(f) => f.sample(u),
            Filter::Triangle(f) => f.sample(u),
        }
    }
}
