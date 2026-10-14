use crate::base::filter::Filter as CpuFilter;
use crate::paramdict::ParameterDictionary;
use crate::util::error::PbrtError;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterKind {
    Box = 0,
    Triangle = 1,
    Gaussian = 2,
    Mitchell = 3,
    Sinc = 4,
}

/// v4 FilterSampler's signed 2D table and absolute-value sampling distribution.
/// CDFs and integrals use the unit square, as in the shared CPU FilterSampler.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterDistribution {
    pub dimensions: [u32; 2],
    pub values: Vec<f32>,
    pub conditional_cdf: Vec<f32>,
    pub marginal_cdf: Vec<f32>,
    pub integral: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Filter {
    pub kind: FilterKind,
    pub radius: [f32; 2],
    pub distribution: Option<FilterDistribution>,
}

impl Filter {
    pub fn create(name: &str, params: &ParameterDictionary) -> Result<Self, PbrtError> {
        let (kind, default_radius) = match name {
            "box" => (FilterKind::Box, 0.5),
            "triangle" => (FilterKind::Triangle, 2.0),
            "gaussian" => (FilterKind::Gaussian, 1.5),
            "mitchell" => (FilterKind::Mitchell, 2.0),
            "sinc" => (FilterKind::Sinc, 4.0),
            _ => return Err(PbrtError::error(&format!("Filter \"{name}\" unknown."))),
        };
        let radius = [
            params.get_one_float("xradius", default_radius),
            params.get_one_float("yradius", default_radius),
        ];
        if radius.iter().any(|r| !r.is_finite() || *r <= 0.0)
            || !(4.0 * radius[0] * radius[1]).is_finite()
        {
            return Err(PbrtError::error(
                "GPU filter radii must be positive and finite.",
            ));
        }
        let tabulated = !matches!(kind, FilterKind::Box | FilterKind::Triangle);
        if tabulated {
            let dims = radius.map(|r| (32.0 * r).trunc());
            if dims
                .iter()
                .any(|n| !n.is_finite() || *n < 1.0 || *n >= u32::MAX as f32)
            {
                return Err(PbrtError::error("GPU filter table dimensions are invalid."));
            }
            // Check every array length and the packed byte count before CPU allocation.
            let [nx, ny] = dims.map(|n| n as u64);
            let words = nx
                .checked_mul(ny)
                .and_then(|n| (nx + 1).checked_mul(ny).and_then(|cdf| n.checked_add(cdf)))
                .and_then(|n| n.checked_add(ny + 1 + 12));
            if words
                .and_then(|n| n.checked_mul(4))
                .filter(|n| *n <= u32::MAX as u64)
                .is_none()
            {
                return Err(PbrtError::error(
                    "GPU filter table exceeds the addressable buffer size.",
                ));
            }
        }
        if kind == FilterKind::Gaussian {
            let sigma = params.get_one_float("sigma", 0.5);
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err(PbrtError::error(
                    "GPU Gaussian filter sigma must be positive and finite.",
                ));
            }
        }
        let cpu = CpuFilter::create(name, params)?;
        let sampler = match &cpu {
            CpuFilter::Gaussian(f) => Some(&f.sampler),
            CpuFilter::Mitchell(f) => Some(&f.sampler),
            CpuFilter::LanczosSinc(f) => Some(&f.sampler),
            _ => None,
        };
        let distribution = if let Some(sampler) = sampler {
            let dist = sampler.distribution();
            let values = sampler.values();
            let absolute_integral = (4.0 * radius[0] * radius[1]) * dist.marginal.func_int;
            if values.iter().any(|v| !v.is_finite())
                || !dist.marginal.func_int.is_finite()
                || dist.marginal.func_int <= 0.0
                || dist
                    .conditional_v
                    .iter()
                    .any(|row| !row.func_int.is_finite())
                // WGSL may flush subnormal weights to zero.
                || !absolute_integral.is_normal()
            {
                return Err(PbrtError::error("GPU filter table must have finite values and a positive normal absolute integral."));
            }
            Some(FilterDistribution {
                dimensions: sampler.dimensions().map(|n| n as u32),
                values: values.to_vec(),
                conditional_cdf: dist
                    .conditional_v
                    .iter()
                    .flat_map(|row| row.cdf.iter().copied())
                    .collect(),
                marginal_cdf: dist.marginal.cdf.clone(),
                integral: dist.marginal.func_int,
            })
        } else {
            None
        };
        Ok(Self {
            kind,
            radius,
            distribution,
        })
    }
}
