// Medium trait - base interface for all media
// Moved from core::medium to match pbrt-v4 structure

use std::fmt::Debug;
use std::sync::Arc;

use crate::media::majorant_iterator::{
    DDAMajorantIterator, HomogeneousMajorantIterator, RayMajorantSegment,
};
use crate::media::{
    CloudMedium, GridMedium, HGPhaseFunction, HomogeneousMedium, NanoVDBMedium, PhaseFunction,
    RGBGridMedium,
};
use crate::samplers::Sampler;
use crate::util::base::Float;
use crate::util::base::{Point3f, FLOAT_ONE_MINUS_EPSILON};
use crate::util::geometry::ray::Ray;
use crate::util::rng::RNG;
use crate::util::spectrum::*;

/// Mirrors pbrt-v4 `MediumProperties` (`media.h:73`).
/// `sigma_a`, `sigma_s`, `le` are `SampledSpectrum` (4 floats on the
/// stack) so each event in `sample_t_maj` doesn't pay an
/// `Arc<[Float; 471]>` allocation per `Spectrum::Mul<Float>`.
#[derive(Clone)]
pub struct MediumProperties {
    pub sigma_a: SampledSpectrum,
    pub sigma_s: SampledSpectrum,
    pub phase: Arc<PhaseFunction>,
    pub le: SampledSpectrum,
}

#[derive(Clone, Copy, Debug)]
pub struct MediumCoefficients {
    pub sigma_a: SampledSpectrum,
    pub sigma_s: SampledSpectrum,
    pub le: SampledSpectrum,
}

#[derive(Clone, Copy, Debug)]
pub struct MediumSigma {
    pub sigma_a: SampledSpectrum,
    pub sigma_s: SampledSpectrum,
}

impl MediumSigma {
    pub fn new(sigma_a: SampledSpectrum, sigma_s: SampledSpectrum) -> Self {
        Self { sigma_a, sigma_s }
    }
}

impl MediumCoefficients {
    pub fn new(sigma_a: SampledSpectrum, sigma_s: SampledSpectrum, le: SampledSpectrum) -> Self {
        Self {
            sigma_a,
            sigma_s,
            le,
        }
    }
}

impl MediumProperties {
    pub fn new(
        sigma_a: SampledSpectrum,
        sigma_s: SampledSpectrum,
        phase: Arc<PhaseFunction>,
        le: SampledSpectrum,
    ) -> Self {
        Self {
            sigma_a,
            sigma_s,
            phase,
            le,
        }
    }

    pub fn vacuum() -> Self {
        Self {
            sigma_a: SampledSpectrum::zero(),
            sigma_s: SampledSpectrum::zero(),
            phase: Arc::new(PhaseFunction::from(HGPhaseFunction::new(0.0))),
            le: SampledSpectrum::zero(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum RayMajorantIterator {
    Empty,
    Homogeneous(HomogeneousMajorantIterator),
    Dda(DDAMajorantIterator),
}

impl RayMajorantIterator {
    pub fn next(&mut self) -> Option<RayMajorantSegment> {
        match self {
            Self::Empty => None,
            Self::Homogeneous(iter) => iter.next(),
            Self::Dda(iter) => iter.next(),
        }
    }
}

pub enum Medium {
    Homogeneous(Box<HomogeneousMedium>),
    Grid(Box<GridMedium>),
    RGBGrid(Box<RGBGridMedium>),
    Cloud(Box<CloudMedium>),
    NanoVDB(Box<NanoVDBMedium>),
}

impl std::fmt::Debug for Medium {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Homogeneous(m) => m.fmt(f),
            Self::Grid(m) => m.fmt(f),
            Self::RGBGrid(m) => m.fmt(f),
            Self::Cloud(m) => m.fmt(f),
            Self::NanoVDB(m) => m.fmt(f),
        }
    }
}

impl From<HomogeneousMedium> for Medium {
    fn from(medium: HomogeneousMedium) -> Self {
        Self::Homogeneous(Box::new(medium))
    }
}

impl From<GridMedium> for Medium {
    fn from(medium: GridMedium) -> Self {
        Self::Grid(Box::new(medium))
    }
}

impl From<RGBGridMedium> for Medium {
    fn from(medium: RGBGridMedium) -> Self {
        Self::RGBGrid(Box::new(medium))
    }
}

impl From<CloudMedium> for Medium {
    fn from(medium: CloudMedium) -> Self {
        Self::Cloud(Box::new(medium))
    }
}

impl From<NanoVDBMedium> for Medium {
    fn from(medium: NanoVDBMedium) -> Self {
        Self::NanoVDB(Box::new(medium))
    }
}

impl Medium {
    pub fn is_emissive(&self) -> bool {
        match self {
            Self::Homogeneous(m) => m.is_emissive(),
            Self::Grid(m) => m.is_emissive(),
            Self::RGBGrid(m) => m.is_emissive(),
            Self::Cloud(_) => false,
            Self::NanoVDB(m) => m.is_emissive(),
        }
    }

    pub fn sample_point(&self, p: &Point3f, lambda: &SampledWavelengths) -> MediumProperties {
        match self {
            Self::Homogeneous(m) => m.sample_point(p, lambda),
            Self::Grid(m) => m.sample_point(p, lambda),
            Self::RGBGrid(m) => m.sample_point(p, lambda),
            Self::Cloud(m) => m.sample_point(p, lambda),
            Self::NanoVDB(m) => m.sample_point(p, lambda),
        }
    }

    pub fn sample_point_coefficients(
        &self,
        p: &Point3f,
        lambda: &SampledWavelengths,
    ) -> MediumCoefficients {
        match self {
            Self::Homogeneous(m) => m.sample_point_coefficients(p, lambda),
            Self::Grid(m) => m.sample_point_coefficients(p, lambda),
            Self::RGBGrid(m) => m.sample_point_coefficients(p, lambda),
            Self::Cloud(m) => m.sample_point_coefficients(p, lambda),
            Self::NanoVDB(m) => m.sample_point_coefficients(p, lambda),
        }
    }

    pub fn sample_point_sigma(&self, p: &Point3f, lambda: &SampledWavelengths) -> MediumSigma {
        match self {
            Self::Homogeneous(m) => m.sample_point_sigma(p, lambda),
            Self::Grid(m) => m.sample_point_sigma(p, lambda),
            Self::RGBGrid(m) => m.sample_point_sigma(p, lambda),
            Self::Cloud(m) => m.sample_point_sigma(p, lambda),
            Self::NanoVDB(m) => m.sample_point_sigma(p, lambda),
        }
    }

    pub fn sample_phase_function(
        &self,
        p: &Point3f,
        lambda: &SampledWavelengths,
    ) -> Arc<PhaseFunction> {
        self.sample_point(p, lambda).phase
    }

    pub fn sample_ray(
        &self,
        ray: &Ray,
        t_max: Float,
        lambda: &SampledWavelengths,
    ) -> RayMajorantIterator {
        match self {
            Self::Homogeneous(m) => m
                .sample_ray(ray, t_max, lambda)
                .map(RayMajorantIterator::Homogeneous)
                .unwrap_or(RayMajorantIterator::Empty),
            Self::Grid(m) => m
                .sample_ray(ray, t_max, lambda)
                .map(RayMajorantIterator::Dda)
                .unwrap_or(RayMajorantIterator::Empty),
            Self::RGBGrid(m) => m
                .sample_ray(ray, t_max, lambda)
                .map(RayMajorantIterator::Dda)
                .unwrap_or(RayMajorantIterator::Empty),
            Self::Cloud(m) => m
                .sample_ray(ray, t_max, lambda)
                .map(RayMajorantIterator::Homogeneous)
                .unwrap_or(RayMajorantIterator::Empty),
            Self::NanoVDB(m) => m
                .sample_ray(ray, t_max, lambda)
                .map(RayMajorantIterator::Dda)
                .unwrap_or(RayMajorantIterator::Empty),
        }
    }
}

pub fn rng_from_sampler(sampler: &mut Sampler) -> RNG {
    let sequence = (sampler
        .get_1d()
        .clamp(0.0, FLOAT_ONE_MINUS_EPSILON as Float)
        * (u32::MAX as Float)) as u64;
    let seed = (sampler
        .get_1d()
        .clamp(0.0, FLOAT_ONE_MINUS_EPSILON as Float)
        * (u32::MAX as Float)) as u64;
    let mut rng = RNG::new();
    rng.set_sequence_with_seed(sequence, seed ^ 0x9e37_79b9_7f4a_7c15);
    rng
}

#[inline(always)]
fn sample_t_maj_impl<T, S, F>(
    medium: &Medium,
    ray: &Ray,
    t_max: Float,
    mut u: Float,
    lambda: &SampledWavelengths,
    rng: &mut RNG,
    mut sample_point: S,
    mut callback: F,
) -> SampledSpectrum
where
    S: FnMut(&Medium, &Point3f, &SampledWavelengths) -> T,
    F: FnMut(Point3f, T, SampledSpectrum, SampledSpectrum, &mut RNG) -> bool,
{
    let ray_length = ray.d.length();
    if ray_length == 0.0 {
        return SampledSpectrum::one();
    }

    let mut local_ray = ray.clone();
    local_ray.d = local_ray.d / ray_length;
    let mut iter = medium.sample_ray(&local_ray, t_max * ray_length, lambda);
    let mut t_maj = SampledSpectrum::one();
    let mut done = false;

    while !done {
        let Some(seg) = iter.next() else {
            return t_maj;
        };

        if seg.sigma_maj[0] == 0.0 {
            let mut dt = seg.t_max - seg.t_min;
            if dt.is_infinite() {
                dt = Float::MAX;
            }
            t_maj *= (-(seg.sigma_maj) * dt).exp();
            continue;
        }

        let mut t_min = seg.t_min;
        loop {
            let t = t_min - Float::ln(1.0 - u) / seg.sigma_maj[0];
            u = rng.uniform_float();
            if t < seg.t_max {
                t_maj *= (-(t - t_min) * seg.sigma_maj).exp();
                let p = local_ray.position(t);
                let sampled = sample_point(medium, &p, lambda);
                if !callback(p, sampled, seg.sigma_maj, t_maj, rng) {
                    done = true;
                    break;
                }
                t_maj = SampledSpectrum::one();
                t_min = t;
            } else {
                let mut dt = seg.t_max - t_min;
                if dt.is_infinite() {
                    dt = Float::MAX;
                }
                t_maj *= (-(seg.sigma_maj) * dt).exp();
                break;
            }
        }
    }

    SampledSpectrum::one()
}

/// pbrt-v4 `SampleT_maj(ray, tMax, u, rng, lambda, callback)`
/// (`media.h:735-808`). `sigma_maj` and `T_maj` are both packed
/// SampledSpectrum values throughout — no `Spectrum`-enum heap
/// allocations on the hot path.
pub fn sample_t_maj<F>(
    medium: &Medium,
    ray: &Ray,
    t_max: Float,
    u: Float,
    lambda: &SampledWavelengths,
    rng: &mut RNG,
    callback: F,
) -> SampledSpectrum
where
    F: FnMut(Point3f, MediumProperties, SampledSpectrum, SampledSpectrum, &mut RNG) -> bool,
{
    sample_t_maj_impl(
        medium,
        ray,
        t_max,
        u,
        lambda,
        rng,
        |medium, p, lambda| medium.sample_point(p, lambda),
        callback,
    )
}

pub fn sample_t_maj_coefficients<F>(
    medium: &Medium,
    ray: &Ray,
    t_max: Float,
    u: Float,
    lambda: &SampledWavelengths,
    rng: &mut RNG,
    callback: F,
) -> SampledSpectrum
where
    F: FnMut(Point3f, MediumCoefficients, SampledSpectrum, SampledSpectrum, &mut RNG) -> bool,
{
    if let Medium::NanoVDB(medium) = medium {
        return medium.sample_t_maj_coefficients(ray, t_max, u, lambda, rng, callback);
    }

    sample_t_maj_impl(
        medium,
        ray,
        t_max,
        u,
        lambda,
        rng,
        |medium, p, lambda| medium.sample_point_coefficients(p, lambda),
        callback,
    )
}

pub fn sample_t_maj_sigma<F>(
    medium: &Medium,
    ray: &Ray,
    t_max: Float,
    u: Float,
    lambda: &SampledWavelengths,
    rng: &mut RNG,
    callback: F,
) -> SampledSpectrum
where
    F: FnMut(Point3f, MediumSigma, SampledSpectrum, SampledSpectrum, &mut RNG) -> bool,
{
    if let Medium::NanoVDB(medium) = medium {
        return medium.sample_t_maj_sigma(ray, t_max, u, lambda, rng, callback);
    }

    sample_t_maj_impl(
        medium,
        ray,
        t_max,
        u,
        lambda,
        rng,
        |medium, p, lambda| medium.sample_point_sigma(p, lambda),
        callback,
    )
}
