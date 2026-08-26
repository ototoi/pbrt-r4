use crate::base::lightsampler::LightSampleContext;
use crate::base::shape::Shape;
use crate::interaction::*;
use crate::lights::*;
use crate::media::*;
use crate::paramdict::ParameterDictionary;
use crate::util::base::*;
use crate::util::error::PbrtError;
use crate::util::geometry::*;
use crate::util::spectrum::*;
use crate::util::transform::*;
use std::sync::Arc;

pub use crate::lights::base::LightBase;
pub use crate::lights::VisibilityTester;

/// pbrt-v4 `enum class LightType { DeltaPosition, DeltaDirection, Area,
/// Infinite }`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum LightType {
    DeltaPosition = 1,
    DeltaDirection = 2,
    Area = 4,
    Infinite = 8,
}

/// pbrt-v4 `inline bool IsDeltaLight(LightType type)` in lights.h.
pub fn is_delta_light(t: LightType) -> bool {
    matches!(t, LightType::DeltaPosition | LightType::DeltaDirection)
}

#[derive(Clone)]
pub struct LightLiSample {
    pub l: SampledSpectrum,
    pub wi: Vector3f,
    pub pdf: Float,
    pub p_light: Interaction,
}

impl LightLiSample {
    pub fn new(l: SampledSpectrum, wi: Vector3f, pdf: Float, p_light: Interaction) -> Self {
        Self {
            l,
            wi,
            pdf,
            p_light,
        }
    }
}

#[derive(Clone)]
pub struct LightLeSample {
    pub l: SampledSpectrum,
    pub ray: Ray,
    pub intr: Option<Interaction>,
    pub pdf_pos: Float,
    pub pdf_dir: Float,
}

impl LightLeSample {
    pub fn new(l: SampledSpectrum, ray: Ray, pdf_pos: Float, pdf_dir: Float) -> Self {
        Self {
            l,
            ray,
            intr: None,
            pdf_pos,
            pdf_dir,
        }
    }
    pub fn with_intr(
        l: SampledSpectrum,
        ray: Ray,
        intr: Interaction,
        pdf_pos: Float,
        pdf_dir: Float,
    ) -> Self {
        Self {
            l,
            ray,
            intr: Some(intr),
            pdf_pos,
            pdf_dir,
        }
    }

    pub fn abs_cos_theta(&self, w: Vector3f) -> Float {
        match &self.intr {
            Some(it) => Float::abs(Vector3f::dot(&w, &it.get_n())),
            None => 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LightBounds {
    pub bounds: Bounds3f,
    pub w: Vector3f,
    pub phi: Float,
    pub cos_theta_o: Float,
    pub cos_theta_e: Float,
    pub two_sided: bool,
}

impl LightBounds {
    pub fn new(
        bounds: Bounds3f,
        w: Vector3f,
        phi: Float,
        cos_theta_o: Float,
        cos_theta_e: Float,
        two_sided: bool,
    ) -> Self {
        Self {
            bounds,
            w: w.normalize(),
            phi,
            cos_theta_o,
            cos_theta_e,
            two_sided,
        }
    }

    pub fn centroid(&self) -> Point3f {
        (self.bounds.min + self.bounds.max) * 0.5
    }

    pub fn importance(&self, p: Point3f, n: Normal3f) -> Float {
        // Return importance for light bounds at reference point
        // Compute clamped squared distance to reference point
        let pc = (self.bounds.min + self.bounds.max) / 2.0;
        let mut d2 = Point3f::distance_squared(&p, &pc);
        d2 = Float::max(d2, self.bounds.diagonal().length() / 2.0);

        // Define cosine and sine clamped subtraction lambdas
        let cos_sub_clamped =
            |sin_theta_a: Float, cos_theta_a: Float, sin_theta_b: Float, cos_theta_b: Float| {
                if cos_theta_a > cos_theta_b {
                    1.0
                } else {
                    cos_theta_a * cos_theta_b + sin_theta_a * sin_theta_b
                }
            };
        let sin_sub_clamped =
            |sin_theta_a: Float, cos_theta_a: Float, sin_theta_b: Float, cos_theta_b: Float| {
                if cos_theta_a > cos_theta_b {
                    0.0
                } else {
                    sin_theta_a * cos_theta_b - cos_theta_a * sin_theta_b
                }
            };

        // Compute sine and cosine of angle to vector w, theta_w
        let wi = (p - pc).normalize();
        let mut cos_theta_w = Vector3f::dot(&self.w, &wi);
        if self.two_sided {
            cos_theta_w = Float::abs(cos_theta_w);
        }
        let sin_theta_w = Float::sqrt(Float::max(0.0, 1.0 - cos_theta_w * cos_theta_w));

        // Compute cos theta_b for reference point
        let cos_theta_b = bound_subtended_directions(&self.bounds, &p).cos_theta;
        let sin_theta_b = Float::sqrt(Float::max(0.0, 1.0 - cos_theta_b * cos_theta_b));

        // Compute cos theta' and test against cos theta_e
        let sin_theta_o = Float::sqrt(Float::max(0.0, 1.0 - self.cos_theta_o * self.cos_theta_o));
        let cos_theta_x = cos_sub_clamped(sin_theta_w, cos_theta_w, sin_theta_o, self.cos_theta_o);
        let sin_theta_x = sin_sub_clamped(sin_theta_w, cos_theta_w, sin_theta_o, self.cos_theta_o);
        let cos_theta_p = cos_sub_clamped(sin_theta_x, cos_theta_x, sin_theta_b, cos_theta_b);
        if cos_theta_p <= self.cos_theta_e {
            return 0.0;
        }

        // Return final importance at reference point
        let mut importance = self.phi * cos_theta_p / d2;
        // Account for cos theta_i in importance at surfaces
        if n != Normal3f::zero() {
            let cos_theta_i = Vector3f::abs_dot(&wi, &n);
            let sin_theta_i = Float::sqrt(Float::max(0.0, 1.0 - cos_theta_i * cos_theta_i));
            let cos_theta_p_i = cos_sub_clamped(sin_theta_i, cos_theta_i, sin_theta_b, cos_theta_b);
            importance *= cos_theta_p_i;
        }

        Float::max(importance, 0.0)
    }
}

pub fn union_light_bounds(a: &LightBounds, b: &LightBounds) -> LightBounds {
    // If one LightBounds has zero power, return the other
    if a.phi == 0.0 {
        return b.clone();
    }
    if b.phi == 0.0 {
        return a.clone();
    }

    // Find average direction and updated angles for LightBounds
    let cone = union_direction_cones(
        &DirectionCone::new(a.w, a.cos_theta_o),
        &DirectionCone::new(b.w, b.cos_theta_o),
    );
    // Return final LightBounds union
    LightBounds::new(
        a.bounds.union(&b.bounds),
        cone.w,
        a.phi + b.phi,
        cone.cos_theta,
        Float::min(a.cos_theta_e, b.cos_theta_e),
        a.two_sided | b.two_sided,
    )
}

pub enum Light {
    Point(Box<PointLight>),
    Distant(Box<DistantLight>),
    Projection(Box<ProjectionLight>),
    Goniometric(Box<GoniometricLight>),
    Spot(Box<SpotLight>),
    DiffuseArea(Box<DiffuseAreaLight>),
    Infinite(Box<InfiniteLight>),
}

impl Light {
    pub fn create(
        name: &str,
        light2world: &Transform,
        medium_interface: &MediumInterface,
        params: &ParameterDictionary,
        render_from_world: &Transform,
    ) -> Result<Arc<Light>, PbrtError> {
        let medium = medium_interface.get_outside();
        match name {
            "point" => PointLight::create(light2world, &medium, params)
                .map(|l| Arc::new(Light::Point(Box::new(l)))),
            "spot" => SpotLight::create(light2world, &medium, params)
                .map(|l| Arc::new(Light::Spot(Box::new(l)))),
            "goniometric" => GoniometricLight::create(light2world, &medium, params)
                .map(|l| Arc::new(Light::Goniometric(Box::new(l)))),
            "projection" => ProjectionLight::create(light2world, medium_interface, params)
                .map(|l| Arc::new(Light::Projection(Box::new(l)))),
            "distant" => DistantLight::create(light2world, params)
                .map(|l| Arc::new(Light::Distant(Box::new(l)))),
            "infinite" => InfiniteLight::create(light2world, params, render_from_world)
                .map(|l| Arc::new(Light::Infinite(Box::new(l)))),
            _ => Err(PbrtError::error(&format!("Light \"{}\" unknown.", name))),
        }
    }

    pub fn create_area(
        name: &str,
        light2world: &Transform,
        medium_interface: &MediumInterface,
        params: &ParameterDictionary,
        shape: &Arc<Shape>,
    ) -> Result<Arc<Light>, PbrtError> {
        match name {
            "area" | "diffuse" => {
                let medium = medium_interface.get_outside();
                DiffuseAreaLight::create(light2world, &medium, params, shape)
                    .map(|l| Arc::new(Light::DiffuseArea(Box::new(l))))
            }
            _ => Err(PbrtError::error(&format!(
                "Area light \"{}\" unknown.",
                name
            ))),
        }
    }

    pub fn light_type(&self) -> LightType {
        match self {
            Light::Point(l) => l.light_type(),
            Light::Distant(l) => l.light_type(),
            Light::Projection(l) => l.light_type(),
            Light::Goniometric(l) => l.light_type(),
            Light::Spot(l) => l.light_type(),
            Light::DiffuseArea(l) => l.light_type(),
            Light::Infinite(l) => l.light_type(),
        }
    }

    pub fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            Light::Point(l) => l.phi(lambda),
            Light::Distant(l) => l.phi(lambda),
            Light::Projection(l) => l.phi(lambda),
            Light::Goniometric(l) => l.phi(lambda),
            Light::Spot(l) => l.phi(lambda),
            Light::DiffuseArea(l) => l.phi(lambda),
            Light::Infinite(l) => l.phi(lambda),
        }
    }

    pub fn sample_li(
        &self,
        ctx: &LightSampleContext,
        u: Point2f,
        lambda: &SampledWavelengths,
        allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        match self {
            Light::Point(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
            Light::Distant(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
            Light::Projection(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
            Light::Goniometric(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
            Light::Spot(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
            Light::DiffuseArea(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
            Light::Infinite(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
        }
    }

    pub fn pdf_li(
        &self,
        ctx: &LightSampleContext,
        wi: Vector3f,
        allow_incomplete_pdf: bool,
    ) -> Float {
        match self {
            Light::Point(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
            Light::Distant(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
            Light::Projection(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
            Light::Goniometric(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
            Light::Spot(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
            Light::DiffuseArea(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
            Light::Infinite(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
        }
    }

    pub fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            Light::Infinite(l) => l.le(ray, lambda),
            _ => SampledSpectrum::zero(),
        }
    }

    pub fn preprocess(&self, scene_bounds: &Bounds3f) {
        match self {
            Light::Distant(l) => l.preprocess(scene_bounds),
            Light::Infinite(l) => l.preprocess(scene_bounds),
            _ => {}
        }
    }

    pub fn sample_le(
        &self,
        u1: Point2f,
        u2: Point2f,
        lambda: &SampledWavelengths,
        time: Float,
    ) -> Option<LightLeSample> {
        match self {
            Light::Point(l) => l.sample_le(u1, u2, lambda, time),
            Light::Distant(l) => l.sample_le(u1, u2, lambda, time),
            Light::Projection(l) => l.sample_le(u1, u2, lambda, time),
            Light::Goniometric(l) => l.sample_le(u1, u2, lambda, time),
            Light::Spot(l) => l.sample_le(u1, u2, lambda, time),
            Light::DiffuseArea(l) => l.sample_le(u1, u2, lambda, time),
            Light::Infinite(l) => l.sample_le(u1, u2, lambda, time),
        }
    }

    pub fn pdf_le_ray(&self, ray: &Ray) -> (Float, Float) {
        match self {
            Light::Point(l) => l.pdf_le_ray(ray),
            Light::Distant(l) => l.pdf_le_ray(ray),
            Light::Projection(l) => l.pdf_le_ray(ray),
            Light::Goniometric(l) => l.pdf_le_ray(ray),
            Light::Spot(l) => l.pdf_le_ray(ray),
            Light::DiffuseArea(l) => l.pdf_le_ray(ray),
            Light::Infinite(l) => l.pdf_le_ray(ray),
        }
    }

    pub fn pdf_le_interaction(&self, intr: &Interaction, w: Vector3f) -> (Float, Float) {
        match self {
            Light::DiffuseArea(l) => l.pdf_le_interaction(intr, w),
            _ => (0.0, 0.0),
        }
    }

    pub fn l(
        &self,
        p: Point3f,
        n: Normal3f,
        uv: Point2f,
        w: Vector3f,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        match self {
            Light::DiffuseArea(l) => l.l(p, n, uv, w, lambda),
            _ => SampledSpectrum::zero(),
        }
    }

    pub fn bounds(&self) -> Option<LightBounds> {
        match self {
            Light::Point(l) => l.bounds(),
            Light::Distant(l) => l.bounds(),
            Light::Projection(l) => l.bounds(),
            Light::Goniometric(l) => l.bounds(),
            Light::Spot(l) => l.bounds(),
            Light::DiffuseArea(l) => l.bounds(),
            Light::Infinite(l) => l.bounds(),
        }
    }

    pub fn get_light_flags(&self) -> u32 {
        self.light_type() as u32
    }
    pub fn is_infinite(&self) -> bool {
        matches!(self.light_type(), LightType::Infinite)
    }
    pub fn is_delta(&self) -> bool {
        is_delta_light(self.light_type())
    }
    pub fn is_delta_direction(&self) -> bool {
        matches!(self.light_type(), LightType::DeltaDirection)
    }
    pub fn is_area(&self) -> bool {
        matches!(self.light_type(), LightType::Area)
    }
    pub fn get_sample_count(&self) -> u32 {
        1
    }
}
