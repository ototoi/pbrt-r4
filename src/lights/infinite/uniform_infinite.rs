use crate::base::light::*;
use crate::base::lightsampler::LightSampleContext;
use crate::interaction::*;
use crate::lights::*;
use crate::media::*;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::geometry::{Bounds3f, Ray};
use crate::util::sampling::*;
use crate::util::spectrum::*;
use crate::util::transform::*;

use std::sync::RwLock;

// ===========================================================================
// UniformInfiniteLight - v4 lights.cpp:943-1004
// ===========================================================================

pub struct UniformInfiniteLight {
    base: LightBase,
    // v4 `const DenselySampledSpectrum *Lemit`.
    l_emit: Spectrum,
    // v4 `Float scale` (lights.h:538).
    scale: Float,
    scene_center: RwLock<Point3f>,
    scene_radius: RwLock<Float>,
}

impl UniformInfiniteLight {
    pub fn new(
        light_to_world: &Transform,
        medium_interface: &MediumInterface,
        l_emit: Spectrum,
        scale: Float,
    ) -> Self {
        let base = LightBase::new(LightType::Infinite as u32, light_to_world, medium_interface);
        Self {
            base,
            l_emit,
            scale,
            scene_center: RwLock::new(Point3f::zero()),
            scene_radius: RwLock::new(1.0),
        }
    }

    fn radius(&self) -> Float {
        *self.scene_radius.read().unwrap()
    }

    fn sampled_emission(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.l_emit.sample(lambda) * self.scale
    }
}

impl UniformInfiniteLight {
    pub fn light_type(&self) -> LightType {
        LightType::Infinite
    }

    pub fn preprocess(&self, scene_bounds: &Bounds3f) {
        let (center, radius) = scene_bounds.bounding_sphere();
        *self.scene_center.write().unwrap() = center;
        *self.scene_radius.write().unwrap() = radius;
    }

    // v4 lights.cpp:950-953:
    //   return scale * Lemit->Sample(lambda);
    pub fn le(&self, _ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.sampled_emission(lambda)
    }

    // v4 lights.cpp:974-976:
    //   return 4 * Pi * Pi * Sqr(sceneRadius) * scale * Lemit->Sample(lambda);
    pub fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let r = self.radius();
        self.sampled_emission(lambda) * (4.0 * PI * PI * r * r)
    }

    // v4 lights.cpp:955-965:
    //   if (allowIncompletePDF) return {};
    //   Vector3f wi = SampleUniformSphere(u);
    //   Float pdf = UniformSpherePDF();
    //   return LightLiSample(scale * Lemit->Sample(lambda), wi, pdf,
    //                        Interaction(ctx.p() + wi*(2*sceneRadius), &mediumInterface));
    pub fn sample_li(
        &self,
        ctx: &LightSampleContext,
        u: Point2f,
        lambda: &SampledWavelengths,
        allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        if allow_incomplete_pdf {
            return None;
        }
        let wi = uniform_sample_sphere(&u);
        let pdf = uniform_sphere_pdf();
        let p = ctx.p + wi * (2.0 * self.radius());
        let p_light = Interaction::from_light_sample(&p, 0.0, &self.base.medium_interface);
        Some(LightLiSample::new(
            self.sampled_emission(lambda),
            wi,
            pdf,
            p_light,
        ))
    }

    // v4 lights.cpp:967-972:
    //   if (allowIncompletePDF) return 0;
    //   return UniformSpherePDF();
    pub fn pdf_li(
        &self,
        _ctx: &LightSampleContext,
        _wi: Vector3f,
        allow_incomplete_pdf: bool,
    ) -> Float {
        if allow_incomplete_pdf {
            0.0
        } else {
            uniform_sphere_pdf()
        }
    }

    // v4 lights.cpp:978-995.
    pub fn sample_le(
        &self,
        u1: Point2f,
        u2: Point2f,
        lambda: &SampledWavelengths,
        time: Float,
    ) -> Option<LightLeSample> {
        let w = uniform_sample_sphere(&u1);
        let center = *self.scene_center.read().unwrap();
        let radius = *self.scene_radius.read().unwrap();
        let (v1, v2) = coordinate_system(&-w);
        let cd = concentric_sample_disk(&u2);
        let p_disk = center + radius * (cd.x * v1 + cd.y * v2);
        let ray = Ray::new(&(p_disk + radius * -w), &w, Float::INFINITY, time);
        let pdf_pos = 1.0 / (PI * radius * radius);
        let pdf_dir = uniform_sphere_pdf();
        Some(LightLeSample::new(
            self.sampled_emission(lambda),
            ray,
            pdf_pos,
            pdf_dir,
        ))
    }

    // v4 lights.cpp:997-1000.
    pub fn pdf_le_ray(&self, _ray: &Ray) -> (Float, Float) {
        let r = self.radius();
        let pdf_pos = 1.0 / (PI * r * r);
        let pdf_dir = uniform_sphere_pdf();
        (pdf_pos, pdf_dir)
    }

    // v4 lights.h:531: `Bounds() const { return {}; }`
    pub fn bounds(&self) -> Option<LightBounds> {
        None
    }
}

unsafe impl Send for UniformInfiniteLight {}
unsafe impl Sync for UniformInfiniteLight {}
