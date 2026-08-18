use crate::base::light::*;
use crate::base::lightsampler::LightSampleContext;
use crate::interaction::*;
use crate::lights::*;
use crate::media::*;
use crate::paramdict::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::sampling::*;
use crate::util::spectrum::*;
use crate::util::transform::*;

use std::sync::Arc;

#[derive(Clone)]
pub struct PointLight {
    base: LightBase,
    intensity: Arc<DenselySampledSpectrum>,
    scale: Float,
}

fn lookup_spectrum(s: &Spectrum) -> Arc<DenselySampledSpectrum> {
    Arc::new(s.to_dense())
}

impl PointLight {
    pub fn new(
        light_to_world: &Transform,
        medium_interface: &MediumInterface,
        intensity: &Spectrum,
        scale: Float,
    ) -> Self {
        let base = LightBase::new(
            LightType::DeltaPosition as u32,
            light_to_world,
            medium_interface,
        );
        PointLight {
            base,
            intensity: lookup_spectrum(intensity),
            scale,
        }
    }

    /// pbrt-v4 `scale * I->Sample(lambda)`.
    fn sampled_intensity(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.intensity.sample(lambda) * self.scale
    }

    pub fn create(
        light2world: &Transform,
        medium: &Option<Arc<Medium>>,
        params: &ParameterDictionary,
    ) -> Result<Self, PbrtError> {
        // v4 lights.cpp:192-213 `PointLight::Create`. "scale" is a Float;
        // v4 normalizes via `sc /= SpectrumToPhotometric(I)` (line 200)
        // and applies a power adjustment `sc *= phi_v / (4*Pi)` when
        // `power` is set (lines 202-206).
        let color_space = params.color_space();
        let white: DenselySampledSpectrum = color_space.illuminant.to_dense();
        let white = Spectrum::from(white);
        let i = params.get_one_spectrum_typed("I", &white, SpectrumType::Illuminant);
        let photometric = spectrum_to_photometric(&i);
        let mut sc =
            params.get_one_float("scale", 1.0) / if photometric > 0.0 { photometric } else { 1.0 };
        let phi_v = params.get_one_float("power", -1.0);
        if phi_v > 0.0 {
            sc *= phi_v / (4.0 * std::f64::consts::PI as Float);
        }
        let p = params.get_one_point3f("from", &Point3f::zero());
        let l2w = (*light2world) * Transform::translate(p.x, p.y, p.z);
        let mi = MediumInterface::from(medium);
        Ok(PointLight::new(&l2w, &mi, &i, sc))
    }
}

impl PointLight {
    pub fn light_type(&self) -> LightType {
        LightType::DeltaPosition
    }

    // v4: SampledSpectrum PointLight::Phi(SampledWavelengths lambda) const
    //   { return 4 * Pi * scale * I->Sample(lambda); }
    pub fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.sampled_intensity(lambda) * (4.0 * PI)
    }

    // v4 lights.h:221-228:
    //   Point3f p = renderFromLight(Point3f(0, 0, 0));
    //   Vector3f wi = Normalize(p - ctx.p());
    //   SampledSpectrum Li = scale * I->Sample(lambda) / DistanceSquared(p, ctx.p());
    //   return LightLiSample(Li, wi, 1, Interaction(p, &mediumInterface));
    pub fn sample_li(
        &self,
        ctx: &LightSampleContext,
        _u: Point2f,
        lambda: &SampledWavelengths,
        _allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        let p = self
            .base
            .render_from_light
            .transform_point(&Point3f::zero());
        let wi = (p - ctx.p).normalize();
        let li = self.sampled_intensity(lambda) / Vector3f::distance_squared(&p, &ctx.p);
        let p_light = Interaction::from_light_sample(&p, 0.0, &self.base.medium_interface);
        Some(LightLiSample::new(li, wi, 1.0, p_light))
    }

    // v4 lights.h:230-233: returns 0 (delta light).
    pub fn pdf_li(
        &self,
        _ctx: &LightSampleContext,
        _wi: Vector3f,
        _allow_incomplete_pdf: bool,
    ) -> Float {
        0.0
    }

    // v4 lights.cpp:175-181:
    //   Point3f p = renderFromLight(Point3f(0, 0, 0));
    //   Ray ray(p, SampleUniformSphere(u1), time, mediumInterface.outside);
    //   return LightLeSample(scale * I->Sample(lambda), ray, 1, UniformSpherePDF());
    pub fn sample_le(
        &self,
        u1: Point2f,
        _u2: Point2f,
        lambda: &SampledWavelengths,
        time: Float,
    ) -> Option<LightLeSample> {
        let p = self
            .base
            .render_from_light
            .transform_point(&Point3f::zero());
        let medium = self.base.medium_interface.get_outside();
        let d = uniform_sample_sphere(&u1);
        let ray = Ray::from((&p, &d, Float::INFINITY, time, &medium));
        Some(LightLeSample::new(
            self.sampled_intensity(lambda),
            ray,
            1.0,
            uniform_sphere_pdf(),
        ))
    }

    // v4 lights.cpp:183-186:
    //   *pdfPos = 0;
    //   *pdfDir = UniformSpherePDF();
    pub fn pdf_le_ray(&self, _ray: &Ray) -> (Float, Float) {
        (0.0, uniform_sphere_pdf())
    }

    // v4 lights.h:168-173:
    //   Float phi = 4 * Pi * scale * I->MaxValue();
    //   return LightBounds(Bounds3f(p, p), Vector3f(0,0,1), phi, cos(Pi), cos(Pi/2), false);
    pub fn bounds(&self) -> Option<LightBounds> {
        let p = self
            .base
            .render_from_light
            .transform_point(&Point3f::zero());
        let phi = 4.0 * PI * self.intensity.max_value() * self.scale;
        Some(LightBounds::new(
            Bounds3f::new(&p, &p),
            Vector3f::new(0.0, 0.0, 1.0),
            phi,
            Float::cos(PI),
            Float::cos(PI / 2.0),
            false,
        ))
    }
}
