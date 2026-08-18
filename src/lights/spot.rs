use crate::base::light::*;
use crate::base::lightsampler::LightSampleContext;
use crate::interaction::*;
use crate::lights::*;
use crate::media::*;
use crate::paramdict::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::math::newton_bisection;
use crate::util::sampling::safe_sqrt as sampling_safe_sqrt;
use crate::util::sampling::*;
use crate::util::scattering::cos_theta;
use crate::util::spectrum::*;
use crate::util::transform::*;

use std::sync::Arc;

#[derive(Clone)]
pub struct SpotLight {
    base: LightBase,
    // v4 `const DenselySampledSpectrum *Iemit`.
    intensity: Spectrum,
    // v4 `Float scale` (lights.h:788).
    scale: Float,
    // v4 `Float cosFalloffStart, cosFalloffEnd`.
    cos_falloff_start: Float,
    cos_falloff_end: Float,
}

#[inline]
fn radians(x: Float) -> Float {
    x * (PI / 180.0)
}

// pbrt-v4 `inline Float SmoothStep(Float x, Float a, Float b)`
// (util/math.h:268-274).
#[inline]
fn smooth_step(x: Float, a: Float, b: Float) -> Float {
    if a == b {
        return if x < a { 0.0 } else { 1.0 };
    }
    let t = Float::clamp((x - a) / (b - a), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// pbrt-v4 `inline Float SmoothStepPDF(Float x, Float a, Float b)`
// (util/sampling.h:287).
#[inline]
fn smooth_step_pdf(x: Float, a: Float, b: Float) -> Float {
    if x < a || x > b {
        return 0.0;
    }
    (2.0 / (b - a)) * smooth_step(x, a, b)
}

// pbrt-v4 `inline Float SampleSmoothStep(Float u, Float a, Float b)`
// (util/sampling.h:294).
#[inline]
fn sample_smooth_step(u: Float, a: Float, b: Float) -> Float {
    if a == b {
        return a;
    }
    let cdf_minus_u = |x: Float| {
        let t = (x - a) / (b - a);
        let cdf = 2.0 * t * t * t - t * t * t * t;
        (cdf - u, smooth_step_pdf(x, a, b))
    };
    newton_bisection(a, b, cdf_minus_u)
}

// pbrt-v4 `int SampleDiscrete(span weights, Float u, Float *pmf,
// Float *uRemapped)` (util/sampling.h:79). Returns (index, pmf,
// remapped).
fn sample_discrete2(p0: Float, p1: Float, u: Float) -> (usize, Float, Float) {
    let sum = p0 + p1;
    if sum == 0.0 {
        return (0, 0.0, u);
    }
    let q0 = p0 / sum;
    if u < q0 {
        let pmf = q0;
        let u_remapped = u / q0;
        (0, pmf, u_remapped)
    } else {
        let pmf = 1.0 - q0;
        let u_remapped = (u - q0) / pmf;
        (1, pmf, u_remapped)
    }
}

impl SpotLight {
    pub fn new(
        light_to_world: &Transform,
        medium_interface: &MediumInterface,
        intensity: Spectrum,
        scale: Float,
        total_width: Float,
        falloff_start: Float,
    ) -> Self {
        let base = LightBase::new(
            LightType::DeltaPosition as u32,
            light_to_world,
            medium_interface,
        );
        // v4 ctor uses `cosFalloffEnd = cos(Radians(totalWidth))` and
        // `cosFalloffStart = cos(Radians(falloffStart))`.
        let cos_falloff_end = Float::cos(radians(total_width));
        let cos_falloff_start = Float::cos(radians(falloff_start));
        SpotLight {
            base,
            intensity,
            scale,
            cos_falloff_start,
            cos_falloff_end,
        }
    }

    fn sampled_intensity(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.intensity.sample(lambda) * self.scale
    }

    // v4 lights.cpp:1360-1363:
    //   SampledSpectrum SpotLight::I(Vector3f w, SampledWavelengths lambda) const {
    //     return SmoothStep(CosTheta(w), cosFalloffEnd, cosFalloffStart) *
    //            scale * Iemit->Sample(lambda);
    //   }
    fn i(&self, w: Vector3f, lambda: &SampledWavelengths) -> SampledSpectrum {
        let s = smooth_step(cos_theta(&w), self.cos_falloff_end, self.cos_falloff_start);
        self.sampled_intensity(lambda) * s
    }
}

impl SpotLight {
    pub fn light_type(&self) -> LightType {
        LightType::DeltaPosition
    }

    // v4 lights.cpp:1365-1368:
    //   SampledSpectrum Phi(...) {
    //     return scale * Iemit->Sample(lambda) * 2 * Pi *
    //            ((1 - cosFalloffStart) + (cosFalloffStart - cosFalloffEnd) / 2);
    //   }
    pub fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let solid = 2.0
            * PI
            * ((1.0 - self.cos_falloff_start)
                + (self.cos_falloff_start - self.cos_falloff_end) / 2.0);
        self.sampled_intensity(lambda) * solid
    }

    // v4 lights.h:770-783:
    //   Point3f p = renderFromLight(Point3f(0, 0, 0));
    //   Vector3f wi = Normalize(p - ctx.p());
    //   Vector3f wLight = Normalize(renderFromLight.ApplyInverse(-wi));
    //   SampledSpectrum Li = I(wLight, lambda) / DistanceSquared(p, ctx.p());
    //   if (!Li) return {};
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
        let w_light = self
            .base
            .world_to_light()
            .transform_vector(&-wi)
            .normalize();
        let li = self.i(w_light, lambda) / Vector3f::distance_squared(&p, &ctx.p);
        if li.is_black() {
            return None;
        }
        let p_light = Interaction::from_light_sample(&p, 0.0, &self.base.medium_interface);
        Some(LightLiSample::new(li, wi, 1.0, p_light))
    }

    pub fn pdf_li(
        &self,
        _ctx: &LightSampleContext,
        _wi: Vector3f,
        _allow_incomplete_pdf: bool,
    ) -> Float {
        0.0
    }

    // v4 lights.cpp:1382-1413: choose either the inner cone or the
    // falloff annulus by SampleDiscrete, then sample within it.
    pub fn sample_le(
        &self,
        u1: Point2f,
        u2: Point2f,
        lambda: &SampledWavelengths,
        time: Float,
    ) -> Option<LightLeSample> {
        let p0 = 1.0 - self.cos_falloff_start;
        let p1 = (self.cos_falloff_start - self.cos_falloff_end) / 2.0;
        let (section, section_pdf, _) = sample_discrete2(p0, p1, u2.x);

        let (w_light, pdf_dir) = if section == 0 {
            // v4: sample center cone.
            let w = uniform_sample_cone(&u1, self.cos_falloff_start);
            (w, uniform_cone_pdf(self.cos_falloff_start) * section_pdf)
        } else {
            // v4: sample falloff annulus via SampleSmoothStep on cosTheta.
            let cos_theta_v =
                sample_smooth_step(u1.x, self.cos_falloff_end, self.cos_falloff_start);
            let sin_theta_v = sampling_safe_sqrt(1.0 - cos_theta_v * cos_theta_v);
            let phi = u1.y * 2.0 * PI;
            let w = spherical_direction(sin_theta_v, cos_theta_v, phi);
            let pdf = smooth_step_pdf(cos_theta_v, self.cos_falloff_end, self.cos_falloff_start)
                * section_pdf
                / (2.0 * PI);
            (w, pdf)
        };

        let medium = self.base.medium_interface.get_outside();
        // v4: `Ray ray = renderFromLight(Ray(P(0,0,0), wLight, time, medium.outside));`
        let d = self.base.render_from_light.transform_vector(&w_light);
        let o = self
            .base
            .render_from_light
            .transform_point(&Point3f::zero());
        let ray = Ray::from((&o, &d, Float::INFINITY, time, &medium));
        Some(LightLeSample::new(
            self.i(w_light, lambda),
            ray,
            1.0,
            pdf_dir,
        ))
    }

    // v4 lights.cpp:1415-1425.
    pub fn pdf_le_ray(&self, ray: &Ray) -> (Float, Float) {
        let p0 = 1.0 - self.cos_falloff_start;
        let p1 = (self.cos_falloff_start - self.cos_falloff_end) / 2.0;
        let cos_theta_v = cos_theta(&self.base.world_to_light().transform_vector(&ray.d));
        let pdf_dir = if cos_theta_v >= self.cos_falloff_start {
            uniform_cone_pdf(self.cos_falloff_start) * p0 / (p0 + p1)
        } else {
            smooth_step_pdf(cos_theta_v, self.cos_falloff_end, self.cos_falloff_start) * p1
                / ((p0 + p1) * (2.0 * PI))
        };
        (0.0, pdf_dir)
    }

    // v4 lights.cpp:1370-1380.
    pub fn bounds(&self) -> Option<LightBounds> {
        let p = self
            .base
            .render_from_light
            .transform_point(&Point3f::zero());
        let w = self
            .base
            .render_from_light
            .transform_vector(&Vector3f::new(0.0, 0.0, 1.0))
            .normalize();
        let phi = self.intensity.max_value() * self.scale * 4.0 * PI;
        let mut cos_theta_e =
            Float::cos(Float::acos(self.cos_falloff_end) - Float::acos(self.cos_falloff_start));
        if cos_theta_e == 1.0 && self.cos_falloff_end != self.cos_falloff_start {
            cos_theta_e = 0.999;
        }
        Some(LightBounds::new(
            Bounds3f::new(&p, &p),
            w,
            phi,
            self.cos_falloff_start,
            cos_theta_e,
            false,
        ))
    }

    fn get_light_flags(&self) -> u32 {
        self.base.flags
    }
}

impl SpotLight {
    pub fn create(
        l2w: &Transform,
        medium: &Option<Arc<Medium>>,
        params: &ParameterDictionary,
    ) -> Result<Self, PbrtError> {
        // v4 lights.cpp:1433-1464 `SpotLight::Create`. "scale" is a Float;
        // v4 normalizes via `sc /= SpectrumToPhotometric(I)` (line 1451)
        // and applies a cone-aware power adjustment when `power` is set
        // (lines 1453-1460).
        let white = Spectrum::from(params.color_space().illuminant.to_dense());
        let intensity = params.get_one_spectrum_typed("I", &white, SpectrumType::Illuminant);
        let photometric = spectrum_to_photometric(&intensity);
        let mut sc =
            params.get_one_float("scale", 1.0) / if photometric > 0.0 { photometric } else { 1.0 };
        let coneangle = params.get_one_float("coneangle", 30.0);
        let conedelta = params.get_one_float("conedelta", 5.0);
        let conedelta = params.get_one_float("conedeltaangle", conedelta);
        let phi_v = params.get_one_float("power", -1.0);
        if phi_v > 0.0 {
            let cos_falloff_end = (coneangle.to_radians()).cos();
            let cos_falloff_start = ((coneangle - conedelta).to_radians()).cos();
            let two_pi = 2.0 * std::f64::consts::PI as Float;
            let k_e =
                two_pi * ((1.0 - cos_falloff_start) + (cos_falloff_start - cos_falloff_end) / 2.0);
            sc *= phi_v / k_e;
        }
        let from = params.get_one_point3f("from", &Point3f::new(0.0, 0.0, 0.0));
        let to = params.get_one_point3f("to", &Point3f::new(0.0, 0.0, 1.0));
        let dir = (to - from).normalize();
        let (dir, du, dv) = Vector3f::coordinate_system(&dir);
        let dir_to_z = Transform::from(Matrix4x4::new(
            du.x, du.y, du.z, 0.0, dv.x, dv.y, dv.z, 0., dir.x, dir.y, dir.z, 0.0, 0.0, 0.0, 0.0,
            1.0,
        ));
        let light2world =
            *l2w * Transform::translate(from.x, from.y, from.z) * Transform::inverse(&dir_to_z);
        let mi = MediumInterface::from(medium);
        Ok(SpotLight::new(
            &light2world,
            &mi,
            intensity,
            sc,
            coneangle,
            coneangle - conedelta,
        ))
    }
}
