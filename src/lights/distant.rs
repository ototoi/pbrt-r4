// pbrt-v4 verbatim translation of `class DistantLight` (lights.h:242-297,
// lights.cpp:215-276). A `DeltaDirection` (parallel) light source whose
// direction is the light-local +z axis after `renderFromLight`.

use crate::base::light::*;
use crate::base::lightsampler::LightSampleContext;
use crate::interaction::*;
use crate::lights::*;
use crate::media::*;
use crate::paramdict::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::profile::*;
use crate::util::sampling::*;
use crate::util::spectrum::*;
use crate::util::transform::*;

use std::sync::Arc;
use std::sync::RwLock;

pub struct DistantLight {
    base: LightBase,
    // v4 `const DenselySampledSpectrum *Lemit`.
    l_emit: Spectrum,
    // v4 `Float scale` (lights.h:294).
    scale: Float,
    // v4 preprocess fills sceneCenter / sceneRadius.
    scene_center: RwLock<Point3f>,
    scene_radius: RwLock<Float>,
    // v4 stores `renderFromLight` and recomputes direction; r4 caches.
    wi: Vector3f,
}

impl DistantLight {
    pub fn new(render_from_light: &Transform, l: Spectrum, scale: Float, wi: &Vector3f) -> Self {
        let wi = render_from_light.transform_vector(wi).normalize();
        let base = LightBase::new(
            LightType::DeltaDirection as u32,
            render_from_light,
            &MediumInterface::new(),
        );
        DistantLight {
            base,
            l_emit: l,
            scale,
            scene_center: RwLock::new(Point3f::new(0.0, 0.0, 0.0)),
            scene_radius: RwLock::new(Float::INFINITY),
            wi,
        }
    }

    fn sampled_emission(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.l_emit.sample(lambda) * self.scale
    }

    pub fn create(
        render_from_light: &Transform,
        params: &ParameterDictionary,
    ) -> Result<Self, PbrtError> {
        // v4 lights.cpp:246-275 `DistantLight::Create`. "scale" is a Float;
        // v4 normalizes via `sc /= SpectrumToPhotometric(L)` (line 266)
        // and multiplies by an optional `illuminance` value (lines 271-273).
        let white = Spectrum::from(params.color_space().illuminant.to_dense());
        let l = params.get_one_spectrum_typed("L", &white, SpectrumType::Illuminant);
        let photometric = spectrum_to_photometric(&l);
        let mut sc =
            params.get_one_float("scale", 1.0) / if photometric > 0.0 { photometric } else { 1.0 };
        let e_v = params.get_one_float("illuminance", -1.0);
        if e_v > 0.0 {
            sc *= e_v;
        }
        let from = params.get_one_point3f("from", &Point3f::new(0.0, 0.0, 0.0));
        let to = params.get_one_point3f("to", &Point3f::new(0.0, 0.0, 1.0));
        let w = from - to;
        Ok(DistantLight::new(render_from_light, l, sc, &w))
    }
}

impl DistantLight {
    pub fn light_type(&self) -> LightType {
        LightType::DeltaDirection
    }

    // v4 lights.cpp:216-218:
    //   return scale * Lemit->Sample(lambda) * Pi * Sqr(sceneRadius);
    pub fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let r = *self.scene_radius.read().unwrap();
        self.sampled_emission(lambda) * (PI * r * r)
    }

    // v4 lights.h:277-279:
    //   sceneBounds.BoundingSphere(&sceneCenter, &sceneRadius);
    pub fn preprocess(&self, scene_bounds: &Bounds3f) {
        let (center, radius) = scene_bounds.bounding_sphere();
        *self.scene_center.write().unwrap() = center;
        *self.scene_radius.write().unwrap() = radius;
    }

    // v4 lights.h:282-289:
    //   Vector3f wi = Normalize(renderFromLight(Vector3f(0,0,1)));
    //   Point3f pOutside = ctx.p() + wi * (2 * sceneRadius);
    //   return LightLiSample(scale * Lemit->Sample(lambda), wi, 1,
    //                        Interaction(pOutside, nullptr));
    pub fn sample_li(
        &self,
        ctx: &LightSampleContext,
        _u: Point2f,
        lambda: &SampledWavelengths,
        _allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        let _p = ProfilePhase::new(Prof::LightSample);
        let radius = *self.scene_radius.read().unwrap();
        let w = self.wi;
        let p_outside = ctx.p + w * (2.0 * radius);
        // v4: `Interaction(pOutside, nullptr)` -- no medium interface.
        let p_light = Interaction::from_light_sample(&p_outside, 0.0, &self.base.medium_interface);
        Some(LightLiSample::new(
            self.sampled_emission(lambda),
            w,
            1.0,
            p_light,
        ))
    }

    pub fn pdf_li(
        &self,
        _ctx: &LightSampleContext,
        _wi: Vector3f,
        _allow_incomplete_pdf: bool,
    ) -> Float {
        0.0
    }

    // v4 lights.cpp:220-234:
    //   Vector3f w = Normalize(renderFromLight(Vector3f(0,0,1)));
    //   Frame wFrame = Frame::FromZ(w);
    //   Point2f cd = SampleUniformDiskConcentric(u1);
    //   Point3f pDisk = sceneCenter + sceneRadius * wFrame.FromLocal(Vector3f(cd.x, cd.y, 0));
    //   Ray ray(pDisk + sceneRadius * w, -w, time);
    //   return LightLeSample(scale * Lemit->Sample(lambda), ray, 1/(Pi*Sqr(sceneRadius)), 1);
    pub fn sample_le(
        &self,
        u1: Point2f,
        _u2: Point2f,
        lambda: &SampledWavelengths,
        time: Float,
    ) -> Option<LightLeSample> {
        let _p = ProfilePhase::new(Prof::LightSample);
        let w = self.wi;
        let scene_center = *self.scene_center.read().unwrap();
        let scene_radius = *self.scene_radius.read().unwrap();
        let (v1, v2) = coordinate_system(&w);
        let cd = concentric_sample_disk(&u1);
        let p_disk = scene_center + scene_radius * (cd.x * v1 + cd.y * v2);
        let ray = Ray::new(&(p_disk + scene_radius * w), &-w, Float::INFINITY, time);
        let pdf_pos = 1.0 / (PI * scene_radius * scene_radius);
        Some(LightLeSample::new(
            self.sampled_emission(lambda),
            ray,
            pdf_pos,
            1.0,
        ))
    }

    // v4 lights.cpp:236-239:
    //   *pdfPos = 1 / (Pi * sceneRadius * sceneRadius);
    //   *pdfDir = 0;
    pub fn pdf_le_ray(&self, _ray: &Ray) -> (Float, Float) {
        let _p = ProfilePhase::new(Prof::LightPdf);
        let r = *self.scene_radius.read().unwrap();
        (1.0 / (PI * r * r), 0.0)
    }

    // v4 lights.h:273: `pstd::optional<LightBounds> Bounds() const { return {}; }`
    pub fn bounds(&self) -> Option<LightBounds> {
        None
    }
}
