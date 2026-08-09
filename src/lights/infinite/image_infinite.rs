use crate::base::light::*;
use crate::base::lightsampler::LightSampleContext;
use crate::interaction::*;
use crate::lights::*;
use crate::media::*;
use crate::textures::{ImageWrap, MIPMap};
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::profile::*;
use crate::util::sampling::*;
use crate::util::spectrum::rgb_to_spectrum::RGBColorSpace;
use crate::util::spectrum::*;
use crate::util::transform::*;
use std::sync::RwLock;

// ===========================================================================
// ImageInfiniteLight - v4 lights.cpp:1006-1106
// ===========================================================================

pub struct ImageInfiniteLight {
    base: LightBase,
    lmap: MIPMap<RGBSpectrum>,
    // v4 `Float scale` (lights.h:623).
    scale: Float,
    // v4 stores `const RGBColorSpace *imageColorSpace` on the light so
    // `image_le` can interpret each pixel through the table that
    // matches the EXR's chromaticities (lights.cpp:1607,1656). r4
    // defaults to sRGB for EXRs that don't carry chromaticities.
    image_color_space: &'static RGBColorSpace,
    scene_center: RwLock<Point3f>,
    scene_radius: RwLock<Float>,
    distribution: Distribution2D,
    // pbrt-v4 (lights.cpp:1020-1029) keeps a second distribution that
    // subtracts the image's per-pixel-average from each pixel and
    // clips at zero, leaving only the "peaks" (sun, bright clouds).
    // `sample_li(..., allow_incomplete_pdf = true)` uses it so MIS-driven
    // path tracers concentrate NEE on the bright spots; the regular
    // distribution is still used for the unconditional PDF query.
    compensated_distribution: Distribution2D,
}

impl ImageInfiniteLight {
    pub fn new(
        light_to_world: &Transform,
        medium_interface: &MediumInterface,
        lmap: MIPMap<RGBSpectrum>,
        scale: Float,
        distribution: Distribution2D,
        compensated_distribution: Distribution2D,
        image_color_space: &'static RGBColorSpace,
    ) -> Self {
        let base = LightBase::new(LightType::Infinite as u32, light_to_world, medium_interface);
        ImageInfiniteLight {
            base,
            lmap,
            scale,
            image_color_space,
            compensated_distribution,
            scene_center: RwLock::new(Point3f::zero()),
            scene_radius: RwLock::new(Float::INFINITY),
            distribution,
        }
    }

    fn radius(&self) -> Float {
        *self.scene_radius.read().unwrap()
    }

    // Lookup with octahedral wrapping and evaluate through the image color
    // space before applying the light scale.
    fn image_le(&self, uv: Point2f, lambda: &SampledWavelengths) -> SampledSpectrum {
        let w = self.lmap.width() as Float;
        let h = self.lmap.height() as Float;
        let sx = (uv.x * w).floor() as i32;
        let sy = (uv.y * h).floor() as i32;
        let rgb = MIPMap::<RGBSpectrum>::texel_static(
            &self.lmap.storage.pyramid[0],
            sx,
            sy,
            ImageWrap::OctahedralSphere,
            ImageWrap::OctahedralSphere,
        );
        let spec = self
            .image_color_space
            .illuminant_to_sampled_spectrum(rgb.to_rgb(), lambda);
        self.scale * spec
    }
}

impl ImageInfiniteLight {
    pub fn light_type(&self) -> LightType {
        LightType::Infinite
    }

    pub fn preprocess(&self, scene_bounds: &Bounds3f) {
        let (center, radius) = scene_bounds.bounding_sphere();
        *self.scene_center.write().unwrap() = center;
        *self.scene_radius.write().unwrap() = radius;
    }

    // v4 lights.h:573-578:
    //   Vector3f wLight = Normalize(renderFromLight.ApplyInverse(ray.d));
    //   Point2f uv = EqualAreaSphereToSquare(wLight);
    //   return ImageLe(uv, lambda);
    pub fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        let w = self
            .base
            .world_to_light()
            .transform_vector(&ray.d)
            .normalize();
        let uv = equal_area_sphere_to_square(&w);
        self.image_le(uv, lambda)
    }

    // v4 `ImageInfiniteLight::Phi` (lights.cpp:1054-1071): integrate
    // each pixel's RGB-illuminant emission over the whole image and
    // convert to power via 4π²·r²·scale.
    //
    pub fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        // v4 `ImageInfiniteLight::Phi` integrates spectrum(rgb_i, lambda)
        // over every base-level pixel. The MIPMap top would be O(1) but
        // Because RGB-to-spectrum reconstruction is nonlinear, the
        // spectrum of the mean RGB value is not the mean spectrum.
        use rayon::prelude::*;
        let r = self.radius();
        let w = self.lmap.width() as i32;
        let h = self.lmap.height() as i32;
        let total = (w as usize) * (h as usize);
        let sum = (0..total)
            .into_par_iter()
            .map(|i| {
                let x = (i % w as usize) as i32;
                let y = (i / w as usize) as i32;
                let rgb = self.lmap.texel(0, x, y).to_rgb();
                self.image_color_space
                    .illuminant_to_sampled_spectrum(rgb, lambda)
            })
            .reduce(SampledSpectrum::zero, |a, b| a + b);
        let inv = 1.0 / total as Float;
        sum * (4.0 * PI * PI * r * r * self.scale * inv)
    }

    // v4 lights.h:581-605:
    //   uv = (allowIncompletePDF ? compensatedDistribution : distribution).Sample(u, &mapPDF);
    //   if (mapPDF == 0) return {};
    //   Vector3f wLight = EqualAreaSquareToSphere(uv);
    //   Vector3f wi = renderFromLight(wLight);
    //   Float pdf = mapPDF / (4 * Pi);
    //   return LightLiSample(ImageLe(uv, lambda), wi, pdf,
    //                        Interaction(ctx.p() + wi*(2*sceneRadius), &mediumInterface));
    pub fn sample_li(
        &self,
        ctx: &LightSampleContext,
        u: Point2f,
        lambda: &SampledWavelengths,
        allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        let _p = ProfilePhase::new(Prof::LightSample);
        // pbrt-v4 (lights.h:581-583): MIS-driven path tracers ask for
        // the "incomplete PDF" mode which samples from the compensated
        // distribution (peaks only). Bistro / sanmiguel / etc. with a
        // sun in the env map relied on this — sampling the regular
        // distribution over a near-constant sky gave systematic +35%
        // bias because every bright sun sample also paid the dim-sky
        // PDF.
        let (uv, map_pdf) = if allow_incomplete_pdf {
            self.compensated_distribution.sample_continuous(&u)
        } else {
            self.distribution.sample_continuous(&u)
        };
        if map_pdf <= 0.0 {
            return None;
        }
        let w_light = equal_area_square_to_sphere(&uv);
        let wi = self.base.render_from_light.transform_vector(&w_light);
        let pdf = map_pdf / (4.0 * PI);
        if pdf <= 0.0 {
            return None;
        }
        let p = ctx.p + wi * (2.0 * self.radius());
        let p_light = Interaction::from_light_sample(&p, 0.0, &self.base.medium_interface);
        Some(LightLiSample::new(
            self.image_le(uv, lambda),
            wi,
            pdf,
            p_light,
        ))
    }

    // v4 lights.cpp:1042-1052: (allowIncompletePDF ? compensated :
    // distribution).PDF / (4 Pi).
    pub fn pdf_li(
        &self,
        _ctx: &LightSampleContext,
        wi: Vector3f,
        allow_incomplete_pdf: bool,
    ) -> Float {
        let _p = ProfilePhase::new(Prof::LightPdf);
        let w = self.base.world_to_light().transform_vector(&wi);
        let uv = equal_area_sphere_to_square(&w);
        let dist = if allow_incomplete_pdf {
            &self.compensated_distribution
        } else {
            &self.distribution
        };
        dist.pdf(&uv) / (4.0 * PI)
    }

    // v4 lights.cpp:1073-1095.
    pub fn sample_le(
        &self,
        u1: Point2f,
        u2: Point2f,
        lambda: &SampledWavelengths,
        time: Float,
    ) -> Option<LightLeSample> {
        let _p = ProfilePhase::new(Prof::LightSample);
        let (uv, map_pdf) = self.distribution.sample_continuous(&u1);
        if map_pdf <= 0.0 {
            return None;
        }
        let w_light = equal_area_square_to_sphere(&uv);
        let d = -self.base.render_from_light.transform_vector(&w_light);
        let (v1, v2) = coordinate_system(&-d);
        let cd = concentric_sample_disk(&u2);
        let center = *self.scene_center.read().unwrap();
        let radius = *self.scene_radius.read().unwrap();
        let p_disk = center + radius * (cd.x * v1 + cd.y * v2);
        let o = p_disk + radius * -d;
        let ray = Ray::new(&o, &d, Float::INFINITY, time);
        let pdf_dir = map_pdf / (4.0 * PI);
        let pdf_pos = 1.0 / (PI * radius * radius);
        Some(LightLeSample::new(
            self.image_le(uv, lambda),
            ray,
            pdf_pos,
            pdf_dir,
        ))
    }

    // v4 lights.cpp:1097-1102.
    pub fn pdf_le_ray(&self, ray: &Ray) -> (Float, Float) {
        let _ = ProfilePhase::new(Prof::LightPdf);
        let d = -self.base.world_to_light().transform_vector(&ray.d);
        let uv = equal_area_sphere_to_square(&d);
        let map_pdf = Float::max(0.0, self.distribution.pdf(&uv));
        let pdf_dir = map_pdf / (4.0 * PI);
        let r = self.radius();
        let pdf_pos = 1.0 / (PI * r * r);
        (pdf_pos, pdf_dir)
    }

    pub fn bounds(&self) -> Option<LightBounds> {
        None
    }

    fn is_infinite(&self) -> bool {
        true
    }

    fn get_light_flags(&self) -> u32 {
        self.base.flags
    }
}

unsafe impl Send for ImageInfiniteLight {}
unsafe impl Sync for ImageInfiniteLight {}
