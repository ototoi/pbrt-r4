use crate::base::light::*;
use crate::base::lightsampler::LightSampleContext;
use crate::base::shape::{Shape, ShapeSampleContext};
use crate::interaction::*;
use crate::lights::*;
use crate::media::*;
use crate::options::*;
use crate::paramdict::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::image::*;
use crate::util::imageio::*;
use crate::util::sampling::*;
use crate::util::spectrum::rgb_to_spectrum::RGBColorSpace;
use crate::util::spectrum::*;
use crate::util::transform::*;

use std::path::Path;
use std::sync::Arc;

#[derive(Clone)]
pub struct DiffuseAreaLight {
    base: LightBase,
    // v4 `const DenselySampledSpectrum *Lemit`.
    l_emit: Spectrum,
    // v4 `Float scale` (lights.h:477) -- single radiance multiplier.
    scale: Float,
    // v4 `Image image`.
    image: Option<Image>,
    // v4 `Shape shape`.
    shape: Arc<Shape>,
    // v4 `const RGBColorSpace *colorSpace`.
    color_space: &'static RGBColorSpace,
    // v4 `bool twoSided`.
    two_sided: bool,
    // v4 `Float area = shape.Area();` cache.
    area: Float,
}

impl DiffuseAreaLight {
    pub fn new(
        light_to_world: &Transform,
        medium_interface: &MediumInterface,
        light_type: LightType,
        le: Spectrum,
        scale: Float,
        image: Option<Image>,
        color_space: &'static RGBColorSpace,
        shape: &Arc<Shape>,
        two_sided: bool,
    ) -> Self {
        let shape = shape.clone();
        let area = shape.area();
        let base = LightBase::new(light_type as u32, light_to_world, medium_interface);
        DiffuseAreaLight {
            base,
            l_emit: le,
            scale,
            image,
            shape,
            color_space,
            two_sided,
            area,
        }
    }

    fn sampled_emission(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.l_emit.sample(lambda) * self.scale
    }

    // v4 lights.cpp:823-841: cosine-weighted hemisphere sampling,
    // optionally two-sided (split probability 1/2 between hemispheres).
    fn sample_wo(&self, _u1: &Point2f, u2: &Point2f) -> (Vector3f, Float) {
        if self.two_sided {
            let mut u = *u2;
            let mut w;
            if u[0] < 0.5 {
                u.x = Float::min(u[0] * 2.0, ONE_MINUS_EPSILON);
                w = cosine_sample_hemisphere(&u);
            } else {
                u.x = Float::min((u[0] - 0.5) * 2.0, ONE_MINUS_EPSILON);
                w = cosine_sample_hemisphere(&u);
                w.z *= -1.0;
            };
            let pdf = 0.5 * cosine_hemisphere_pdf(Float::abs(w.z));
            (w, pdf)
        } else {
            let w = cosine_sample_hemisphere(u2);
            let pdf = cosine_hemisphere_pdf(w.z);
            (w, pdf)
        }
    }

    pub fn create(
        light2world: &Transform,
        medium: &Option<Arc<Medium>>,
        params: &ParameterDictionary,
        shape: &Arc<Shape>,
    ) -> Result<Self, PbrtError> {
        // v4 lights.cpp:868-940 `DiffuseAreaLight::Create`. "scale" is a Float
        // (not a Spectrum) in v4. r4 honors that signature change here.
        let color_space = params.color_space();
        let white = Spectrum::from(color_space.illuminant.to_dense());
        let l = params.get_one_spectrum_typed("L", &white, SpectrumType::Illuminant);
        // v4 lights.cpp:909-910: scale /= SpectrumToPhotometric(L). Together
        // with the v4-aligned PixelSensor (`src/film/pixel_sensor.rs`) this
        // makes emission units 1 nit per default scale=1. SpectrumToPhotometric
        // is ∫Y(λ)·L(λ) dλ, computed as `spectrum_to_photometric` in r4.
        let photometric = spectrum_to_photometric(&l);
        let mut scale =
            params.get_one_float("scale", 1.0) / if photometric > 0.0 { photometric } else { 1.0 };
        let mi = MediumInterface::from(medium);
        let two_sided = params.get_one_bool("twosided", false);
        let filename = params.get_one_string("filename", "");
        let (image, color_space) = if filename.is_empty() {
            (None, color_space)
        } else {
            if l != Spectrum::one() {
                return Err(PbrtError::error(
                    "Both \"L\" and \"filename\" specified for DiffuseAreaLight.",
                ));
            }
            let texname = Path::new(&filename);
            let (texels, resolution, cs) = read_image_exr_with_metadata(texname)?;
            (
                Some(Image::try_with_color_space(resolution, texels, cs)?),
                cs,
            )
        };
        let mut n_samples = params.get_one_int("nsamples", 1) as u32;
        {
            let options = PbrtOptions::get();
            if options.quick_render {
                n_samples = (n_samples / 4).max(1);
            }
        }
        let light_type = if shape.has_constant_zero_alpha_mask() {
            LightType::DeltaPosition
        } else {
            LightType::Area
        };
        Ok(DiffuseAreaLight::new(
            light2world,
            &mi,
            light_type,
            l,
            scale,
            image,
            color_space,
            shape,
            two_sided,
        ))
    }
}

impl DiffuseAreaLight {
    pub fn light_type(&self) -> LightType {
        match self.base.flags {
            flags if flags == LightType::DeltaPosition as u32 => LightType::DeltaPosition,
            _ => LightType::Area,
        }
    }

    // v4 lights.cpp:769-786:
    //   if image: average image emission; else L = Lemit->Sample(lambda) * scale;
    //   return Pi * (twoSided ? 2 : 1) * area * L;
    pub fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        if let Some(image) = &self.image {
            let mut l = SampledSpectrum::zero();
            for rgb in image.texels() {
                l += self
                    .color_space
                    .illuminant_to_sampled_spectrum(rgb.to_rgb(), lambda);
            }
            l *= self.scale / (image.width() as Float * image.height() as Float);
            return PI * (if self.two_sided { 2.0 } else { 1.0 }) * self.area * l;
        }
        let n = if self.two_sided { 2.0 } else { 1.0 };
        self.sampled_emission(lambda) * (n * self.area * PI)
    }

    pub fn l(
        &self,
        _p: Point3f,
        n: Normal3f,
        uv: Point2f,
        w: Vector3f,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        if !self.two_sided && Vector3f::dot(&n, &w) < 0.0 {
            return SampledSpectrum::zero();
        }
        if let Some(image) = &self.image {
            let rgb = image.lookup_nearest(&uv);
            self.color_space
                .illuminant_to_sampled_spectrum(rgb.to_rgb(), lambda)
                * self.scale
        } else {
            self.sampled_emission(lambda)
        }
    }

    // v4 lights.cpp:739-761:
    //   ShapeSampleContext shapeCtx(ctx.pi, ctx.n, ctx.ns, 0);
    //   ss = shape.Sample(shapeCtx, u);
    //   if !ss || ss.pdf==0 || (ss.intr.p - ctx.p) == 0 return {};
    //   ss.intr.mediumInterface = &mediumInterface;
    //   ... alpha gate ...
    //   wi = Normalize(ss.intr.p - ctx.p);
    //   Le = L(ss.intr.p, ss.intr.n, ss.intr.uv, -wi, lambda);
    //   if !Le return {};
    //   return LightLiSample(Le, wi, ss.pdf, ss.intr);
    pub fn sample_li(
        &self,
        ctx: &LightSampleContext,
        u: Point2f,
        lambda: &SampledWavelengths,
        _allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        // v4 builds `ShapeSampleContext(ctx.pi, ctx.n, ctx.ns, 0)`.
        let shape_ctx = ShapeSampleContext {
            p: ctx.p,
            n: ctx.n,
            ns: ctx.ns,
            time: 0.0,
        };
        let shape = self.shape.as_ref();
        let (mut intr, pdf) = shape.sample_from(&shape_ctx, &u)?;
        intr.set_medium_interface(&self.base.medium_interface);
        if self.light_type() == LightType::Area
            && shape.alpha(&intr).is_some_and(|alpha| alpha <= 0.0)
        {
            return None;
        }
        if pdf <= 0.0 || (intr.get_p() - ctx.p).length_squared() <= 0.0 {
            return None;
        }
        let wi = (intr.get_p() - ctx.p).normalize();
        let le = self.l(intr.get_p(), intr.get_n(), intr.get_uv(), -wi, lambda);
        if le.is_black() {
            return None;
        }
        Some(LightLiSample::new(le, wi, pdf, intr))
    }

    // v4 lights.cpp:763-767:
    //   ShapeSampleContext shapeCtx(ctx.pi, ctx.n, ctx.ns, 0);
    //   return shape.PDF(shapeCtx, wi);
    pub fn pdf_li(
        &self,
        ctx: &LightSampleContext,
        wi: Vector3f,
        _allow_incomplete_pdf: bool,
    ) -> Float {
        // v4 builds `ShapeSampleContext(ctx.pi, ctx.n, ctx.ns, 0)`
        // (lights.cpp:765); the spherical-triangle cosine warp in
        // `Triangle::pdf_from` reads `shading.n` from a
        // SurfaceInteraction.
        let shape_ctx = ShapeSampleContext {
            p: ctx.p,
            n: ctx.n,
            ns: ctx.ns,
            time: 0.0,
        };
        self.shape.as_ref().pdf_from(&shape_ctx, &wi)
    }

    // v4 lights.cpp:809-851:
    //   ss = shape.Sample(u1); if !ss return {};
    //   ss.intr.time = time; ss.intr.mediumInterface = &mediumInterface;
    //   ... alpha gate ...
    //   sample cosine-weighted w (or two-sided); pdfDir;
    //   if pdfDir == 0 return {};
    //   Frame nFrame = Frame::FromZ(intr.n); w = nFrame.FromLocal(w);
    //   Le = L(intr.p, intr.n, intr.uv, w, lambda);
    //   return LightLeSample(Le, intr.SpawnRay(w), intr, ss.pdf, pdfDir);
    pub fn sample_le(
        &self,
        u1: Point2f,
        u2: Point2f,
        lambda: &SampledWavelengths,
        time: Float,
    ) -> Option<LightLeSample> {
        let shape = self.shape.as_ref();
        let (mut intr, pdf_pos) = shape.sample(&u1)?;
        intr.set_time(time);
        intr.set_medium_interface(&self.base.medium_interface);
        if self.light_type() == LightType::Area
            && shape.alpha(&intr).is_some_and(|alpha| alpha <= 0.0)
        {
            return None;
        }
        let n = intr.get_n();
        let uv = intr.get_uv();

        let (w_local, pdf_dir) = self.sample_wo(&u1, &u2);
        if pdf_dir == 0.0 {
            return None;
        }
        // v4 `Frame::FromZ(intr.n).FromLocal(w)`.
        let (v1, v2) = coordinate_system(&n);
        let w = w_local.x * v1 + w_local.y * v2 + w_local.z * n;
        let mut ray = intr.spawn_ray(&w);
        ray.time = time;
        let le = self.l(intr.get_p(), n, uv, w, lambda);
        Some(LightLeSample::with_intr(le, ray, intr, pdf_pos, pdf_dir))
    }

    // v4 lights.cpp:853-859:
    //   *pdfPos = shape.PDF(intr);
    //   *pdfDir = twoSided ? CosineHemispherePDF(AbsDot(n, w))/2
    //                      : CosineHemispherePDF(Dot(n, w));
    pub fn pdf_le_interaction(&self, intr: &Interaction, w: Vector3f) -> (Float, Float) {
        let shape = self.shape.as_ref();
        let n = intr.get_n();
        let pdf_pos = shape.pdf(intr);
        let pdf_dir = if self.two_sided {
            0.5 * cosine_hemisphere_pdf(Float::abs(Vector3f::dot(&n, &w)))
        } else {
            cosine_hemisphere_pdf(Vector3f::dot(&n, &w)).max(0.0)
        };
        (pdf_pos, pdf_dir)
    }

    pub fn pdf_le_ray(&self, _ray: &Ray) -> (Float, Float) {
        (0.0, 0.0)
    }

    pub fn bounds(&self) -> Option<LightBounds> {
        let phi = if let Some(image) = &self.image {
            let mut phi = 0.0;
            for rgb in image.texels() {
                let rgb = rgb.to_rgb();
                phi += rgb[0] + rgb[1] + rgb[2];
            }
            phi /= 3.0 * image.width() as Float * image.height() as Float;
            phi * self.scale * self.area * PI
        } else {
            self.l_emit.max_value() * self.scale * self.area * PI
        };
        let nb = self.shape.normal_bounds();
        Some(LightBounds::new(
            self.shape.world_bound(),
            nb.w,
            phi,
            nb.cos_theta,
            Float::cos(PI / 2.0),
            self.two_sided,
        ))
    }
}

unsafe impl Sync for DiffuseAreaLight {}
unsafe impl Send for DiffuseAreaLight {}
