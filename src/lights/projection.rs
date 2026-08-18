use crate::base::light::*;
use crate::base::lightsampler::LightSampleContext;
use crate::interaction::*;
use crate::lights::*;
use crate::media::*;
use crate::paramdict::*;
use crate::textures::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::image::*;
use crate::util::imageio::read_image_exr_with_metadata;
use crate::util::imageio::*;
use crate::util::sampling::PiecewiseConstant2D;
use crate::util::sampling::*;
use crate::util::scattering::cos_theta;
use crate::util::spectrum::rgb_to_spectrum::RGBColorSpace;
use crate::util::spectrum::*;
use crate::util::transform::*;

use std::path::Path;
use std::sync::Arc;

#[derive(Clone)]
pub struct ProjectionLight {
    base: LightBase,
    image: Image,
    // v4 `Float scale` (lights.h:344).
    scale: Float,
    // v4 `Transform screenFromLight, lightFromScreen`.
    screen_from_light: Transform,
    light_from_screen: Transform,
    // v4 `Float hither` (1e-3 in v4 ctor).
    hither: Float,
    #[allow(dead_code)]
    yon: Float,
    // v4 `Bounds2f screenBounds`.
    screen_bounds: Bounds2f,
    // v4 `Float A` -- projected image area factor.
    a: Float,
    distribution: PiecewiseConstant2D,
}

impl ProjectionLight {
    pub fn new(
        render_from_light: &Transform,
        medium_interface: &MediumInterface,
        image: Image,
        scale: Float,
        fov: Float,
    ) -> Self {
        let base = LightBase::new(
            LightType::DeltaPosition as u32,
            render_from_light,
            medium_interface,
        );
        let resolution = image.resolution();
        let aspect = resolution.x as Float / resolution.y as Float;
        let screen_bounds = if aspect > 1.0 {
            Bounds2f::new(&Point2f::new(-aspect, -1.0), &Point2f::new(aspect, 1.0))
        } else {
            Bounds2f::new(
                &Point2f::new(-1.0, -1.0 / aspect),
                &Point2f::new(1.0, 1.0 / aspect),
            )
        };
        let hither = 1e-3;
        let yon = 1e30;
        let screen_from_light = Transform::perspective(fov, hither, yon);
        let light_from_screen = screen_from_light.inverse();
        let opposite = (0.5 * fov.to_radians()).tan();
        let a = 4.0 * opposite * opposite * aspect.max(1.0 / aspect);
        let distribution = make_distribution(&image);
        ProjectionLight {
            base,
            image,
            scale,
            screen_from_light,
            light_from_screen,
            hither,
            yon,
            screen_bounds,
            a,
            distribution,
        }
    }

    fn render_from_image(&self, uv: &Point2f) -> Vector3f {
        let alpha = -std::f64::consts::FRAC_PI_2 as Float + uv.x * PI;
        let beta = -std::f64::consts::FRAC_PI_2 as Float + uv.y * PI;
        let x = alpha.tan();
        let y = beta.tan();
        let w = Vector3f::new(x, y, 1.0).normalize();
        self.base.render_from_light.transform_vector(&w)
    }

    // pbrt-v4 `SampledSpectrum I(Vector3f w, lambda)` (lights.cpp:343-360):
    //   if (w.z < hither) return 0;
    //   Point3f ps = screenFromLight(Point3f(w));
    //   if (!Inside(Point2f(ps.x, ps.y), screenBounds)) return 0;
    //   Point2f uv = screenBounds.Offset(...);
    //   ... LookupNearestChannel + RGBIlluminantSpectrum::Sample
    fn intensity(&self, w: Vector3f, lambda: &SampledWavelengths) -> SampledSpectrum {
        if w.z < self.hither {
            return SampledSpectrum::zero();
        }
        let ps = self
            .screen_from_light
            .transform_point(&Point3f::new(w.x, w.y, w.z));
        if !self.screen_bounds.inside(&Point2f::new(ps.x, ps.y)) {
            return SampledSpectrum::zero();
        }
        let uv = self.screen_bounds.offset(&Point2f::new(ps.x, ps.y));
        let rgb = self.image.lookup_nearest(&uv);
        let modulation = self
            .image
            .color_space()
            .illuminant_to_sampled_spectrum(rgb.to_rgb(), lambda);
        self.scale * modulation
    }
}

impl ProjectionLight {
    pub fn light_type(&self) -> LightType {
        LightType::DeltaPosition
    }

    // v4 lights.cpp:320-330:
    //   Point3f p = renderFromLight(Point3f(0,0,0));
    //   Vector3f wi = Normalize(p - ctx.p());
    //   Vector3f wl = renderFromLight.ApplyInverse(-wi);
    //   SampledSpectrum Li = I(wl, lambda) / DistanceSquared(p, ctx.p());
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
        let wl = self.base.world_to_light().transform_vector(&-wi);
        let li = self.intensity(wl, lambda) / Point3f::distance_squared(&p, &ctx.p);
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

    // v4 lights.cpp:362-382: integrate `I` over the image with the
    // change-of-variables Pow<3>(cosTheta(w)).
    pub fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut sum = SampledSpectrum::zero();
        let res = self.image.resolution();
        for y in 0..res.y {
            for x in 0..res.x {
                let uv = Point2f::new(
                    (x as Float + 0.5) / res.x as Float,
                    (y as Float + 0.5) / res.y as Float,
                );
                let w = self.render_from_image(&uv);
                let dwd_a = w.z * w.z * w.z;
                let rgb = self.image.lookup_nearest(&uv);
                let modulation = self
                    .image
                    .color_space()
                    .illuminant_to_sampled_spectrum(rgb.to_rgb(), lambda);
                sum += modulation * dwd_a;
            }
        }
        self.scale * self.a * sum / (res.x as Float * res.y as Float)
    }

    pub fn sample_le(
        &self,
        u1: Point2f,
        _u2: Point2f,
        lambda: &SampledWavelengths,
        time: Float,
    ) -> Option<LightLeSample> {
        let medium = self.base.medium_interface.get_outside();
        let (ps, pdf) = self.distribution.sample_continuous(&u1);
        if pdf <= 0.0 {
            return None;
        }
        let p = self
            .light_from_screen
            .transform_point(&Point3f::new(ps.x, ps.y, 0.0));
        let w = Vector3f::new(p.x, p.y, p.z);
        let cos_theta = w.normalize().z;
        if cos_theta <= 0.0 {
            return None;
        }
        let pdf_dir = pdf * self.screen_bounds.area() / (self.a * cos_theta.powi(3));
        let p = self
            .base
            .render_from_light
            .transform_point(&Point3f::zero());
        let ray = Ray::from((&p, &w.normalize(), Float::INFINITY, time, &medium));
        let uv = self.screen_bounds.offset(&Point2f::new(ps.x, ps.y));
        let rgb = self.image.lookup_nearest(&uv);
        let l = self.scale
            * self
                .image
                .color_space()
                .illuminant_to_sampled_spectrum(rgb.to_rgb(), lambda);
        Some(LightLeSample::new(l, ray, 1.0, pdf_dir))
    }

    // v4 lights.cpp:428-446.
    pub fn pdf_le_ray(&self, ray: &Ray) -> (Float, Float) {
        let w = self
            .base
            .world_to_light()
            .transform_vector(&ray.d)
            .normalize();
        if w.z < self.hither {
            return (0.0, 0.0);
        }
        let ps = self
            .screen_from_light
            .transform_point(&Point3f::new(w.x, w.y, w.z));
        if !self.screen_bounds.inside(&Point2f::new(ps.x, ps.y)) {
            return (0.0, 0.0);
        }
        let pdf_dir = self.distribution.pdf(&Point2f::new(ps.x, ps.y)) * self.screen_bounds.area()
            / (self.a * w.z.powi(3));
        (0.0, pdf_dir)
    }

    // v4 lights.cpp:384-399: phi from image + projected cone area bound.
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
        let res = self.image.resolution();
        let mut sum = 0.0;
        for y in 0..res.y {
            for x in 0..res.x {
                let uv = Point2f::new(
                    (x as Float + 0.5) / res.x as Float,
                    (y as Float + 0.5) / res.y as Float,
                );
                let rgb = self.image.lookup_nearest(&uv).to_rgb();
                sum += rgb[0].max(rgb[1]).max(rgb[2]);
            }
        }
        let phi = self.scale * sum / (res.x as Float * res.y as Float);
        Some(LightBounds::new(
            Bounds3f::new(&p, &p),
            w,
            phi,
            Float::cos(0.0),
            self.a,
            false,
        ))
    }

    fn get_light_flags(&self) -> u32 {
        self.base.flags
    }
}

unsafe impl Sync for ProjectionLight {}
unsafe impl Send for ProjectionLight {}

fn make_image(path: &str) -> Result<Image, PbrtError> {
    if !path.is_empty() {
        if path.ends_with(".exr") || path.ends_with(".EXR") {
            let p = Path::new(path);
            let (mut texels, resolution, color_space) = read_image_exr_with_metadata(p)?;
            for texel in texels.iter_mut() {
                let cc = texel.to_rgb();
                let cc = cc.iter().map(|x| x.max(0.0)).collect::<Vec<Float>>();
                *texel = RGBSpectrum::from(cc);
            }
            return Ok(Image::try_with_color_space(
                resolution,
                texels,
                color_space,
            )?);
        }
        let (mut texels, resolution) = read_image(path)?;
        let total = (resolution.x * resolution.y) as usize;
        for i in 0..total {
            let cc = texels[i].to_rgb();
            let cc = cc.iter().map(|x| x.max(0.0)).collect::<Vec<Float>>();
            texels[i] = RGBSpectrum::from(cc);
        }
        return Ok(Image::new(resolution, texels));
    } else {
        let c = RGBSpectrum::one();
        return Ok(Image::new(Point2i::new(1, 1), vec![c]));
    }
}

fn make_distribution(image: &Image) -> PiecewiseConstant2D {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut data = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let uv = Point2f::new(
                (x as Float + 0.5) / width as Float,
                (y as Float + 0.5) / height as Float,
            );
            let rgb = image.lookup_nearest(&uv).to_rgb();
            data[y * width + x] = ((rgb[0] + rgb[1] + rgb[2]) / 3.0).max(0.0);
        }
    }
    PiecewiseConstant2D::new(&data, width, height)
}

impl ProjectionLight {
    pub fn create(
        render_from_light: &Transform,
        medium_interface: &MediumInterface,
        params: &ParameterDictionary,
    ) -> Result<Self, PbrtError> {
        // v4 lights.cpp `ProjectionLight::Create`. "scale" is a Float; v4
        // normalizes via `sc /= SpectrumToPhotometric(I)`.
        let intensity =
            params.get_one_spectrum_typed("I", &Spectrum::from(1.0), SpectrumType::Illuminant);
        let photometric = spectrum_to_photometric(&intensity);
        let sc =
            params.get_one_float("scale", 1.0) / if photometric > 0.0 { photometric } else { 1.0 };
        let fov = params.get_one_float("fov", 45.0);
        let texname = params.get_one_filename("mapname", "");
        let image = make_image(&texname)?;
        let light = ProjectionLight::new(render_from_light, medium_interface, image, sc, fov);
        Ok(light)
    }
}
