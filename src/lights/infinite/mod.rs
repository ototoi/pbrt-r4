pub mod image_infinite;
pub mod portal_image_infinite;
pub mod uniform_infinite;

pub use image_infinite::*;
pub use portal_image_infinite::*;
pub use uniform_infinite::*;

use crate::base::light::*;
use crate::base::lightsampler::LightSampleContext;
use crate::lights::*;
use crate::media::*;
use crate::paramdict::*;
use crate::textures::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::{Bounds3f, Ray};
use crate::util::imageio::read_image_exr::read_image_exr_with_metadata;
use crate::util::imageio::*;
use crate::util::sampling::*;
use crate::util::spectrum::rgb_to_spectrum::{RGBColorSpace, SRGB};
use crate::util::spectrum::*;
use crate::util::transform::*;

use std::path::Path;

pub enum InfiniteLight {
    Uniform(Box<UniformInfiniteLight>),
    Image(Box<ImageInfiniteLight>),
    PortalImage(Box<PortalImageInfiniteLight>),
}

impl InfiniteLight {
    pub fn create(
        light2world: &Transform,
        params: &ParameterDictionary,
        render_from_world: &Transform,
    ) -> Result<Self, PbrtError> {
        // pbrt-v4 lights.cpp:1527-1665 ("infinite" branch). Three sub-cases:
        //   (a) no L, no filename, no portal -> UniformInfiniteLight scaled by
        //       the default sRGB illuminant.
        //   (b) L + no portal               -> UniformInfiniteLight(L).
        //   (c) (filename) or (L + portal)  -> ImageInfiniteLight or
        //       PortalImageInfiniteLight. When L is provided without a
        //       filename, v4 synthesizes a 1x1 RGB image from L so that the
        //       image-path can still drive a portal restriction.
        let white = Spectrum::from(params.color_space().illuminant.to_dense());
        let l_param = params.get_one_spectrum_typed("L", &white, SpectrumType::Illuminant);
        let raw_scale = params.get_one_float("scale", 1.0);
        let e_v = params.get_one_float("illuminance", -1.0);
        let texmap = {
            let fn_v4 = params.get_one_filename("filename", "");
            if !fn_v4.is_empty() {
                fn_v4
            } else {
                params.get_one_filename("mapname", "")
            }
        };
        let portal_points: Vec<Point3f> = params
            .get_points_ref("portal")
            .map(|raw| {
                raw.chunks_exact(3)
                    .map(|c| render_from_world.transform_point(&Point3f::new(c[0], c[1], c[2])))
                    .collect()
            })
            .unwrap_or_default();
        let base = LightBase::new(
            LightType::Infinite as u32,
            light2world,
            &MediumInterface::new(),
        );

        let apply_illuminance = |sc: &mut Float| {
            if e_v > 0.0 {
                *sc *= e_v / (std::f64::consts::PI as Float);
            }
        };

        if texmap.is_empty() && portal_points.len() != 4 {
            let photometric = spectrum_to_photometric(&l_param);
            let mut sc = raw_scale / if photometric > 0.0 { photometric } else { 1.0 };
            apply_illuminance(&mut sc);
            return Ok(InfiniteLight::Uniform(Box::new(UniformInfiniteLight::new(
                &base.render_from_light,
                &base.medium_interface,
                l_param,
                sc,
            ))));
        }

        let cs_illuminant = Spectrum::from_rgb_illuminant(&[1.0, 1.0, 1.0]);
        let cs_photometric = spectrum_to_photometric(&cs_illuminant);
        let mut sc = raw_scale
            / if cs_photometric > 0.0 {
                cs_photometric
            } else {
                1.0
            };
        apply_illuminance(&mut sc);

        let (lmap, image_color_space) = if texmap.is_empty() {
            let xyz = spectrum_to_xyz(&l_param);
            let rgb = xyz_to_rgb(&xyz);
            let texels: Vec<RGBSpectrum> = vec![RGBSpectrum::from(&rgb)];
            let mm = create_spectrum_mipmap(&Point2i::new(1, 1), &texels)?;
            (mm, &SRGB)
        } else {
            make_mipmap_with_colorspace(&texmap)?
        };

        if portal_points.len() == 4 {
            let portal = [
                portal_points[0],
                portal_points[1],
                portal_points[2],
                portal_points[3],
            ];
            return Ok(InfiniteLight::PortalImage(Box::new(
                PortalImageInfiniteLight::new(base, &lmap, sc, portal, image_color_space)?,
            )));
        }

        let (distribution, compensated_distribution) = make_distribution_pair(&lmap)?;
        Ok(InfiniteLight::Image(Box::new(ImageInfiniteLight::new(
            &base.render_from_light,
            &base.medium_interface,
            lmap,
            sc,
            distribution,
            compensated_distribution,
            image_color_space,
        ))))
    }

    pub fn light_type(&self) -> LightType {
        LightType::Infinite
    }

    pub fn preprocess(&self, scene_bounds: &Bounds3f) {
        match self {
            InfiniteLight::Uniform(l) => l.preprocess(scene_bounds),
            InfiniteLight::Image(l) => l.preprocess(scene_bounds),
            InfiniteLight::PortalImage(l) => l.preprocess(scene_bounds),
        }
    }

    pub fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            InfiniteLight::Uniform(l) => l.le(ray, lambda),
            InfiniteLight::Image(l) => l.le(ray, lambda),
            InfiniteLight::PortalImage(l) => l.le(ray, lambda),
        }
    }

    pub fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            InfiniteLight::Uniform(l) => l.phi(lambda),
            InfiniteLight::Image(l) => l.phi(lambda),
            InfiniteLight::PortalImage(l) => l.phi(lambda),
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
            InfiniteLight::Uniform(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
            InfiniteLight::Image(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
            InfiniteLight::PortalImage(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
        }
    }

    pub fn pdf_li(
        &self,
        ctx: &LightSampleContext,
        wi: Vector3f,
        allow_incomplete_pdf: bool,
    ) -> Float {
        match self {
            InfiniteLight::Uniform(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
            InfiniteLight::Image(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
            InfiniteLight::PortalImage(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
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
            InfiniteLight::Uniform(l) => l.sample_le(u1, u2, lambda, time),
            InfiniteLight::Image(l) => l.sample_le(u1, u2, lambda, time),
            InfiniteLight::PortalImage(l) => l.sample_le(u1, u2, lambda, time),
        }
    }

    pub fn pdf_le_ray(&self, ray: &Ray) -> (Float, Float) {
        match self {
            InfiniteLight::Uniform(l) => l.pdf_le_ray(ray),
            InfiniteLight::Image(l) => l.pdf_le_ray(ray),
            InfiniteLight::PortalImage(l) => l.pdf_le_ray(ray),
        }
    }

    pub fn bounds(&self) -> Option<LightBounds> {
        match self {
            InfiniteLight::Uniform(l) => l.bounds(),
            InfiniteLight::Image(l) => l.bounds(),
            InfiniteLight::PortalImage(l) => l.bounds(),
        }
    }
}

// ===========================================================================
// Helpers and factory functions
// ===========================================================================

/// Load `path` into a `MIPMap<RGBSpectrum>` and report the EXR's
/// declared colour space. Non-EXR or chromaticities-less files fall
/// back to sRGB (matches pbrt-v4 `ImageMetadata::GetColorSpace`).
fn make_mipmap_with_colorspace(
    path: &str,
) -> Result<(MIPMap<RGBSpectrum>, &'static RGBColorSpace), PbrtError> {
    if path.is_empty() {
        let texels: Vec<RGBSpectrum> = vec![RGBSpectrum::one()];
        let resolution = Point2i::new(1, 1);
        let mm = create_spectrum_mipmap(&resolution, texels.as_ref())?;
        return Ok((mm, &SRGB));
    }

    // pbrt-v4 reads chromaticities only for EXR. Other image formats
    // (PNG/JPG) carry sRGB by convention; r4 honours that by routing
    // them through the generic `read_image` path and assuming sRGB.
    let is_exr = path.to_ascii_lowercase().ends_with(".exr");
    let (mut texels, resolution, cs) = if is_exr {
        read_image_exr_with_metadata(Path::new(path))?
    } else {
        let (texels, resolution) = read_image(path)?;
        (texels, resolution, &SRGB)
    };
    let total = (resolution.x * resolution.y) as usize;
    for i in 0..total {
        let cc = texels[i].to_rgb();
        let cc = cc.iter().map(|x| x.max(0.0)).collect::<Vec<Float>>();
        texels[i] = RGBSpectrum::from(cc);
        assert!(texels[i].y() >= 0.0);
    }
    let mipmap = create_spectrum_mipmap(&resolution, texels.as_ref())?;
    Ok((mipmap, cs))
}

/// pbrt-v4 (lights.cpp:1015-1029) builds two distributions in one
/// pass: a regular per-pixel average distribution and a compensated
/// distribution that subtracts the mean from every pixel and clips at
/// zero. The latter focuses MIS-driven NEE samples on bright spots
/// (sun, fireballs, lamps in HDR env maps) and is selected when
/// `sample_li` is invoked with `allow_incomplete_pdf = true`.
fn make_distribution_pair(
    lmap: &MIPMap<RGBSpectrum>,
) -> Result<(Distribution2D, Distribution2D), PbrtError> {
    let width = lmap.width();
    let height = lmap.height();
    let mut img = vec![0.0; width * height];
    for v in 0..height {
        for u in 0..width {
            let rgb = lmap.texel(0, u as i32, v as i32).to_rgb();
            let avg = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
            img[v * width + u] = avg.max(0.0);
        }
    }
    let regular = Distribution2D::new(&img, width, height);

    // pbrt-v4 (lights.cpp:1023-1029):
    //   Float average = sum(d) / d.size();
    //   for each v in d: v = max(v - average, 0);
    //   if all zero, fill with 1 for uniform sampling.
    let total = (width * height) as Float;
    let mean = if total > 0.0 {
        img.iter().sum::<Float>() / total
    } else {
        0.0
    };
    let mut comp: Vec<Float> = img.iter().map(|v| (v - mean).max(0.0)).collect();
    if comp.iter().all(|&v| v == 0.0) {
        for v in comp.iter_mut() {
            *v = 1.0;
        }
    }
    let compensated = Distribution2D::new(&comp, width, height);

    Ok((regular, compensated))
}
