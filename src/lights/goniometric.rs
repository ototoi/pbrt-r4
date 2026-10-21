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
use crate::util::imageio::*;
use crate::util::sampling::PiecewiseConstant2D;
use crate::util::sampling::*;
use crate::util::spectrum::*;
use crate::util::transform::*;

use log::warn;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone)]
pub struct GoniometricLight {
    base: LightBase,
    intensity: Spectrum,
    scale: Float,
    image: Image,
    distribution: PiecewiseConstant2D,
}

impl GoniometricLight {
    pub fn new(
        render_from_light: &Transform,
        medium_interface: &MediumInterface,
        intensity: Spectrum,
        scale: Float,
        image: Image,
    ) -> Self {
        let distribution = make_distribution(&image);
        Self {
            base: LightBase::new(
                LightType::DeltaPosition as u32,
                render_from_light,
                medium_interface,
            ),
            intensity,
            scale,
            image,
            distribution,
        }
    }

    /// pbrt-v4 `GoniometricLight::I` (lights.h:393-396). `w` is in light
    /// space and the image holds the luminance channel `Y`.
    fn i(&self, w: Vector3f, lambda: &SampledWavelengths) -> SampledSpectrum {
        let uv = equal_area_sphere_to_square(&w.normalize());
        self.scale * self.intensity.sample(lambda) * self.image.lookup_nearest(&uv).to_rgb()[0]
    }
}

impl GoniometricLight {
    pub fn light_type(&self) -> LightType {
        LightType::DeltaPosition
    }

    // v4 lights.cpp:572-579.
    pub fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let res = self.image.resolution();
        self.scale * self.intensity.sample(lambda) * 4.0 * PI * sum_y(&self.image)
            / (res.x as Float * res.y as Float)
    }

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
        let l = self.i(wl, lambda) / Point3f::distance_squared(&p, &ctx.p);
        Some(LightLiSample::new(
            l,
            wi,
            1.0,
            Interaction::from_light_sample(&p, 0.0, &self.base.medium_interface),
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

    pub fn sample_le(
        &self,
        u1: Point2f,
        _u2: Point2f,
        lambda: &SampledWavelengths,
        time: Float,
    ) -> Option<LightLeSample> {
        let medium = self.base.medium_interface.get_outside();
        let (uv, pdf) = self.distribution.sample_continuous(&u1);
        if pdf <= 0.0 {
            return None;
        }
        let w_light = equal_area_square_to_sphere(&uv);
        let d = self.base.render_from_light.transform_vector(&w_light);
        let o = self
            .base
            .render_from_light
            .transform_point(&Point3f::zero());
        let ray = Ray::from((&o, &d, Float::INFINITY, time, &medium));
        let pdf_dir = pdf / (4.0 * PI);
        Some(LightLeSample::new(
            self.i(w_light, lambda),
            ray,
            1.0,
            pdf_dir,
        ))
    }

    pub fn pdf_le_ray(&self, ray: &Ray) -> (Float, Float) {
        let w = self
            .base
            .world_to_light()
            .transform_vector(&ray.d)
            .normalize();
        let uv = equal_area_sphere_to_square(&w);
        (0.0, self.distribution.pdf(&uv) / (4.0 * PI))
    }

    pub fn bounds(&self) -> Option<LightBounds> {
        let p = self
            .base
            .render_from_light
            .transform_point(&Point3f::zero());
        // v4 lights.cpp:581-593: bound it as an isotropic point light.
        let res = self.image.resolution();
        let phi = self.scale * self.intensity.max_value() * 4.0 * PI * sum_y(&self.image)
            / (res.x as Float * res.y as Float);
        Some(LightBounds::new(
            Bounds3f::new(&p, &p),
            Vector3f::new(0.0, 0.0, 1.0),
            phi,
            Float::cos(PI),
            Float::cos(PI / 2.0),
            false,
        ))
    }

    fn get_light_flags(&self) -> u32 {
        self.base.flags
    }
}

unsafe impl Sync for GoniometricLight {}
unsafe impl Send for GoniometricLight {}

/// Loads the goniometric diagram as a luminance image. Like pbrt-v4
/// (lights.cpp:625-640), RGB images are reduced to the average of their
/// channels; every channel of the returned texels holds that value.
fn make_image(path: &str) -> Result<Image, PbrtError> {
    if path.is_empty() {
        warn!("No \"filename\" parameter provided for goniometric light.");
        return Ok(Image::new(Point2i::new(1, 1), vec![RGBSpectrum::one()]));
    }
    // pbrt-v4 `Image::Read` decodes 8- and 16-bit images as sRGB; float
    // formats ignore the encoding.
    let (texels, resolution) = if path.ends_with(".exr") || path.ends_with(".EXR") {
        let (texels, resolution, _) = read_image_exr_with_metadata(Path::new(path))?;
        (texels, resolution)
    } else {
        read_image_with_encoding(path, ColorEncoding::SRgb)?
    };
    if resolution.x != resolution.y {
        return Err(PbrtError::error(&format!(
            "{}: image resolution ({}, {}) is non-square. It's unlikely this is an equal-area environment map.",
            path, resolution.x, resolution.y
        )));
    }
    let texels = texels
        .iter()
        .map(|texel| {
            let rgb = texel.to_rgb();
            let y = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
            RGBSpectrum::from(vec![y, y, y])
        })
        .collect();
    Ok(Image::new(resolution, texels))
}

/// Sum of the luminance channel over all pixels.
fn sum_y(image: &Image) -> Float {
    let res = image.resolution();
    let mut sum = 0.0;
    for y in 0..res.y {
        for x in 0..res.x {
            let uv = Point2f::new(
                (x as Float + 0.5) / res.x as Float,
                (y as Float + 0.5) / res.y as Float,
            );
            sum += image.lookup_nearest(&uv).to_rgb()[0];
        }
    }
    sum
}

fn make_distribution(image: &Image) -> PiecewiseConstant2D {
    let res = image.resolution();
    let mut data = vec![0.0; (res.x * res.y) as usize];
    for y in 0..res.y {
        for x in 0..res.x {
            let uv = Point2f::new(
                (x as Float + 0.5) / res.x as Float,
                (y as Float + 0.5) / res.y as Float,
            );
            data[(y * res.x + x) as usize] = image.lookup_nearest(&uv).to_rgb()[0];
        }
    }
    PiecewiseConstant2D::new(&data, res.x as usize, res.y as usize)
}

impl GoniometricLight {
    pub fn create(
        light2world: &Transform,
        medium: &Option<Arc<Medium>>,
        params: &ParameterDictionary,
    ) -> Result<Self, PbrtError> {
        // v4 lights.cpp `GoniometricLight::Create`. The legacy pbrt-v3
        // "mapname" is accepted when "filename" is absent.
        let white = Spectrum::from(params.color_space().illuminant.to_dense());
        let intensity = params.get_one_spectrum_typed("I", &white, SpectrumType::Illuminant);
        let mut scale = params.get_one_float("scale", 1.0);
        let texmap = {
            let filename = params.get_one_filename("filename", "");
            if !filename.is_empty() {
                filename
            } else {
                params.get_one_filename("mapname", "")
            }
        };
        let image = make_image(&texmap)?;

        scale /= spectrum_to_photometric(&intensity);
        let phi_v = params.get_one_float("power", -1.0);
        if phi_v > 0.0 {
            let res = image.resolution();
            let k_e = 4.0 * PI * sum_y(&image) / (res.x as Float * res.y as Float);
            scale *= phi_v / k_e;
        }

        // v4 swaps the light-space y and z axes so that +y is "up" in the
        // goniometric diagram.
        let swap_yz = Transform::from([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let render_from_light = light2world.clone() * swap_yz;
        let mi = MediumInterface::from(medium);
        Ok(GoniometricLight::new(
            &render_from_light,
            &mi,
            intensity,
            scale,
            image,
        ))
    }
}
