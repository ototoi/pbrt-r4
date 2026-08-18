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

    fn i(&self, w: Vector3f, lambda: &SampledWavelengths) -> SampledSpectrum {
        let wp = self.base.world_to_light().transform_vector(&w);
        let uv = equal_area_sphere_to_square(&wp.normalize());
        let rgb = self.image.lookup_nearest(&uv);
        self.scale
            * self.intensity.sample(lambda)
            * self
                .image
                .color_space()
                .illuminant_to_sampled_spectrum(rgb.to_rgb(), lambda)
    }
}

impl GoniometricLight {
    pub fn light_type(&self) -> LightType {
        LightType::DeltaPosition
    }

    pub fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut sum = SampledSpectrum::zero();
        let res = self.image.resolution();
        for y in 0..res.y {
            for x in 0..res.x {
                let uv = Point2f::new(
                    (x as Float + 0.5) / res.x as Float,
                    (y as Float + 0.5) / res.y as Float,
                );
                let rgb = self.image.lookup_nearest(&uv).to_rgb();
                let y = rgb[0].max(rgb[1]).max(rgb[2]);
                sum += self
                    .image
                    .color_space()
                    .illuminant_to_sampled_spectrum([y, y, y], lambda);
            }
        }
        self.scale * self.intensity.sample(lambda) * 4.0 * PI * sum
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
        let l = self.i(-wi, lambda) / Point3f::distance_squared(&p, &ctx.p);
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
        let phi = self.intensity.max_value() * self.scale * 4.0 * PI;
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

fn make_image(path: &str) -> Result<Image, PbrtError> {
    if !path.is_empty() {
        let (mut texels, resolution) = read_image(path)?;
        if resolution.x != resolution.y {
            return Err(PbrtError::error("goniometric light image must be square"));
        }
        let total = (resolution.x * resolution.y) as usize;
        for i in 0..total {
            let cc = texels[i].to_rgb();
            let cc = cc.iter().map(|x| x.max(0.0)).collect::<Vec<Float>>();
            texels[i] = RGBSpectrum::from(cc);
        }
        return Ok(Image::new(resolution, texels));
    }
    Ok(Image::new(Point2i::new(1, 1), vec![RGBSpectrum::one()]))
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
            let rgb = image.lookup_nearest(&uv).to_rgb();
            data[(y * res.x + x) as usize] = rgb[0].max(rgb[1]).max(rgb[2]).max(0.0);
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
        let white = Spectrum::from(params.color_space().illuminant.to_dense());
        let intensity = params.get_one_spectrum_typed("L", &white, SpectrumType::Illuminant);
        let photometric = spectrum_to_photometric(&intensity);
        let scale =
            params.get_one_float("scale", 1.0) / if photometric > 0.0 { photometric } else { 1.0 };
        let texmap = params.get_one_filename("mapname", "");
        let mi = MediumInterface::from(medium);
        let image = make_image(&texmap)?;
        Ok(GoniometricLight::new(
            light2world,
            &mi,
            intensity,
            scale,
            image,
        ))
    }
}
