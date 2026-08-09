use crate::base::light::*;
use crate::base::lightsampler::LightSampleContext;
use crate::interaction::*;
use crate::lights::*;
use crate::media::*;
use crate::textures::create_spectrum_mipmap;
use crate::textures::MIPMap;
use crate::util::base::*;
use crate::util::error::PbrtError;
use crate::util::geometry::*;
use crate::util::profile::*;
use crate::util::sampling::*;
use crate::util::spectrum::rgb_to_spectrum::RGBColorSpace;
use crate::util::spectrum::*;
use crate::util::transform::*;
use crate::util::vecmath::frame::Frame;
use std::sync::RwLock;

// ===========================================================================
// PortalImageInfiniteLight - v4 lights.cpp:1108-1342
// ===========================================================================
//
// Restricts emission to a rectangular portal (e.g., a window). v4 stores
// a "rectified" image computed by resampling the source equal-area env
// map through the portal's tan-based projection and a
// `WindowedPiecewiseConstant2D` for importance sampling within the
// portal's image bounds. r4 mirrors the storage; we only sample
// directions whose image-space uv falls inside the portal.

pub struct PortalImageInfiniteLight {
    base: LightBase,
    /// 4 corner points (CCW: p00, p10, p11, p01 in v4 ordering).
    portal: [Point3f; 4],
    /// Portal-local frame: x = p03 (along edge 0->3), y = p01 (along edge 0->1),
    /// z = portal outward normal.
    portal_frame: Frame,
    /// Rectified env image (RGB MIPMap; level 0 is what we look up).
    image: MIPMap<RGBSpectrum>,
    width: usize,
    height: usize,
    /// Windowed importance distribution over the rectified image.
    distribution: WindowedPiecewiseConstant2D,
    scale: Float,
    image_color_space: &'static RGBColorSpace,
    scene_center: RwLock<Point3f>,
    scene_radius: RwLock<Float>,
}

impl PortalImageInfiniteLight {
    /// `equal_area_mipmap` is the env map in v4's equal-area projection.
    /// The rectified portal-image is built once at construction time
    /// matching pbrt-v4 `PortalImageInfiniteLight::ctor` (lights.cpp:1109).
    pub fn new(
        base: LightBase,
        equal_area_mipmap: &MIPMap<RGBSpectrum>,
        scale: Float,
        portal: [Point3f; 4],
        image_color_space: &'static RGBColorSpace,
    ) -> Result<Self, PbrtError> {
        // Sanity-check portal planarity (matches v4's Error path).
        let p01 = (portal[1] - portal[0]).normalize();
        let p12 = (portal[2] - portal[1]).normalize();
        let p32 = (portal[2] - portal[3]).normalize();
        let p03 = (portal[3] - portal[0]).normalize();
        if (Vector3f::dot(&p01, &p32) - 1.0).abs() > 1e-3
            || (Vector3f::dot(&p12, &p03) - 1.0).abs() > 1e-3
        {
            return Err(PbrtError::error(
                "PortalImageInfiniteLight: portal is not a planar quadrilateral",
            ));
        }
        if Vector3f::dot(&p01, &p12).abs() > 1e-3
            || Vector3f::dot(&p12, &p32).abs() > 1e-3
            || Vector3f::dot(&p32, &p03).abs() > 1e-3
            || Vector3f::dot(&p03, &p01).abs() > 1e-3
        {
            return Err(PbrtError::error(
                "PortalImageInfiniteLight: portal sides are not perpendicular",
            ));
        }
        let portal_frame = Frame::from_xy(p03, p01);
        let width = equal_area_mipmap.width();
        let height = equal_area_mipmap.height();
        let mut rectified: Vec<RGBSpectrum> = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let uv = Point2f::new(
                    (x as Float + 0.5) / width as Float,
                    (y as Float + 0.5) / height as Float,
                );
                // uv (rectified) -> direction in portal local frame, then to render space.
                let w_render = render_from_image_with(&portal_frame, uv);
                // Move into the light's local frame and look up the equal-area image.
                let w_light = base
                    .world_to_light()
                    .transform_vector(&w_render)
                    .normalize();
                let uv_equi = equal_area_sphere_to_square(&w_light);
                let rgb = equal_area_mipmap.lookup(&uv_equi, 0.0);
                rectified.push(rgb);
            }
        }
        let resolution = Point2i::new(width as i32, height as i32);
        let image = create_spectrum_mipmap(&resolution, &rectified)?;

        // Build the windowed distribution from the rectified image
        // (v4 lights.cpp:1173-1180). Weight = average RGB * |duv/dω|.
        let mut dist = vec![0.0; width * height];
        for y in 0..height {
            for x in 0..width {
                let uv = Point2f::new(
                    (x as Float + 0.5) / width as Float,
                    (y as Float + 0.5) / height as Float,
                );
                let (_, duv_dw) = render_from_image_with_jacobian(&portal_frame, uv);
                let rgb = image.texel(0, x as i32, y as i32).to_rgb();
                let avg = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
                dist[y * width + x] = if duv_dw > 0.0 {
                    avg.max(0.0) * duv_dw
                } else {
                    0.0
                };
            }
        }
        let distribution = WindowedPiecewiseConstant2D::new(dist, width, height);

        Ok(PortalImageInfiniteLight {
            base,
            portal,
            portal_frame,
            image,
            width,
            height,
            distribution,
            scale,
            image_color_space,
            scene_center: RwLock::new(Point3f::zero()),
            scene_radius: RwLock::new(Float::INFINITY),
        })
    }

    fn radius(&self) -> Float {
        *self.scene_radius.read().unwrap()
    }

    fn image_from_render(&self, w_render: &Vector3f) -> Option<(Point2f, Float)> {
        image_from_render_with(&self.portal_frame, w_render)
    }

    fn render_from_image(&self, uv: &Point2f) -> (Vector3f, Float) {
        render_from_image_with_jacobian(&self.portal_frame, *uv)
    }

    fn image_bounds(&self, p: &Point3f) -> Option<Bounds2f> {
        // The two diagonally-opposite portal corners drive the bounds.
        let (uv0, _) = self.image_from_render(&(self.portal[0] - *p).normalize())?;
        let (uv1, _) = self.image_from_render(&(self.portal[2] - *p).normalize())?;
        Some(Bounds2f::new(&uv0, &uv1))
    }

    fn area(&self) -> Float {
        (self.portal[1] - self.portal[0]).length() * (self.portal[3] - self.portal[0]).length()
    }

    fn image_lookup(&self, uv: Point2f, lambda: &SampledWavelengths) -> SampledSpectrum {
        let sx = ((uv.x * self.width as Float).floor() as i32).clamp(0, self.width as i32 - 1);
        let sy = ((uv.y * self.height as Float).floor() as i32).clamp(0, self.height as i32 - 1);
        let rgb = self.image.texel(0, sx, sy);
        let spec = self
            .image_color_space
            .illuminant_to_sampled_spectrum(rgb.to_rgb(), lambda);
        spec * self.scale
    }
}

impl PortalImageInfiniteLight {
    pub fn light_type(&self) -> LightType {
        LightType::Infinite
    }

    pub fn preprocess(&self, scene_bounds: &Bounds3f) {
        let (center, radius) = scene_bounds.bounding_sphere();
        *self.scene_center.write().unwrap() = center;
        *self.scene_radius.write().unwrap() = radius;
    }

    pub fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        let uv = match self.image_from_render(&ray.d.normalize()) {
            Some((uv, _)) => uv,
            None => return SampledSpectrum::zero(),
        };
        let bounds = match self.image_bounds(&ray.o) {
            Some(b) => b,
            None => return SampledSpectrum::zero(),
        };
        if uv.x < bounds.min.x || uv.x > bounds.max.x || uv.y < bounds.min.y || uv.y > bounds.max.y
        {
            return SampledSpectrum::zero();
        }
        self.image_lookup(uv, lambda)
    }

    pub fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut sum = SampledSpectrum::zero();
        for y in 0..self.height {
            for x in 0..self.width {
                let rgb = self.image.texel(0, x as i32, y as i32).to_rgb();
                let spec = self
                    .image_color_space
                    .illuminant_to_sampled_spectrum(rgb, lambda);
                let uv = Point2f::new(
                    (x as Float + 0.5) / self.width as Float,
                    (y as Float + 0.5) / self.height as Float,
                );
                let (_, duv_dw) = self.render_from_image(&uv);
                if duv_dw > 0.0 {
                    sum += spec / duv_dw;
                }
            }
        }
        sum * self.scale * self.area() / ((self.width * self.height) as Float)
    }

    pub fn sample_li(
        &self,
        ctx: &LightSampleContext,
        u: Point2f,
        lambda: &SampledWavelengths,
        _allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        let _p = ProfilePhase::new(Prof::LightSample);
        let bounds = self.image_bounds(&ctx.p)?;
        let (uv, map_pdf) = self.distribution.sample(&u, &bounds)?;
        let (wi, duv_dw) = self.render_from_image(&uv);
        if duv_dw <= 0.0 {
            return None;
        }
        let pdf = map_pdf / duv_dw;
        let l = self.image_lookup(uv, lambda);
        let p = ctx.p + wi * (2.0 * self.radius());
        let p_light = Interaction::from_light_sample(&p, 0.0, &self.base.medium_interface);
        Some(LightLiSample::new(l, wi, pdf, p_light))
    }

    pub fn pdf_li(
        &self,
        ctx: &LightSampleContext,
        wi: Vector3f,
        _allow_incomplete_pdf: bool,
    ) -> Float {
        let _p = ProfilePhase::new(Prof::LightPdf);
        let (uv, duv_dw) = match self.image_from_render(&wi) {
            Some(x) => x,
            None => return 0.0,
        };
        if duv_dw <= 0.0 {
            return 0.0;
        }
        let bounds = match self.image_bounds(&ctx.p) {
            Some(b) => b,
            None => return 0.0,
        };
        self.distribution.pdf(&uv, &bounds) / duv_dw
    }

    pub fn sample_le(
        &self,
        u1: Point2f,
        u2: Point2f,
        lambda: &SampledWavelengths,
        time: Float,
    ) -> Option<LightLeSample> {
        let _p = ProfilePhase::new(Prof::LightSample);
        let unit = Bounds2f::new(&Point2f::new(0.0, 0.0), &Point2f::new(1.0, 1.0));
        let (uv, map_pdf) = self.distribution.sample(&u1, &unit)?;
        let (w_pos, duv_dw) = self.render_from_image(&uv);
        if duv_dw <= 0.0 {
            return None;
        }
        let w = -w_pos;
        let pdf_dir = map_pdf / duv_dw;

        // Disk on the scene-bounding sphere, perpendicular to -w.
        let center = *self.scene_center.read().unwrap();
        let radius = *self.scene_radius.read().unwrap();
        let (v1, v2) = coordinate_system(&-w);
        let cd = concentric_sample_disk(&u2);
        let p_disk = center + radius * (cd.x * v1 + cd.y * v2);
        let ray = Ray::new(&(p_disk + radius * -w), &w, Float::INFINITY, time);
        let pdf_pos = 1.0 / (PI * radius * radius);

        let l = self.image_lookup(uv, lambda);
        Some(LightLeSample::new(l, ray, pdf_pos, pdf_dir))
    }

    pub fn pdf_le_ray(&self, ray: &Ray) -> (Float, Float) {
        let _p = ProfilePhase::new(Prof::LightPdf);
        let w = -ray.d.normalize();
        let (uv, duv_dw) = match self.image_from_render(&w) {
            Some(x) => x,
            None => return (0.0, 0.0),
        };
        if duv_dw <= 0.0 {
            return (0.0, 0.0);
        }
        let unit = Bounds2f::new(&Point2f::new(0.0, 0.0), &Point2f::new(1.0, 1.0));
        let map_pdf = self.distribution.pdf(&uv, &unit);
        let r = self.radius();
        let pdf_pos = 1.0 / (PI * r * r);
        let pdf_dir = map_pdf / duv_dw;
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

unsafe impl Send for PortalImageInfiniteLight {}
unsafe impl Sync for PortalImageInfiniteLight {}

// ---- Portal projection helpers (free functions so they're usable in
// `new` before `self` exists) -------------------------------------------

/// pbrt-v4 `PortalImageInfiniteLight::RenderFromImage(uv)`
/// (lights.h:694-705) without the Jacobian output.
fn render_from_image_with(frame: &Frame, uv: Point2f) -> Vector3f {
    let alpha = -std::f64::consts::FRAC_PI_2 as Float + uv.x * PI;
    let beta = -std::f64::consts::FRAC_PI_2 as Float + uv.y * PI;
    let x = alpha.tan();
    let y = beta.tan();
    let w = Vector3f::new(x, y, 1.0).normalize();
    frame.from_local(w)
}

/// Same as `render_from_image_with` but also returns `|duv/dω|`
/// (v4 lights.h:702-704).
fn render_from_image_with_jacobian(frame: &Frame, uv: Point2f) -> (Vector3f, Float) {
    let alpha = -std::f64::consts::FRAC_PI_2 as Float + uv.x * PI;
    let beta = -std::f64::consts::FRAC_PI_2 as Float + uv.y * PI;
    let x = alpha.tan();
    let y = beta.tan();
    let w_local = Vector3f::new(x, y, 1.0).normalize();
    let duv_dw = if w_local.z > 0.0 {
        PI * PI * (1.0 - w_local.x * w_local.x) * (1.0 - w_local.y * w_local.y) / w_local.z
    } else {
        0.0
    };
    (frame.from_local(w_local), duv_dw)
}

/// pbrt-v4 `PortalImageInfiniteLight::ImageFromRender(wRender)`
/// (lights.h:677-691). Returns `None` if the direction is behind the
/// portal plane.
fn image_from_render_with(frame: &Frame, w_render: &Vector3f) -> Option<(Point2f, Float)> {
    let w = frame.to_local(*w_render);
    if w.z <= 0.0 {
        return None;
    }
    let duv_dw = PI * PI * (1.0 - w.x * w.x) * (1.0 - w.y * w.y) / w.z;
    let alpha = w.x.atan2(w.z);
    let beta = w.y.atan2(w.z);
    let u = ((alpha + std::f64::consts::FRAC_PI_2 as Float) / PI).clamp(0.0, 1.0);
    let v = ((beta + std::f64::consts::FRAC_PI_2 as Float) / PI).clamp(0.0, 1.0);
    Some((Point2f::new(u, v), duv_dw))
}
