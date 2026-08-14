use crate::base::bssrdf::BSSRDF;
use crate::base::bxdf::*;
use crate::base::camera::Camera;
use crate::base::material::Material;
use crate::base::sampler::Sampler;
use crate::base::Light;
use crate::bsdf::BSDF;
use crate::bxdfs::DiffuseBxDF;
use crate::materials::MaterialEvalContext;
use crate::media::*;
use crate::options::PbrtOptions;
use crate::textures::UniversalTextureEvaluator;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::spectrum::*;

use std::fmt;
use std::sync::Arc;

#[derive(Default, Debug, Clone, Copy)]
pub struct SurfaceInteractionShading {
    pub n: Normal3f,
    pub dpdu: Vector3f,
    pub dpdv: Vector3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
}

#[derive(Default, Clone)]
pub struct SurfaceInteraction {
    pub p: Point3f,
    pub p_error: Vector3f,
    pub n: Normal3f,
    pub time: Float,
    pub wo: Vector3f,
    pub medium_interface: MediumInterface,
    pub uv: Point2f,
    pub dpdu: Vector3f,
    pub dpdv: Vector3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
    pub material: Option<Arc<Material>>,
    pub area_light: Option<Arc<Light>>,
    pub shading: SurfaceInteractionShading,

    pub bsdf: Option<BSDF>,
    pub bssrdf: Option<Arc<BSSRDF>>,

    pub dpdx: Vector3f,
    pub dpdy: Vector3f,
    pub dudx: Float,
    pub dvdx: Float,
    pub dudy: Float,
    pub dvdy: Float,
    pub face_index: u32,
}

impl fmt::Debug for SurfaceInteraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SurfaceInteraction")
            .field("p", &self.p)
            .field("p_error", &self.p_error)
            .field("n", &self.n)
            .field("time", &self.time)
            .field("wo", &self.wo)
            .field("medium_interface", &self.medium_interface)
            .field("uv", &self.uv)
            .field("dpdu", &self.dpdu)
            .field("dpdv", &self.dpdv)
            .field("dndu", &self.dndu)
            .field("dndv", &self.dndv)
            .field("has_material", &self.material.is_some())
            .field("has_area_light", &self.area_light.is_some())
            .field("shading", &self.shading)
            .field("bsdf", &self.bsdf)
            .field("bssrdf", &self.bssrdf.is_some())
            .field("dpdx", &self.dpdx)
            .field("dpdy", &self.dpdy)
            .field("dudx", &self.dudx)
            .field("dvdx", &self.dvdx)
            .field("dudy", &self.dudy)
            .field("dvdy", &self.dvdy)
            .field("face_index", &self.face_index)
            .finish()
    }
}

impl SurfaceInteraction {
    pub fn new(
        p: &Point3f,
        p_error: &Point3f,
        uv: &Point2f,
        wo: &Vector3f,
        n: &Normal3f,
        dpdu: &Vector3f,
        dpdv: &Vector3f,
        dndu: &Vector3f,
        dndv: &Vector3f,
        time: Float,
        face_index: u32,
    ) -> Self {
        let shading = SurfaceInteractionShading {
            n: *n,
            dpdu: *dpdu,
            dpdv: *dpdv,
            dndu: *dndu,
            dndv: *dndv,
        };
        SurfaceInteraction {
            p: *p,
            p_error: *p_error,
            uv: *uv,
            wo: *wo,
            n: *n,
            medium_interface: MediumInterface::new(),
            dpdu: *dpdu,
            dpdv: *dpdv,
            dndu: *dndu,
            dndv: *dndv,
            time,
            material: None,
            area_light: None,
            shading,
            bsdf: None,
            bssrdf: None,
            dpdx: Vector3f::zero(),
            dpdy: Vector3f::zero(),
            dudx: 0.0,
            dudy: 0.0,
            dvdx: 0.0,
            dvdy: 0.0,
            face_index,
        }
    }

    pub fn set_shading_geometry(
        &mut self,
        ns: &Normal3f,
        dpdus: &Vector3f,
        dpdvs: &Vector3f,
        dndus: &Normal3f,
        dndvs: &Normal3f,
        orientation_is_authoritative: bool,
    ) {
        // Follow pbrt-v4: callers provide the shading normal explicitly
        // instead of recomputing it from the tangents here.
        self.shading.n = *ns;
        if orientation_is_authoritative {
            self.n = face_forward(&self.n, &self.shading.n);
        } else {
            self.shading.n = face_forward(&self.shading.n, &self.n);
        }

        // Initialize _shading_ partial derivative values
        self.shading.dpdu = *dpdus;
        self.shading.dpdv = *dpdvs;
        self.shading.dndu = *dndus;
        self.shading.dndv = *dndvs;
        while self.shading.dpdu.length_squared() > 1e16 || self.shading.dpdv.length_squared() > 1e16
        {
            self.shading.dpdu /= 1e8;
            self.shading.dpdv /= 1e8;
        }
    }

    pub fn set_shading_geometry_from_tangents(
        &mut self,
        dpdus: &Vector3f,
        dpdvs: &Vector3f,
        dndus: &Normal3f,
        dndvs: &Normal3f,
        orientation_is_authoritative: bool,
    ) {
        let ns = Vector3f::cross(dpdus, dpdvs).normalize();
        self.set_shading_geometry(
            &ns,
            dpdus,
            dpdvs,
            dndus,
            dndvs,
            orientation_is_authoritative,
        );
    }

    pub fn set_intersection_properties(
        &mut self,
        material: &Option<Arc<Material>>,
        area_light: &Option<Arc<Light>>,
        primitive_medium_interface: Option<&MediumInterface>,
        ray_medium: &Option<Arc<Medium>>,
    ) {
        self.material = material.clone();
        self.area_light = area_light.clone();
        if let Some(mi) = primitive_medium_interface {
            if mi.is_medium_transition() {
                self.medium_interface = mi.clone();
                return;
            }
        }
        self.medium_interface = MediumInterface::from(ray_medium);
    }

    pub fn get_material(&self) -> Option<Arc<Material>> {
        self.material.clone()
    }

    pub fn get_area_light(&self) -> Option<Arc<Light>> {
        self.area_light.clone()
    }

    pub fn bsdf(&self) -> Option<&BSDF> {
        self.bsdf.as_ref()
    }

    pub fn spawn_ray(&self, d: &Vector3f) -> Ray {
        let o = offset_ray_origin(&self.p, &self.p_error, &self.n, d);
        let mut r = Ray::new(&o, d, Float::INFINITY, self.time);
        r.medium = self.get_medium(d);
        return r;
    }

    /// pbrt-v4 `SurfaceInteraction::SkipIntersection` (interaction.cpp:91-97).
    /// Used when an intersected primitive has no BSDF (medium boundary) and
    /// we want the ray to continue past the hit without losing ray
    /// differentials. The base ray is respawned via `spawn_ray` (which offsets
    /// the origin to avoid self-intersection), and the differential origins
    /// are advanced by `t * direction`.
    pub fn skip_intersection(&self, ray: &mut RayDifferential, t: Float) {
        ray.ray = self.spawn_ray(&ray.ray.d);
        if ray.has_differentials {
            ray.rx_origin = ray.rx_origin + ray.rx_direction * t;
            ray.ry_origin = ray.ry_origin + ray.ry_direction * t;
        }
    }

    pub fn spawn_ray_with_differentials(
        &self,
        rayi: &RayDifferential,
        _bsdf: &BSDF,
        wi: &Vector3f,
        flags: BxDFFlags,
        eta: Float,
    ) -> RayDifferential {
        let mut rd = RayDifferential::from(self.spawn_ray(wi));
        if rayi.has_differentials {
            let mut n = self.shading.n;
            let mut dndx = self.shading.dndu * self.dudx + self.shading.dndv * self.dvdx;
            let mut dndy = self.shading.dndu * self.dudy + self.shading.dndv * self.dvdy;
            let dwodx = -rayi.rx_direction - self.wo;
            let dwody = -rayi.ry_direction - self.wo;

            if flags == (BXDF_REFLECTION | BXDF_SPECULAR) {
                rd.has_differentials = true;
                rd.rx_origin = self.p + self.dpdx;
                rd.ry_origin = self.p + self.dpdy;

                let dwo_dot_n_dx = Vector3f::dot(&dwodx, &n) + Vector3f::dot(&self.wo, &dndx);
                let dwo_dot_n_dy = Vector3f::dot(&dwody, &n) + Vector3f::dot(&self.wo, &dndy);
                let wo_dot_n = Vector3f::dot(&self.wo, &n);
                rd.rx_direction = *wi - dwodx + 2.0 * (wo_dot_n * dndx + dwo_dot_n_dx * n);
                rd.ry_direction = *wi - dwody + 2.0 * (wo_dot_n * dndy + dwo_dot_n_dy * n);
            } else if flags == (BXDF_TRANSMISSION | BXDF_SPECULAR) {
                rd.has_differentials = true;
                rd.rx_origin = self.p + self.dpdx;
                rd.ry_origin = self.p + self.dpdy;

                if Vector3f::dot(&self.wo, &n) < 0.0 {
                    n = -n;
                    dndx = -dndx;
                    dndy = -dndy;
                }

                let dwo_dot_n_dx = Vector3f::dot(&dwodx, &n) + Vector3f::dot(&self.wo, &dndx);
                let dwo_dot_n_dy = Vector3f::dot(&dwody, &n) + Vector3f::dot(&self.wo, &dndy);
                let wo_dot_n = Vector3f::dot(&self.wo, &n);
                let wi_dot_n = Vector3f::abs_dot(wi, &n);
                let mu = wo_dot_n / eta - wi_dot_n;
                let inv_eta = 1.0 / eta;
                let inv_eta2 = inv_eta * inv_eta;
                let dmudx = dwo_dot_n_dx * (inv_eta + inv_eta2 * wo_dot_n / Vector3f::dot(wi, &n));
                let dmudy = dwo_dot_n_dy * (inv_eta + inv_eta2 * wo_dot_n / Vector3f::dot(wi, &n));

                rd.rx_direction = *wi - eta * dwodx + (mu * dndx + dmudx * n);
                rd.ry_direction = *wi - eta * dwody + (mu * dndy + dmudy * n);
            }
        }

        if rd.rx_direction.length_squared() > 1e16
            || rd.ry_direction.length_squared() > 1e16
            || Vector3f::from(rd.rx_origin).length_squared() > 1e16
            || Vector3f::from(rd.ry_origin).length_squared() > 1e16
        {
            rd.has_differentials = false;
        }
        rd
    }

    pub fn spawn_ray_to_point(&self, p2: &Point3f) -> Ray {
        let d = *p2 - self.p;
        let o = offset_ray_origin(&self.p, &self.p_error, &self.n, &d);
        let mut r = Ray::new(&o, &d, Float::INFINITY, self.time);
        r.medium = self.get_medium(&d);
        return r;
    }

    fn compute_differentials_fail(&mut self) {
        self.dudx = 0.0;
        self.dvdx = 0.0;
        self.dudy = 0.0;
        self.dvdy = 0.0;
        self.dpdx = Vector3f::zero();
        self.dpdy = Vector3f::zero();
    }

    pub fn compute_differentials(
        &mut self,
        ray: &RayDifferential,
        camera: &Camera,
        samples_per_pixel: u32,
    ) {
        if PbrtOptions::get().disable_texture_filtering {
            self.compute_differentials_fail();
            return;
        }
        let p = self.p;
        let n = self.n;
        let mut estimated_dp = false;
        if ray.has_differentials
            && Vector3f::dot(&self.n, &ray.rx_direction) != 0.0
            && Vector3f::dot(&self.n, &ray.ry_direction) != 0.0
        {
            // Estimate screen-space change in $\pt{}$ using ray differentials.
            let d = -Vector3f::dot(&n, &Vector3f::from(p));
            let tx =
                (-Vector3f::dot(&n, &ray.rx_origin) - d) / Vector3f::dot(&n, &ray.rx_direction);
            let ty =
                (-Vector3f::dot(&n, &ray.ry_origin) - d) / Vector3f::dot(&n, &ray.ry_direction);
            if tx.is_finite() && ty.is_finite() {
                let px = ray.rx_origin + tx * ray.rx_direction;
                let py = ray.ry_origin + ty * ray.ry_direction;
                self.dpdx = px - p;
                self.dpdy = py - p;
                estimated_dp = true;
            }
        }

        if !estimated_dp {
            if let Some((dpdx, dpdy)) =
                camera.approximate_dp_dxy(p, n, self.time, samples_per_pixel)
            {
                self.dpdx = dpdx;
                self.dpdy = dpdy;
            } else {
                self.compute_differentials_fail();
                return;
            }
        }

        // Solve the least-squares system from pbrt-v4 to estimate screen-space
        // changes in $(u,v)$ from the geometric parameterization.
        let ata00 = Vector3f::dot(&self.dpdu, &self.dpdu);
        let ata01 = Vector3f::dot(&self.dpdu, &self.dpdv);
        let ata11 = Vector3f::dot(&self.dpdv, &self.dpdv);
        let det = ata00 * ata11 - ata01 * ata01;
        let inv_det = if det.is_finite() && det != 0.0 {
            1.0 / det
        } else {
            0.0
        };

        let atb0x = Vector3f::dot(&self.dpdu, &self.dpdx);
        let atb1x = Vector3f::dot(&self.dpdv, &self.dpdx);
        let atb0y = Vector3f::dot(&self.dpdu, &self.dpdy);
        let atb1y = Vector3f::dot(&self.dpdv, &self.dpdy);

        let dudx = (ata11 * atb0x - ata01 * atb1x) * inv_det;
        let dvdx = (ata00 * atb1x - ata01 * atb0x) * inv_det;
        let dudy = (ata11 * atb0y - ata01 * atb1y) * inv_det;
        let dvdy = (ata00 * atb1y - ata01 * atb0y) * inv_det;

        self.dudx = if dudx.is_finite() {
            Float::clamp(dudx, -1e8, 1e8)
        } else {
            0.0
        };
        self.dvdx = if dvdx.is_finite() {
            Float::clamp(dvdx, -1e8, 1e8)
        } else {
            0.0
        };
        self.dudy = if dudy.is_finite() {
            Float::clamp(dudy, -1e8, 1e8)
        } else {
            0.0
        };
        self.dvdy = if dvdy.is_finite() {
            Float::clamp(dvdy, -1e8, 1e8)
        } else {
            0.0
        };
    }

    pub fn get_bsdf(
        &mut self,
        ray: &RayDifferential,
        camera: &Camera,
        samples_per_pixel: u32,
        lambda: &mut SampledWavelengths,
        mut sampler: Option<&mut Sampler>,
    ) -> Option<BSDF> {
        let material = self.material.clone()?;
        self.compute_differentials(ray, camera, samples_per_pixel);
        material.apply_displacement(self);
        if let Some(new_lambda) = material.maybe_terminate_secondary_wavelengths(self, lambda) {
            *lambda = new_lambda;
        }

        let tex_eval = UniversalTextureEvaluator;
        let ctx = MaterialEvalContext::from(&*self);
        let bsdf = material.get_bsdf(&tex_eval, &ctx, lambda);
        if PbrtOptions::get().force_diffuse {
            let uc = [sampler
                .as_deref_mut()
                .map_or(0.5, |sampler| sampler.get_1d())];
            let u = [sampler
                .as_deref_mut()
                .map_or(Point2f::new(0.5, 0.5), |sampler| sampler.get_2d())];
            let r = bsdf.rho(self.wo, &uc, &u);
            return Some(BSDF::new(
                ctx.ns,
                ctx.dpdus,
                BxDF::Diffuse(Box::new(DiffuseBxDF::new(r))),
            ));
        }
        Some(bsdf)
    }

    pub fn get_bssrdf(&self, lambda: &SampledWavelengths) -> Option<BSSRDF> {
        let material = self.material.clone()?;
        let tex_eval = UniversalTextureEvaluator;
        let ctx = MaterialEvalContext::from(self);
        material.get_bssrdf(&tex_eval, &ctx, lambda)
    }

    /// pbrt-v4 `SurfaceInteraction::Le(w, lambda)` -- emitted radiance
    /// from this hit (only non-zero for area-light-bearing surfaces).
    /// Returns a `SampledSpectrum`, matching v4 verbatim.
    pub fn le(&self, w: Vector3f, lambda: &SampledWavelengths) -> SampledSpectrum {
        if let Some(light) = self.area_light.as_ref() {
            if light.as_ref().is_area() {
                return light.as_ref().l(self.p, self.n, self.uv, w, lambda);
            }
        }
        SampledSpectrum::zero()
    }

    pub fn get_medium(&self, w: &Vector3f) -> Option<Arc<Medium>> {
        if Vector3f::dot(w, &self.n) > 0.0 {
            return self.medium_interface.get_outside();
        } else {
            return self.medium_interface.get_inside();
        }
    }
}
