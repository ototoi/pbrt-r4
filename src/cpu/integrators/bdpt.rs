use crate::base::bxdf::{
    is_non_specular, is_reflective, is_transmissive, TransportMode, BXDF_ALL, BXDF_REFL_TRANS_ALL,
};
use crate::base::camera::Camera;
use crate::base::film::Film;
use crate::base::light::{is_delta_light, Light, LightType};
use crate::base::lightsampler::{LightSampleContext, LightSampler};
use crate::base::medium::sample_t_maj_coefficients;
use crate::base::sampler::Sampler;
use crate::bsdf::BSDF;
use crate::cpu::integrators::*;
use crate::film::VisibleSurface;
use crate::interaction::*;
use crate::paramdict::*;
use crate::scene::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::memory::*;
use crate::util::rng::RNG;
use crate::util::spectrum::*;

use std::sync::{Arc, RwLock};

// ============================================================================
// VertexType / EndpointInteraction / Vertex
// ============================================================================

/// pbrt-v4 `enum class VertexType { Camera, Light, Surface, Medium }`.
#[derive(Clone, Copy, PartialEq, Debug)]
enum VertexType {
    Camera,
    Light,
    Surface,
    Medium,
}

/// pbrt-v4 `struct EndpointInteraction : Interaction` (integrators.cpp:1526).
/// Holds either a `Camera` reference or a `Light` reference plus the
/// minimal Interaction (position/normal/time).
#[derive(Clone)]
pub struct EndpointInteraction {
    base: BaseInteraction,
    camera: Option<Arc<Camera>>,
    light: Option<Arc<Light>>,
}

impl EndpointInteraction {
    fn from_camera_ray(camera: Arc<Camera>, ray: &Ray) -> Self {
        let mut base = BaseInteraction::default();
        base.p = ray.o;
        base.time = ray.time;
        Self {
            base,
            camera: Some(camera),
            light: None,
        }
    }

    fn from_camera_interaction(camera: Arc<Camera>, it: &Interaction) -> Self {
        let (p, p_error, n, time) = it.get_base_tuple();
        let mut base = BaseInteraction::default();
        base.p = p;
        base.p_error = p_error;
        base.n = n;
        base.time = time;
        Self {
            base,
            camera: Some(camera),
            light: None,
        }
    }

    fn from_light_interaction(light: Arc<Light>, intr: &Interaction) -> Self {
        let (p, p_error, n, time) = intr.get_base_tuple();
        let mut base = BaseInteraction::default();
        base.p = p;
        base.p_error = p_error;
        base.n = n;
        base.time = time;
        Self {
            base,
            camera: None,
            light: Some(light),
        }
    }

    fn from_light_ray(light: Arc<Light>, ray: &Ray) -> Self {
        let mut base = BaseInteraction::default();
        base.p = ray.o;
        base.time = ray.time;
        Self {
            base,
            camera: None,
            light: Some(light),
        }
    }

    /// pbrt-v4 `EndpointInteraction(const Ray &ray)` — used for escaped
    /// camera rays. n = Normal3f(-ray.d), p = ray(1).
    fn from_escaped_ray(ray: &Ray) -> Self {
        let mut base = BaseInteraction::default();
        base.p = ray.position(1.0);
        base.n = -ray.d;
        base.time = ray.time;
        Self {
            base,
            camera: None,
            light: None,
        }
    }

    fn p(&self) -> Point3f {
        self.base.p
    }

    fn time(&self) -> Float {
        self.base.time
    }

    fn n(&self) -> Normal3f {
        self.base.n
    }

    fn as_interaction(&self) -> Interaction {
        Interaction::Base(self.base.clone())
    }

    fn spawn_ray(&self, d: &Vector3f) -> Ray {
        self.as_interaction().spawn_ray(d)
    }
}

/// pbrt-v4 `struct Vertex` (integrators.cpp:1736). Surface vertices carry
/// a `SurfaceInteraction` + `BSDF`, medium vertices carry a
/// `MediumInteraction`, and endpoint vertices carry an `EndpointInteraction`.
#[derive(Clone)]
pub struct Vertex {
    vtype: VertexType,
    beta: SampledSpectrum,
    delta: bool,
    pdf_fwd: Float,
    pdf_rev: Float,
    // One of the following is meaningful depending on `vtype`:
    ei: EndpointInteraction,        // Camera / Light
    si: Option<SurfaceInteraction>, // Surface
    mi: Option<MediumInteraction>,  // Medium
    bsdf: Option<BSDF>,             // Surface
}

impl Vertex {
    fn create_camera_from_ray(camera: Arc<Camera>, ray: &Ray, beta: SampledSpectrum) -> Self {
        Self {
            vtype: VertexType::Camera,
            beta,
            delta: false,
            pdf_fwd: 0.0,
            pdf_rev: 0.0,
            ei: EndpointInteraction::from_camera_ray(camera, ray),
            si: None,
            mi: None,
            bsdf: None,
        }
    }

    fn create_camera_from_interaction(
        camera: Arc<Camera>,
        it: &Interaction,
        beta: SampledSpectrum,
    ) -> Self {
        Self {
            vtype: VertexType::Camera,
            beta,
            delta: false,
            pdf_fwd: 0.0,
            pdf_rev: 0.0,
            ei: EndpointInteraction::from_camera_interaction(camera, it),
            si: None,
            mi: None,
            bsdf: None,
        }
    }

    fn create_light_from_ray(
        light: Arc<Light>,
        ray: &Ray,
        le: SampledSpectrum,
        pdf: Float,
    ) -> Self {
        Self {
            vtype: VertexType::Light,
            beta: le,
            delta: false,
            pdf_fwd: pdf,
            pdf_rev: 0.0,
            ei: EndpointInteraction::from_light_ray(light, ray),
            si: None,
            mi: None,
            bsdf: None,
        }
    }

    fn create_light_from_interaction(
        light: Arc<Light>,
        intr: &Interaction,
        le: SampledSpectrum,
        pdf: Float,
    ) -> Self {
        Self {
            vtype: VertexType::Light,
            beta: le,
            delta: false,
            pdf_fwd: pdf,
            pdf_rev: 0.0,
            ei: EndpointInteraction::from_light_interaction(light, intr),
            si: None,
            mi: None,
            bsdf: None,
        }
    }

    fn create_light_from_endpoint(
        ei: EndpointInteraction,
        beta: SampledSpectrum,
        pdf: Float,
    ) -> Self {
        Self {
            vtype: VertexType::Light,
            beta,
            delta: false,
            pdf_fwd: pdf,
            pdf_rev: 0.0,
            ei,
            si: None,
            mi: None,
            bsdf: None,
        }
    }

    fn create_surface(
        si: SurfaceInteraction,
        bsdf: BSDF,
        beta: SampledSpectrum,
        pdf: Float,
        prev: &Vertex,
    ) -> Self {
        let mut v = Self {
            vtype: VertexType::Surface,
            beta,
            delta: false,
            pdf_fwd: 0.0,
            pdf_rev: 0.0,
            // Endpoint placeholder; not used on surface vertex.
            ei: EndpointInteraction {
                base: BaseInteraction::default(),
                camera: None,
                light: None,
            },
            si: Some(si),
            mi: None,
            bsdf: Some(bsdf),
        };
        v.pdf_fwd = prev.convert_density(pdf, &v);
        v
    }

    fn create_medium(
        mi: MediumInteraction,
        beta: SampledSpectrum,
        pdf: Float,
        prev: &Vertex,
    ) -> Self {
        let mut v = Self {
            vtype: VertexType::Medium,
            beta,
            delta: false,
            pdf_fwd: 0.0,
            pdf_rev: 0.0,
            ei: EndpointInteraction {
                base: BaseInteraction::default(),
                camera: None,
                light: None,
            },
            si: None,
            mi: Some(mi),
            bsdf: None,
        };
        v.pdf_fwd = prev.convert_density(pdf, &v);
        v
    }

    fn p(&self) -> Point3f {
        match self.vtype {
            VertexType::Surface => self.si.as_ref().unwrap().p,
            VertexType::Medium => self.mi.as_ref().unwrap().p,
            _ => self.ei.p(),
        }
    }

    pub fn time(&self) -> Float {
        match self.vtype {
            VertexType::Surface => self.si.as_ref().unwrap().time,
            VertexType::Medium => self.mi.as_ref().unwrap().time,
            _ => self.ei.time(),
        }
    }

    fn ng(&self) -> Normal3f {
        match self.vtype {
            VertexType::Surface => self.si.as_ref().unwrap().n,
            VertexType::Medium => self.mi.as_ref().unwrap().n,
            _ => self.ei.n(),
        }
    }

    fn ns(&self) -> Normal3f {
        match self.vtype {
            VertexType::Surface => self.si.as_ref().unwrap().shading.n,
            VertexType::Medium => self.mi.as_ref().unwrap().n,
            _ => self.ei.n(),
        }
    }

    fn is_on_surface(&self) -> bool {
        self.ng().length_squared() > 0.0
    }

    fn get_interaction(&self) -> Interaction {
        match self.vtype {
            VertexType::Surface => Interaction::Surface(self.si.as_ref().unwrap().clone()),
            VertexType::Medium => Interaction::Medium(self.mi.as_ref().unwrap().clone()),
            _ => self.ei.as_interaction(),
        }
    }

    fn f(&self, next: &Vertex, mode: TransportMode) -> SampledSpectrum {
        let wi = next.p() - self.p();
        if wi.length_squared() == 0.0 {
            return SampledSpectrum::zero();
        }
        let wi = wi.normalize();
        match self.vtype {
            VertexType::Surface => {
                let si = self.si.as_ref().unwrap();
                self.bsdf.as_ref().unwrap().f(si.wo, wi, mode)
            }
            VertexType::Medium => {
                let mi = self.mi.as_ref().unwrap();
                SampledSpectrum::new(mi.phase.p(&mi.wo, &wi))
            }
            _ => SampledSpectrum::zero(),
        }
    }

    fn is_connectible(&self) -> bool {
        match self.vtype {
            VertexType::Light => self
                .ei
                .light
                .as_ref()
                .map(|l| l.as_ref().light_type() != LightType::DeltaDirection)
                .unwrap_or(false),
            VertexType::Camera => true,
            VertexType::Surface => is_non_specular(self.bsdf.as_ref().unwrap().flags()),
            VertexType::Medium => true,
        }
    }

    fn is_light(&self) -> bool {
        if matches!(self.vtype, VertexType::Light) {
            return true;
        }
        if matches!(self.vtype, VertexType::Surface) {
            if let Some(si) = self.si.as_ref() {
                if let Some(area_light) = si.get_area_light() {
                    let _ = area_light;
                    return true;
                }
            }
        }
        false
    }

    fn is_delta_light(&self) -> bool {
        matches!(self.vtype, VertexType::Light)
            && self
                .ei
                .light
                .as_ref()
                .map(|l| is_delta_light(l.as_ref().light_type()))
                .unwrap_or(false)
    }

    fn is_infinite_light(&self) -> bool {
        matches!(self.vtype, VertexType::Light)
            && self
                .ei
                .light
                .as_ref()
                .map(|l| {
                    let t = l.as_ref().light_type();
                    t == LightType::Infinite || t == LightType::DeltaDirection
                })
                .unwrap_or(true)
    }

    fn le(
        &self,
        infinite_lights: &[Arc<Light>],
        v: &Vertex,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        if !self.is_light() {
            return SampledSpectrum::zero();
        }
        let w_full = v.p() - self.p();
        if w_full.length_squared() == 0.0 {
            return SampledSpectrum::zero();
        }
        let w = w_full.normalize();
        if self.is_infinite_light() {
            let mut le_sum = SampledSpectrum::zero();
            for light in infinite_lights.iter() {
                let r = Ray::new(&self.p(), &(-w), Float::INFINITY, self.time());
                le_sum += light.as_ref().le(&r, lambda);
            }
            return le_sum;
        }
        if let Some(si) = self.si.as_ref() {
            if let Some(area_light) = si.get_area_light() {
                return area_light.as_ref().l(si.p, si.n, si.uv, w, lambda);
            }
        }
        SampledSpectrum::zero()
    }

    fn convert_density(&self, mut pdf: Float, next: &Vertex) -> Float {
        if next.is_infinite_light() {
            return pdf;
        }
        let w = next.p() - self.p();
        if w.length_squared() == 0.0 {
            return 0.0;
        }
        let inv_dist2 = 1.0 / w.length_squared();
        if next.is_on_surface() {
            pdf *= Float::abs(Vector3f::dot(&next.ng(), &(w * inv_dist2.sqrt())));
        }
        pdf * inv_dist2
    }

    fn pdf(&self, base: &IntegratorBase, prev: Option<&Vertex>, next: &Vertex) -> Float {
        if matches!(self.vtype, VertexType::Light) {
            return self.pdf_light(base, next);
        }
        let wn_full = next.p() - self.p();
        if wn_full.length_squared() == 0.0 {
            return 0.0;
        }
        let wn = wn_full.normalize();
        let wp = if let Some(prev) = prev {
            let w = prev.p() - self.p();
            if w.length_squared() == 0.0 {
                return 0.0;
            }
            w.normalize()
        } else {
            // Camera vertex: prev must be None.
            assert!(matches!(self.vtype, VertexType::Camera));
            Vector3f::zero()
        };

        let pdf = match self.vtype {
            VertexType::Camera => {
                let camera = self.ei.camera.as_ref().expect("camera vertex");
                let ray = self.ei.spawn_ray(&wn);
                camera
                    .as_ref()
                    .pdf_we(&ray)
                    .map(|(_pdf_pos, pdf_dir)| pdf_dir)
                    .unwrap_or(0.0)
            }
            VertexType::Surface => self.bsdf.as_ref().unwrap().pdf(
                wp,
                wn,
                TransportMode::Radiance,
                BXDF_REFL_TRANS_ALL,
            ),
            VertexType::Medium => self.mi.as_ref().unwrap().phase.p(&wp, &wn),
            _ => 0.0,
        };
        self.convert_density(pdf, next)
    }

    fn pdf_light(&self, base: &IntegratorBase, v: &Vertex) -> Float {
        let mut w = v.p() - self.p();
        let lsq = w.length_squared();
        if lsq == 0.0 {
            return 0.0;
        }
        let inv_dist2 = 1.0 / lsq;
        w *= inv_dist2.sqrt();
        let mut pdf: Float;
        if self.is_infinite_light() {
            // pdfPos = 1 / (Pi * sceneRadius^2)
            let (_, radius) = base.world_bound().bounding_sphere();
            pdf = 1.0 / (PI * radius * radius);
        } else if self.is_on_surface() {
            let light = self.resolve_area_or_endpoint_light();
            let (_, pdf_dir) = match light.as_ref() {
                Some(l) => l.as_ref().pdf_le_interaction(&self.get_interaction(), w),
                None => return 0.0,
            };
            pdf = pdf_dir * inv_dist2;
        } else {
            // Non-infinite point/directional light: PDF_Le(Ray)
            let light = self.ei.light.as_ref();
            let (_, pdf_dir) = match light {
                Some(l) => {
                    let ray = Ray::new(&self.p(), &w, Float::INFINITY, self.time());
                    l.as_ref().pdf_le_ray(&ray)
                }
                None => return 0.0,
            };
            pdf = pdf_dir * inv_dist2;
        }
        if v.is_on_surface() {
            pdf *= Float::abs(Vector3f::dot(&v.ng(), &w));
        }
        pdf
    }

    fn pdf_light_origin(
        &self,
        infinite_lights: &[Arc<Light>],
        v: &Vertex,
        light_sampler: &LightSampler,
    ) -> Float {
        let w_full = v.p() - self.p();
        if w_full.length_squared() == 0.0 {
            return 0.0;
        }
        let w = w_full.normalize();
        if self.is_infinite_light() {
            return infinite_light_density(infinite_lights, light_sampler, w);
        }
        let light = self.resolve_area_or_endpoint_light();
        let Some(light) = light else { return 0.0 };
        let pdf_choice = light_sampler.pmf(&LightSampleContext::default(), &light);
        let (pdf_pos, _) = if self.is_on_surface() {
            light
                .as_ref()
                .pdf_le_interaction(&self.get_interaction(), w)
        } else {
            let ray = Ray::new(&self.p(), &w, Float::INFINITY, self.time());
            light.as_ref().pdf_le_ray(&ray)
        };
        pdf_pos * pdf_choice
    }

    fn resolve_area_or_endpoint_light(&self) -> Option<Arc<Light>> {
        if matches!(self.vtype, VertexType::Light) {
            return self.ei.light.clone();
        }
        if let Some(si) = self.si.as_ref() {
            return si.get_area_light();
        }
        None
    }
}

// ============================================================================
// Random walk
// ============================================================================

fn hash_f32(x: Float) -> u64 {
    let mut z = (x.to_bits() as u64).wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn sample_discrete3(weights: &[Float; 3], u: Float) -> i32 {
    let sum = weights[0] + weights[1] + weights[2];
    if sum <= 0.0 {
        return -1;
    }
    let mut cdf = weights[0] / sum;
    if u < cdf {
        return 0;
    }
    cdf += weights[1] / sum;
    if u < cdf {
        1
    } else {
        2
    }
}

#[allow(clippy::too_many_arguments)]
fn random_walk(
    base: &IntegratorBase,
    camera: &Camera,
    lambda: &mut SampledWavelengths,
    mut ray: RayDifferential,
    sampler: &mut Sampler,
    _scratch_buffer: &mut MemoryArena,
    mut beta: SampledSpectrum,
    mut pdf_fwd: Float,
    max_depth: usize,
    mode: TransportMode,
    path: &mut Vec<Vertex>,
    regularize: bool,
) -> usize {
    if max_depth == 0 {
        return 0;
    }
    let mut bounces: usize = 0;
    // pbrt-v4 (integrators.cpp:1972) tracks whether any non-specular bounce
    // has occurred so far; the regularize flag only kicks in after the
    // first non-specular vertex (caustic / specular paths are left alone).
    let mut any_non_specular_bounces: bool = false;
    loop {
        if beta.is_black() {
            break;
        }
        let si = base.intersect(&ray.ray, Float::INFINITY);
        let mut scattered = false;
        let mut terminated = false;
        if let Some(medium) = ray.ray.medium.clone() {
            let t_max = si.as_ref().map(|s| s.t_hit).unwrap_or(Float::INFINITY);
            let medium_ray = ray.ray.clone();
            let ray_time = medium_ray.time;
            let ray_medium = medium_ray.medium.clone();
            let mut next_ray: Option<RayDifferential> = None;
            let mut rng = RNG::new();
            rng.set_sequence_with_seed(hash_f32(sampler.get_1d()), hash_f32(sampler.get_1d()));
            let t_maj = sample_t_maj_coefficients(
                medium.as_ref(),
                &medium_ray,
                t_max,
                sampler.get_1d(),
                lambda,
                &mut rng,
                |p, coeff, sigma_maj, t_maj, rng| {
                    if sigma_maj[0] <= 0.0 {
                        terminated = true;
                        return false;
                    }
                    let p_absorb = coeff.sigma_a[0] / sigma_maj[0];
                    let p_scatter = coeff.sigma_s[0] / sigma_maj[0];
                    let p_null = Float::max(0.0, 1.0 - p_absorb - p_scatter);
                    match sample_discrete3(&[p_absorb, p_scatter, p_null], rng.uniform_float()) {
                        0 => {
                            terminated = true;
                            false
                        }
                        1 => {
                            let pdf = t_maj[0] * coeff.sigma_s[0];
                            if pdf <= 0.0 {
                                terminated = true;
                                return false;
                            }
                            beta *= t_maj * coeff.sigma_s / pdf;
                            let phase = medium.sample_phase_function(&p, lambda);
                            let intr = MediumInteraction::new(
                                &p,
                                &-ray.ray.d,
                                ray_time,
                                &ray_medium,
                                &phase,
                            );
                            let prev = path.last().cloned().expect("medium vertex has predecessor");
                            path.push(Vertex::create_medium(intr.clone(), beta, pdf_fwd, &prev));
                            bounces += 1;
                            if bounces >= max_depth {
                                terminated = true;
                                return false;
                            }
                            let (phase_pdf, wi) = intr.phase.sample_p(&intr.wo, &sampler.get_2d());
                            if phase_pdf <= 0.0 {
                                terminated = true;
                                return false;
                            }
                            beta *= intr.phase.p(&intr.wo, &wi) / phase_pdf;
                            pdf_fwd = phase_pdf;
                            let vertex = path.last().cloned().unwrap();
                            let prev_index = path.len() - 2;
                            path[prev_index].pdf_rev =
                                vertex.convert_density(phase_pdf, &path[prev_index]);
                            next_ray = Some(intr.spawn_ray(&wi).into());
                            any_non_specular_bounces = true;
                            scattered = true;
                            false
                        }
                        _ => {
                            let sigma_n = clamp_zero(sigma_maj - coeff.sigma_a - coeff.sigma_s);
                            let pdf = t_maj[0] * sigma_n[0];
                            if pdf <= 0.0 {
                                beta = SampledSpectrum::zero();
                            } else {
                                beta *= t_maj * sigma_n / pdf;
                            }
                            !beta.is_black()
                        }
                    }
                },
            );
            if !scattered && !terminated && t_maj[0] > 0.0 {
                beta *= t_maj / t_maj[0];
            }
            if let Some(next_ray) = next_ray {
                ray = next_ray;
            }
        }
        if terminated {
            return bounces;
        }
        if scattered {
            continue;
        }
        let Some(mut si) = si else {
            // v4: capture escaped rays when tracing from the camera.
            if matches!(mode, TransportMode::Radiance) {
                path.push(Vertex::create_light_from_endpoint(
                    EndpointInteraction::from_escaped_ray(&ray.ray),
                    beta,
                    pdf_fwd,
                ));
                bounces += 1;
            }
            break;
        };

        let bsdf = match si.intr.get_bsdf(
            &ray,
            camera,
            sampler.samples_per_pixel(),
            lambda,
            Some(sampler),
        ) {
            Some(mut b) => {
                // pbrt-v4 (integrators.cpp:2017-2020): if regularization is
                // enabled and we've already taken a non-specular bounce,
                // roughen the BSDF before it participates in MIS to suppress
                // fireflies on caustic paths.
                if regularize && any_non_specular_bounces {
                    b.regularize();
                }
                b
            }
            None => {
                ray = si.intr.spawn_ray(&ray.ray.d).into();
                continue;
            }
        };

        // Initialize new surface vertex referencing previous path vertex.
        let prev_index = path.len() - 1;
        // Borrow prev via clone since we need to mutate path while
        // referencing.
        let prev_snapshot = path[prev_index].clone();
        let vertex =
            Vertex::create_surface(si.intr.clone(), bsdf.clone(), beta, pdf_fwd, &prev_snapshot);
        path.push(vertex);
        bounces += 1;
        if bounces >= max_depth {
            break;
        }

        // Sample BSDF at current vertex
        let wo = si.intr.wo;
        let u = sampler.get_1d();
        let bs = match bsdf.sample_f(wo, u, sampler.get_2d(), mode, BXDF_ALL) {
            Some(s) => s,
            None => break,
        };
        // pbrt-v4 (integrators.cpp:2100) flags any non-specular sample so
        // that subsequent vertices may be regularized.
        any_non_specular_bounces |= !bs.is_specular();
        let new_pdf_fwd = if bs.pdf_is_proportional {
            bsdf.pdf(wo, bs.wi, mode, BXDF_REFL_TRANS_ALL)
        } else {
            bs.pdf
        };
        beta *=
            bs.f * (Float::abs(Vector3f::dot(&bs.wi, &Vector3f::from(si.intr.shading.n))) / bs.pdf);
        let new_ray = si
            .intr
            .spawn_ray_with_differentials(&ray, &bsdf, &bs.wi, bs.flags, bs.eta);

        // pbrt-v4 (integrators.cpp:2104) evaluates the reverse-direction
        // PDF in the opposite transport mode. For symmetric BSDFs (diffuse,
        // conductor) this is a no-op; for asymmetric BSDFs (dielectric with
        // eta, measured) it slightly changes the MIS denominator.
        let pdf_rev = bsdf.pdf(bs.wi, wo, !mode, BXDF_REFL_TRANS_ALL);
        let (final_pdf_fwd, final_pdf_rev) = if bs.is_specular() {
            // delta vertex
            path.last_mut().unwrap().delta = true;
            (0.0, 0.0)
        } else {
            (new_pdf_fwd, pdf_rev)
        };

        // Update prev.pdfRev from this vertex's perspective.
        let curr_snapshot = path.last().unwrap().clone();
        let prev_converted = curr_snapshot.convert_density(final_pdf_rev, &path[prev_index]);
        path[prev_index].pdf_rev = prev_converted;
        pdf_fwd = final_pdf_fwd;
        ray = new_ray;
    }
    bounces
}

// ============================================================================
// Camera / light subpath generation
// ============================================================================

pub fn generate_camera_subpath(
    base: &IntegratorBase,
    camera: &Arc<Camera>,
    ray: &RayDifferential,
    lambda: &mut SampledWavelengths,
    sampler: &mut Sampler,
    scratch_buffer: &mut MemoryArena,
    max_depth: usize,
    path: &mut Vec<Vertex>,
    regularize: bool,
) -> usize {
    if max_depth == 0 {
        return 0;
    }
    let beta = SampledSpectrum::one();
    path.push(Vertex::create_camera_from_ray(
        Arc::clone(camera),
        &ray.ray,
        beta,
    ));
    let pdf_dir = match camera.as_ref().pdf_we(&ray.ray) {
        Some((_, pd)) => pd,
        None => return 1,
    };
    random_walk(
        base,
        camera.as_ref(),
        lambda,
        ray.clone(),
        sampler,
        scratch_buffer,
        beta,
        pdf_dir,
        max_depth - 1,
        TransportMode::Radiance,
        path,
        regularize,
    ) + 1
}

pub fn generate_light_subpath(
    base: &IntegratorBase,
    camera: &Arc<Camera>,
    lambda: &mut SampledWavelengths,
    sampler: &mut Sampler,
    scratch_buffer: &mut MemoryArena,
    max_depth: usize,
    time: Float,
    light_sampler: &LightSampler,
    path: &mut Vec<Vertex>,
    regularize: bool,
) -> usize {
    if max_depth == 0 {
        return 0;
    }
    let ctx = LightSampleContext::default();
    let sampled_light = light_sampler.sample(&ctx, sampler.get_1d());
    let Some(sl) = sampled_light else { return 0 };
    let light = sl.light;
    let light_pdf = sl.p;
    let ul0 = sampler.get_2d();
    let ul1 = sampler.get_2d();
    let les = light.as_ref().sample_le(ul0, ul1, lambda, time);
    let Some(les) = les else { return 0 };
    if les.pdf_pos == 0.0 || les.pdf_dir == 0.0 || les.l.is_black() {
        return 0;
    }
    let ray = RayDifferential::from(les.ray.clone());
    let p_l = light_pdf * les.pdf_pos;

    let v0 = match les.intr.as_ref() {
        Some(intr) => Vertex::create_light_from_interaction(Arc::clone(&light), intr, les.l, p_l),
        None => Vertex::create_light_from_ray(Arc::clone(&light), &les.ray, les.l, p_l),
    };
    path.push(v0);
    let beta = les.l * les.abs_cos_theta(ray.ray.d) / (p_l * les.pdf_dir);

    let n_vertices = random_walk(
        base,
        camera.as_ref(),
        lambda,
        ray.clone(),
        sampler,
        scratch_buffer,
        beta,
        les.pdf_dir,
        max_depth - 1,
        TransportMode::Importance,
        path,
        regularize,
    );

    // Correct subpath sampling densities for infinite area lights.
    if path[0].is_infinite_light() {
        if n_vertices > 0 {
            path[1].pdf_fwd = les.pdf_pos;
            if path[1].is_on_surface() {
                path[1].pdf_fwd *=
                    Float::abs(Vector3f::dot(&ray.ray.d, &Vector3f::from(path[1].ng())));
            }
        }
        path[0].pdf_fwd = infinite_light_density(&base.infinite_lights, light_sampler, ray.ray.d);
    }
    let _ = camera;
    n_vertices + 1
}

// ============================================================================
// InfiniteLightDensity / G / MIS weight
// ============================================================================

fn infinite_light_density(
    infinite_lights: &[Arc<Light>],
    light_sampler: &LightSampler,
    w: Vector3f,
) -> Float {
    let mut pdf = 0.0;
    for light in infinite_lights.iter() {
        let ctx = LightSampleContext::default();
        pdf += light.as_ref().pdf_li(&ctx, -w, false) * light_sampler.pmf(&ctx, light);
    }
    pdf
}

fn geometry_term(
    base: &IntegratorBase,
    v0: &Vertex,
    v1: &Vertex,
    lambda: &SampledWavelengths,
) -> SampledSpectrum {
    let mut d = v0.p() - v1.p();
    let mut g = 1.0 / d.length_squared();
    d *= g.sqrt();
    if v0.is_on_surface() {
        g *= Float::abs(Vector3f::dot(&v0.ns(), &d));
    }
    if v1.is_on_surface() {
        g *= Float::abs(Vector3f::dot(&v1.ns(), &d));
    }
    base.tr(&v0.get_interaction(), &v1.get_interaction(), lambda) * g
}

#[allow(clippy::too_many_arguments)]
fn mis_weight(
    base: &IntegratorBase,
    light_vertices: &mut [Vertex],
    camera_vertices: &mut [Vertex],
    sampled: Option<&Vertex>,
    s: i32,
    t: i32,
    light_sampler: &LightSampler,
    light_tracing_splat_scale: Float,
) -> Float {
    if s + t == 2 {
        return 1.0;
    }
    let remap0 = |f: Float| -> Float {
        if f != 0.0 {
            f
        } else {
            1.0
        }
    };

    // We need pt (the t-1 camera vertex) and qs (the s-1 light vertex).
    // For ScopedAssignment semantics, save originals and restore at end.
    let s_idx = (s - 1) as isize;
    let t_idx = (t - 1) as isize;
    let s_minus = (s - 2) as isize;
    let t_minus = (t - 2) as isize;

    let mut snap = Snapshot {
        qs: None,
        pt: None,
        qs_minus_pdf_rev: None,
        pt_minus_pdf_rev: None,
        pt_delta: None,
        qs_delta: None,
        pt_pdf_rev: None,
        qs_pdf_rev: None,
    };

    // Apply sampled vertex for s==1 or t==1.
    if s == 1 {
        if let (Some(sampled), true) = (sampled, s_idx >= 0) {
            snap.qs = Some(light_vertices[s_idx as usize].clone());
            light_vertices[s_idx as usize] = sampled.clone();
        }
    } else if t == 1 {
        if let (Some(sampled), true) = (sampled, t_idx >= 0) {
            snap.pt = Some(camera_vertices[t_idx as usize].clone());
            camera_vertices[t_idx as usize] = sampled.clone();
        }
    }

    // Mark connection vertices non-degenerate
    if t_idx >= 0 {
        snap.pt_delta = Some(camera_vertices[t_idx as usize].delta);
        camera_vertices[t_idx as usize].delta = false;
    }
    if s_idx >= 0 {
        snap.qs_delta = Some(light_vertices[s_idx as usize].delta);
        light_vertices[s_idx as usize].delta = false;
    }

    // Update pdf_rev of pt (= cameraVertices[t-1])
    if t_idx >= 0 {
        let new_pdf_rev = if s > 0 {
            // qs->PDF(integrator, qsMinus, *pt)
            let qs = &light_vertices[s_idx as usize];
            let qs_minus = if s_minus >= 0 {
                Some(&light_vertices[s_minus as usize])
            } else {
                None
            };
            qs.pdf(base, qs_minus, &camera_vertices[t_idx as usize])
        } else {
            // pt->PDFLightOrigin(infiniteLights, *ptMinus, lightSampler)
            let pt = &camera_vertices[t_idx as usize];
            let pt_minus = if t_minus >= 0 {
                &camera_vertices[t_minus as usize]
            } else {
                return restore_snapshot(
                    snap,
                    light_vertices,
                    camera_vertices,
                    s_idx,
                    t_idx,
                    s_minus,
                    t_minus,
                    1.0,
                );
            };
            pt.pdf_light_origin(&base.infinite_lights, pt_minus, light_sampler)
        };
        snap.pt_pdf_rev = Some(camera_vertices[t_idx as usize].pdf_rev);
        camera_vertices[t_idx as usize].pdf_rev = new_pdf_rev;
    }

    // Update pdf_rev of ptMinus
    if t_minus >= 0 {
        let new_pdf_rev = if s > 0 {
            let pt = &camera_vertices[t_idx as usize];
            let qs = if s_idx >= 0 {
                Some(&light_vertices[s_idx as usize])
            } else {
                None
            };
            pt.pdf(base, qs, &camera_vertices[t_minus as usize])
        } else {
            let pt = &camera_vertices[t_idx as usize];
            pt.pdf_light(base, &camera_vertices[t_minus as usize])
        };
        snap.pt_minus_pdf_rev = Some(camera_vertices[t_minus as usize].pdf_rev);
        camera_vertices[t_minus as usize].pdf_rev = new_pdf_rev;
    }

    // Update pdf_rev of qs and qsMinus
    if s_idx >= 0 {
        let pt = &camera_vertices[t_idx as usize];
        let pt_minus = if t_minus >= 0 {
            Some(&camera_vertices[t_minus as usize])
        } else {
            None
        };
        let new_pdf_rev = pt.pdf(base, pt_minus, &light_vertices[s_idx as usize]);
        snap.qs_pdf_rev = Some(light_vertices[s_idx as usize].pdf_rev);
        light_vertices[s_idx as usize].pdf_rev = new_pdf_rev;
    }
    if s_minus >= 0 {
        let qs = &light_vertices[s_idx as usize];
        let pt = &camera_vertices[t_idx as usize];
        let new_pdf_rev = qs.pdf(base, Some(pt), &light_vertices[s_minus as usize]);
        snap.qs_minus_pdf_rev = Some(light_vertices[s_minus as usize].pdf_rev);
        light_vertices[s_minus as usize].pdf_rev = new_pdf_rev;
    }

    // pbrt-v4 (integrators.cpp:2180-2185), see https://github.com/mmp/pbrt-v4/issues/347.
    // `Film::FullResolution()` is the rendered image's full resolution while
    // `Film::PixelBounds()` may be smaller when a crop window is in effect. The
    // light-tracing t=1 path is splatted onto the (crop-window) film, so the MIS
    // weight needs to compensate for the area ratio so that strategies remain
    // commensurable.
    let mut sum_ri = 0.0;
    let mut ri: Float = 1.0;
    // Camera-subpath hypothetical strategies
    for i in (1..t).rev() {
        let ip = i as usize;
        ri *= remap0(camera_vertices[ip].pdf_rev) / remap0(camera_vertices[ip].pdf_fwd);
        let r_use = if i == 1 {
            ri / light_tracing_splat_scale
        } else {
            ri
        };
        if !camera_vertices[ip].delta && !camera_vertices[ip - 1].delta {
            sum_ri += r_use;
        }
    }
    ri = 1.0;
    for i in (0..s).rev() {
        let ip = i as usize;
        ri *= remap0(light_vertices[ip].pdf_rev) / remap0(light_vertices[ip].pdf_fwd);
        let delta_light_vertex = if i > 0 {
            light_vertices[ip - 1].delta
        } else {
            light_vertices[0].is_delta_light()
        };
        if !light_vertices[ip].delta && !delta_light_vertex {
            sum_ri += ri;
        }
    }
    if t == 1 {
        sum_ri /= light_tracing_splat_scale;
    }
    let result = 1.0 / (1.0 + sum_ri);

    restore_snapshot(
        snap,
        light_vertices,
        camera_vertices,
        s_idx,
        t_idx,
        s_minus,
        t_minus,
        result,
    )
}

fn restore_snapshot(
    snap: Snapshot,
    light_vertices: &mut [Vertex],
    camera_vertices: &mut [Vertex],
    s_idx: isize,
    t_idx: isize,
    s_minus: isize,
    t_minus: isize,
    return_value: Float,
) -> Float {
    if let Some(qs) = snap.qs {
        if s_idx >= 0 {
            light_vertices[s_idx as usize] = qs;
        }
    }
    if let Some(pt) = snap.pt {
        if t_idx >= 0 {
            camera_vertices[t_idx as usize] = pt;
        }
    }
    if let Some(v) = snap.pt_delta {
        if t_idx >= 0 {
            camera_vertices[t_idx as usize].delta = v;
        }
    }
    if let Some(v) = snap.qs_delta {
        if s_idx >= 0 {
            light_vertices[s_idx as usize].delta = v;
        }
    }
    if let Some(v) = snap.pt_pdf_rev {
        if t_idx >= 0 {
            camera_vertices[t_idx as usize].pdf_rev = v;
        }
    }
    if let Some(v) = snap.pt_minus_pdf_rev {
        if t_minus >= 0 {
            camera_vertices[t_minus as usize].pdf_rev = v;
        }
    }
    if let Some(v) = snap.qs_pdf_rev {
        if s_idx >= 0 {
            light_vertices[s_idx as usize].pdf_rev = v;
        }
    }
    if let Some(v) = snap.qs_minus_pdf_rev {
        if s_minus >= 0 {
            light_vertices[s_minus as usize].pdf_rev = v;
        }
    }
    return_value
}

struct Snapshot {
    qs: Option<Vertex>,
    pt: Option<Vertex>,
    qs_minus_pdf_rev: Option<Float>,
    pt_minus_pdf_rev: Option<Float>,
    pt_delta: Option<bool>,
    qs_delta: Option<bool>,
    pt_pdf_rev: Option<Float>,
    qs_pdf_rev: Option<Float>,
}

// ============================================================================
// ConnectBDPT
// ============================================================================

#[allow(clippy::too_many_arguments)]
pub fn connect_bdpt(
    base: &IntegratorBase,
    camera: &Arc<Camera>,
    lambda: &mut SampledWavelengths,
    light_vertices: &mut [Vertex],
    camera_vertices: &mut [Vertex],
    s: i32,
    t: i32,
    light_sampler: &LightSampler,
    sampler: &mut Sampler,
    light_tracing_splat_scale: Float,
) -> (SampledSpectrum, Option<Point2f>) {
    let mut l_pkt = SampledSpectrum::zero();
    let mut p_raster: Option<Point2f> = None;

    // Ignore invalid connections related to infinite area lights
    if t > 1 && s != 0 && matches!(camera_vertices[(t - 1) as usize].vtype, VertexType::Light) {
        return (SampledSpectrum::zero(), None);
    }

    let mut sampled_opt: Option<Vertex> = None;
    if s == 0 {
        let pt = &camera_vertices[(t - 1) as usize];
        if pt.is_light() {
            l_pkt = pt.le(
                &base.infinite_lights,
                &camera_vertices[(t - 2) as usize],
                lambda,
            ) * pt.beta;
        }
    } else if t == 1 {
        // Sample camera, connect to light subpath
        let qs = &light_vertices[(s - 1) as usize];
        if qs.is_connectible() {
            let u = sampler.get_2d();
            let cs = camera.sample_wi(&qs.get_interaction(), &u, lambda);
            if let Some(cs) = cs {
                if cs.pdf != 0.0 {
                    p_raster = Some(cs.p_raster);
                    let wi_spec = cs.wi_spec.sample(lambda);
                    // pLens interaction (built from VisibilityTester::p1)
                    let p_lens_intr = cs.visibility.p1.clone();
                    let sampled = Vertex::create_camera_from_interaction(
                        Arc::clone(camera),
                        &p_lens_intr,
                        wi_spec / cs.pdf,
                    );
                    let mut l = qs.beta * qs.f(&sampled, TransportMode::Importance) * sampled.beta;
                    if qs.is_on_surface() {
                        l *= Float::abs(Vector3f::dot(&cs.wi, &Vector3f::from(qs.ns())));
                    }
                    if !l.is_black() && cs.visibility.unoccluded(base) {
                        // pbrt-v4 (integrators.cpp:2356-2363): scale the
                        // splatted contribution by FullRes^2 / PixelBounds.Area()
                        // so that crop-window renders remain commensurable with
                        // the other MIS strategies. See issue #347.
                        l *= light_tracing_splat_scale;
                        l_pkt = l;
                        sampled_opt = Some(sampled);
                    }
                }
            }
        }
    } else if s == 1 {
        // Sample light, connect to camera subpath
        let pt = &camera_vertices[(t - 1) as usize];
        if pt.is_connectible() {
            let sampled_light =
                light_sampler.sample(&LightSampleContext::default(), sampler.get_1d());
            if let Some(sl) = sampled_light {
                let light = sl.light;
                let p_l = sl.p;
                let mut ctx = if pt.is_on_surface() {
                    let si = pt.si.as_ref().unwrap();
                    let mut c = LightSampleContext::from(&Interaction::Surface(si.clone()));
                    let flags = pt.bsdf.as_ref().unwrap().flags();
                    if is_reflective(flags) && !is_transmissive(flags) {
                        c.p = offset_ray_origin(&si.p, &si.p_error, &si.n, &si.wo);
                    } else if is_transmissive(flags) && !is_reflective(flags) {
                        c.p = offset_ray_origin(&si.p, &si.p_error, &si.n, &(-si.wo));
                    }
                    c
                } else {
                    LightSampleContext::from(&pt.get_interaction())
                };
                let _ = &mut ctx;
                let light_weight = light
                    .as_ref()
                    .sample_li(&ctx, sampler.get_2d(), lambda, false);
                if let Some(lw) = light_weight {
                    if !lw.l.is_black() && lw.pdf > 0.0 {
                        let ei = EndpointInteraction::from_light_interaction(
                            Arc::clone(&light),
                            &lw.p_light,
                        );
                        let sampled =
                            Vertex::create_light_from_endpoint(ei, lw.l / (lw.pdf * p_l), 0.0);
                        let mut sampled = sampled;
                        sampled.pdf_fwd =
                            sampled.pdf_light_origin(&base.infinite_lights, pt, light_sampler);
                        let mut l =
                            pt.beta * pt.f(&sampled, TransportMode::Radiance) * sampled.beta;
                        if pt.is_on_surface() {
                            l *= Float::abs(Vector3f::dot(&lw.wi, &Vector3f::from(pt.ns())));
                        }
                        if !l.is_black() && base.unoccluded(&pt.get_interaction(), &lw.p_light) {
                            l_pkt = l;
                            sampled_opt = Some(sampled);
                        }
                    }
                }
            }
        }
    } else {
        // General case
        let qs = &light_vertices[(s - 1) as usize];
        let pt = &camera_vertices[(t - 1) as usize];
        if qs.is_connectible() && pt.is_connectible() {
            let mut l = qs.beta
                * qs.f(&pt, TransportMode::Importance)
                * pt.f(&qs, TransportMode::Radiance)
                * pt.beta;
            if !l.is_black() {
                l *= geometry_term(base, &qs, &pt, lambda);
            }
            l_pkt = l;
        }
    }

    if l_pkt.is_black() {
        return (SampledSpectrum::zero(), p_raster);
    }
    let mis = mis_weight(
        base,
        light_vertices,
        camera_vertices,
        sampled_opt.as_ref(),
        s,
        t,
        light_sampler,
        light_tracing_splat_scale,
    );
    (l_pkt * mis, p_raster)
}

// ============================================================================
// BDPTIntegrator
// ============================================================================

pub struct BDPTIntegrator {
    base: RayIntegratorBase,
    max_depth: i32,
    visualize_strategies: bool,
    visualize_weights: bool,
    regularize: bool,
    light_sample_strategy: String,
    light_sampler: Option<LightSampler>,
    light_tracing_splat_scale: Float,
}

impl BDPTIntegrator {
    pub fn new(
        scene: &Scene,
        sampler: &Arc<RwLock<Sampler>>,
        camera: &Arc<Camera>,
        max_depth: i32,
        visualize_strategies: bool,
        visualize_weights: bool,
        regularize: bool,
        pixel_bounds: &Bounds2i,
        light_sample_strategy: &str,
        light_tracing_splat_scale: Float,
    ) -> Self {
        BDPTIntegrator {
            base: RayIntegratorBase::new(scene, camera, sampler, pixel_bounds),
            max_depth,
            visualize_strategies,
            visualize_weights,
            regularize,
            light_sample_strategy: light_sample_strategy.to_string(),
            light_sampler: None,
            light_tracing_splat_scale,
        }
    }
}

impl Integrator for BDPTIntegrator {
    fn render(&mut self) {
        RayIntegratorBase::render(self);
    }
    fn get_camera(&self) -> Arc<Camera> {
        self.base.camera.clone()
    }
}

impl RayIntegrator for BDPTIntegrator {
    fn preprocess(&mut self, _sampler: &mut Sampler) {
        match LightSampler::create(&self.light_sample_strategy, &self.base.base) {
            Ok(ls) => self.light_sampler = Some(ls),
            Err(e) => log::warn!("BDPTIntegrator: {:?}", e),
        }
    }

    fn li(
        &self,
        r: &RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        scratch_buffer: &mut MemoryArena,
        _visible_surface: Option<&mut VisibleSurface>,
    ) -> SampledSpectrum {
        let light_sampler = match self.light_sampler.as_ref() {
            Some(ls) => ls,
            None => return SampledSpectrum::zero(),
        };

        let mut camera_vertices: Vec<Vertex> = Vec::with_capacity(self.max_depth as usize + 2);
        let n_camera = generate_camera_subpath(
            &self.base.base,
            &self.base.camera,
            r,
            lambda,
            sampler,
            scratch_buffer,
            self.max_depth as usize + 2,
            &mut camera_vertices,
            self.regularize,
        );

        let mut light_vertices: Vec<Vertex> = Vec::with_capacity(self.max_depth as usize + 1);
        let time = if !camera_vertices.is_empty() {
            camera_vertices[0].time()
        } else {
            0.0
        };
        let n_light = generate_light_subpath(
            &self.base.base,
            &self.base.camera,
            lambda,
            sampler,
            scratch_buffer,
            self.max_depth as usize + 1,
            time,
            light_sampler,
            &mut light_vertices,
            self.regularize,
        );

        let _ = (self.visualize_strategies, self.visualize_weights);

        let mut l = SampledSpectrum::zero();
        for t in 1..=n_camera as i32 {
            for s in 0..=n_light as i32 {
                let depth = t + s - 2;
                if (s == 1 && t == 1) || depth < 0 || depth > self.max_depth {
                    continue;
                }
                let (l_path, p_film_new) = connect_bdpt(
                    &self.base.base,
                    &self.base.camera,
                    lambda,
                    &mut light_vertices,
                    &mut camera_vertices,
                    s,
                    t,
                    light_sampler,
                    sampler,
                    self.light_tracing_splat_scale,
                );
                if l_path.is_black() {
                    continue;
                }
                if t != 1 {
                    l += l_path;
                } else if let Some(p_film) = p_film_new {
                    let film = self.base.camera.get_film();
                    film.read().unwrap().add_splat_packet(
                        &Vector2f::new(p_film.x, p_film.y),
                        &l_path,
                        lambda,
                    );
                }
            }
        }
        l
    }

    fn get_sampler(&self) -> Arc<RwLock<Sampler>> {
        Arc::clone(&self.base.sampler)
    }

    fn get_pixel_bounds(&self) -> Bounds2i {
        self.base.pixel_bounds
    }
}

crate::impl_image_tile_integrator_via_ray!(BDPTIntegrator);

unsafe impl Sync for BDPTIntegrator {}

pub fn create_bdpt_integrator(
    params: &ParameterDictionary,
    sampler: &Arc<RwLock<Sampler>>,
    camera: &Arc<Camera>,
    scene: &Scene,
) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
    let max_depth = params.get_one_int("maxdepth", 5);
    let visualize_strategies = params.get_one_bool("visualizestrategies", false);
    let visualize_weights = params.get_one_bool("visualizeweights", false);
    let regularize = params.get_one_bool("regularize", false);
    let light_strategy = {
        let strategy = params.get_one_string("lightsampler", "");
        if strategy.is_empty() {
            params.get_one_string("lightsamplestrategy", "power")
        } else {
            strategy
        }
    };
    let (pixel_bounds, light_tracing_splat_scale) = {
        let film = camera.get_film();
        let film = film.read().unwrap();
        (
            film.pixel_bounds(),
            compute_light_tracing_splat_scale(&film),
        )
    };
    Ok(Arc::new(RwLock::new(BDPTIntegrator::new(
        scene,
        sampler,
        camera,
        max_depth,
        visualize_strategies,
        visualize_weights,
        regularize,
        &pixel_bounds,
        &light_strategy,
        light_tracing_splat_scale,
    ))))
}

pub fn compute_light_tracing_splat_scale(film: &Film) -> Float {
    let full_resolution = film.full_resolution();
    let pixel_bounds = film.pixel_bounds();
    let full_area = full_resolution.x as Float * full_resolution.y as Float;
    let pixel_area = pixel_bounds.area() as Float;
    if pixel_area > 0.0 {
        full_area / pixel_area
    } else {
        1.0
    }
}
