use crate::interaction::*;
use crate::paramdict::*;

use crate::shapes::*;
use crate::util::base::*;
use crate::util::efloat::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::sampling::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

const MACHINE_EPSILON: Float = Float::EPSILON * 0.5;
const GAMMA5: Float = (5.0 * MACHINE_EPSILON) / (1.0 - (5.0 * MACHINE_EPSILON));

#[derive(Debug, PartialEq, Clone)]
pub struct Sphere {
    pub base: BaseShape,
    pub radius: Float,
    pub z_min: Float,
    pub z_max: Float,
    pub theta_min: Float,
    pub theta_max: Float,
    pub phi_max: Float,
}

impl Sphere {
    pub fn new(
        o2w: &Transform,
        w2o: &Transform,
        reverse_orientation: bool,
        radius: Float,
        z_min: Float,
        z_max: Float,
        phi_max: Float,
    ) -> Self {
        let z_min = Float::clamp(Float::min(z_min, z_max), -radius, radius);
        let z_max = Float::clamp(Float::max(z_min, z_max), -radius, radius);
        let theta_min = Float::acos(Float::clamp(z_min / radius, -1.0, 1.0));
        let theta_max = Float::acos(Float::clamp(z_max / radius, -1.0, 1.0));
        let phi_max = radians(Float::clamp(phi_max, 0.0, 360.0));
        Sphere {
            base: BaseShape::new(o2w, w2o, reverse_orientation),
            radius,
            z_min,
            z_max,
            theta_min,
            theta_max,
            phi_max,
        }
    }

    pub fn create(
        o2w: &Transform,
        w2o: &Transform,
        reverse_orientation: bool,
        params: &ParameterDictionary,
    ) -> Result<Self, PbrtError> {
        let radius = params.get_one_float("radius", 1.0);
        let zmin = params.get_one_float("zmin", -radius);
        let zmax = params.get_one_float("zmax", radius);
        let phimax = params.get_one_float("phimax", 360.0);

        Ok(Self::new(
            o2w,
            w2o,
            reverse_orientation,
            radius,
            zmin,
            zmax,
            phimax,
        ))
    }
}

impl Sphere {
    pub fn object_bound(&self) -> Bounds3f {
        let radius = self.radius * 1.001;
        let diff = radius - self.radius;
        return Bounds3f::new(
            &Vector3f::new(-radius, -radius, self.z_min - diff),
            &Vector3f::new(radius, radius, self.z_max + diff),
        );
    }

    pub fn world_bound(&self) -> Bounds3f {
        return self
            .base
            .object_to_world
            .transform_bounds(&self.object_bound());
    }

    /// pbrt-v4 `Sphere::NormalBounds` (shapes.h:134). A sphere's
    /// surface normals span the whole sphere.
    pub fn normal_bounds(&self) -> DirectionCone {
        DirectionCone::entire_sphere()
    }

    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let (ray, o_err, d_err) = self.base.world_to_object.transform_ray(r);
        let ox = EFloat::from((ray.o.x, o_err.x));
        let oy = EFloat::from((ray.o.y, o_err.y));
        let oz = EFloat::from((ray.o.z, o_err.z));
        let dx = EFloat::from((ray.d.x, d_err.x));
        let dy = EFloat::from((ray.d.y, d_err.y));
        let dz = EFloat::from((ray.d.z, d_err.z));
        let rad = EFloat::from((self.radius, 0.0));
        let a = dx * dx + dy * dy + dz * dz;
        let b = (dx * ox + dy * oy + dz * oz) * 2.0;
        let c = ox * ox + oy * oy + oz * oz - rad * rad;

        let (t0, t1) = EFloat::quadratic(a, b, c)?;
        if t0.v.is_infinite() || t1.v.is_infinite() {
            return None;
        }

        assert!(t0.v <= t1.v);
        // Check quadric shape _t0_ and _t1_ for nearest intersection
        if t0.upper_bound() > t_max || t1.lower_bound() <= 0.0 {
            return None;
        }

        let mut t_shape_hit = t0;
        if t_shape_hit.lower_bound() <= 0.0 {
            t_shape_hit = t1;
            if t_max < t_shape_hit.upper_bound() {
                return None;
            }
        }

        let mut p_hit = ray.o + ray.d * Float::from(t_shape_hit);
        p_hit *= self.radius / Vector3f::distance(&p_hit, &Vector3f::zero());
        if p_hit.x == 0.0 && p_hit.y == 0.0 {
            p_hit.x = 1e-5 * self.radius;
        }
        let mut phi = Float::atan2(p_hit.y, p_hit.x);
        if phi < 0.0 {
            phi += 2.0 * PI;
        }
        if (self.z_min > -self.radius && p_hit.z < self.z_min)
            || (self.z_max < self.radius && p_hit.z > self.z_max)
            || (phi > self.phi_max)
        {
            if t_shape_hit == t1 {
                return None;
            }
            if t1.upper_bound() > t_max {
                return None;
            }
            t_shape_hit = t1;
            p_hit = ray.o + ray.d * Float::from(t_shape_hit);
            p_hit *= self.radius / Vector3f::distance(&p_hit, &Vector3f::zero());
            if p_hit.x == 0.0 && p_hit.y == 0.0 {
                p_hit.x = 1e-5 * self.radius;
            }
            phi = Float::atan2(p_hit.y, p_hit.x);
            if phi < 0.0 {
                phi += PI;
            }
            if (self.z_min > -self.radius && p_hit.z < self.z_min)
                || (self.z_max < self.radius && p_hit.z > self.z_max)
                || (phi > self.phi_max)
            {
                return None;
            }
        }

        let radius = self.radius;
        let theta_min = self.theta_min;
        let theta_max = self.theta_max;
        let dtheta = theta_max - theta_min;
        let phi_max = self.phi_max;

        let u = phi / phi_max;
        let theta = Float::acos(Float::clamp(p_hit.z / radius, -1.0, 1.0));
        let v = (theta - theta_min) / dtheta;
        assert!(u >= 0.0);
        assert!(v >= 0.0);

        let z_radius = Float::sqrt(p_hit.x * p_hit.x + p_hit.y * p_hit.y);
        let inv_z_radius = Float::recip(z_radius);
        let cos_phi = p_hit.x * inv_z_radius;
        let sin_phi = p_hit.y * inv_z_radius;
        let dpdu = Vector3f::new(-phi_max * p_hit.y, self.phi_max * p_hit.x, 0.0);
        let dpdv = Vector3f::new(
            p_hit.z * cos_phi,
            p_hit.z * sin_phi,
            -radius * Float::sin(theta),
        ) * dtheta;

        let d2pduu = Vector3f::new(p_hit.x, p_hit.y, 0.0) * (-phi_max * phi_max);
        let d2pduv = Vector3f::new(-sin_phi, cos_phi, 0.0) * (p_hit.z * dtheta * phi_max);
        let d2pdvv = Vector3f::new(p_hit.x, p_hit.y, p_hit.z) * (-dtheta * dtheta);

        #[allow(non_snake_case)]
        let E = Vector3f::dot(&dpdu, &dpdu);
        #[allow(non_snake_case)]
        let F = Vector3f::dot(&dpdu, &dpdv);
        #[allow(non_snake_case)]
        let G = Vector3f::dot(&dpdv, &dpdv);

        let n = self.base.calc_normal(&dpdu, &dpdv);

        let e = Vector3f::dot(&n, &d2pduu);
        let f = Vector3f::dot(&n, &d2pduv);
        let g = Vector3f::dot(&n, &d2pdvv);

        let inv_egf2 = Float::recip(E * G - F * F);
        let dndu = dpdu * ((f * F - e * G) * inv_egf2) + dpdv * ((e * F - f * E) * inv_egf2);
        let dndv = dpdu * ((g * F - f * G) * inv_egf2) + dpdv * ((f * F - g * E) * inv_egf2);

        let p_error = GAMMA5 * p_hit.abs();
        let mut isect = SurfaceInteraction::new(
            &p_hit,
            &p_error,
            &Point2f::new(u, v),
            &(-ray.d),
            &n,
            &dpdu,
            &dpdv,
            &dndu,
            &dndv,
            ray.time,
            0,
        );
        isect = self
            .base
            .object_to_world
            .transform_surface_interaction(&isect);
        return Some(ShapeIntersection::new(isect, t_shape_hit.into()));
    }

    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        let (ray, o_err, d_err) = self.base.world_to_object.transform_ray(r);
        let ox = EFloat::from((ray.o.x, o_err.x));
        let oy = EFloat::from((ray.o.y, o_err.y));
        let oz = EFloat::from((ray.o.z, o_err.z));
        let dx = EFloat::from((ray.d.x, d_err.x));
        let dy = EFloat::from((ray.d.y, d_err.y));
        let dz = EFloat::from((ray.d.z, d_err.z));
        let rad = EFloat::from((self.radius, 0.0));
        let a = dx * dx + dy * dy + dz * dz;
        let b = (dx * ox + dy * oy + dz * oz) * 2.0;
        let c = ox * ox + oy * oy + oz * oz - rad * rad;

        if let Some((t0, t1)) = EFloat::quadratic(a, b, c) {
            if t0.v.is_infinite() || t1.v.is_infinite() {
                return false;
            }

            if t0.upper_bound() > t_max || t1.lower_bound() <= 0.0 {
                return false;
            }
            let mut t_shape_hit = t0;
            if t_shape_hit.lower_bound() <= 0.0 {
                t_shape_hit = t1;
                if t_max < t_shape_hit.upper_bound() {
                    return false;
                }
            }
            let mut p_hit = ray.o + ray.d * Float::from(t_shape_hit);
            p_hit *= self.radius / Vector3f::distance(&p_hit, &Vector3f::zero());
            if p_hit.x == 0.0 && p_hit.y == 0.0 {
                p_hit.x = 1e-5 * self.radius;
            }
            let mut phi = Float::atan2(p_hit.y, p_hit.x);
            if phi < 0.0 {
                phi += 2.0 * PI; //Float
            }
            if (self.z_min > -self.radius && p_hit.z < self.z_min)
                || (self.z_max < self.radius && p_hit.z > self.z_max)
                || phi > self.phi_max
            {
                if t_shape_hit == t1 {
                    return false;
                }
                if t1.upper_bound() > t_max {
                    return false;
                }
                t_shape_hit = t1;
                p_hit = ray.o + ray.d * Float::from(t_shape_hit);
                p_hit *= self.radius / Vector3f::distance(&p_hit, &Vector3f::zero());
                if p_hit.x == 0.0 && p_hit.y == 0.0 {
                    p_hit.x = 1e-5 * self.radius;
                }
                phi = Float::atan2(p_hit.y, p_hit.x);
                if phi < 0.0 {
                    phi += 2.0 * PI; //Float
                }
                if (self.z_min > -self.radius && p_hit.z < self.z_min)
                    || (self.z_max < self.radius && p_hit.z > self.z_max)
                    || phi > self.phi_max
                {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    pub fn area(&self) -> Float {
        return self.phi_max * self.radius * (self.z_max - self.z_min);
    }

    pub fn sample(&self, u: &Point2f) -> Option<(Interaction, Float)> {
        // pbrt-v4 `Sphere::Sample(Point2f)` (shapes.cpp:38-58). The order is
        // (i) sample a unit vector and scale to radius, (ii) reproject to
        // sphere surface, (iii) compute the normal from the reprojected
        // point, (iv) compute (u, v), (v) return a ShapeSample with the
        // *SurfaceInteraction* (not a bare Interaction) so downstream
        // consumers (DiffuseAreaLight::SampleLe, image-textured area lights)
        // see the populated `uv` field.
        let mut p_obj = Point3f::zero() + self.radius * uniform_sample_sphere(u);
        p_obj *= self.radius / Point3f::distance(&p_obj, &Point3f::zero());
        let p_obj_error = GAMMA5 * p_obj.abs();
        let mut n = self
            .base
            .object_to_world
            .transform_normal(&p_obj)
            .normalize();
        if self.base.reverse_orientation {
            n *= -1.0;
        }
        // v4 shapes.cpp:50-54: (u, v) parameterization from the reprojected
        // object-space point.
        let theta = Float::acos(Float::clamp(p_obj.z / self.radius, -1.0, 1.0));
        let mut phi = Float::atan2(p_obj.y, p_obj.x);
        if phi < 0.0 {
            phi += 2.0 * PI;
        }
        let uv = Point2f::new(
            phi / self.phi_max,
            (theta - self.theta_min) / (self.theta_max - self.theta_min),
        );
        let (p, p_error) = self
            .base
            .object_to_world
            .transform_point_with_abs_error(&p_obj, &p_obj_error);
        let mut si = SurfaceInteraction::default();
        si.p = p;
        si.p_error = p_error;
        si.n = n;
        si.uv = uv;
        si.shading.n = n;
        let pdf = Float::recip(self.area());
        Some((Interaction::Surface(si), pdf))
    }

    pub fn sample_from(&self, inter: &Interaction, u: &Point2f) -> Option<(Interaction, Float)> {
        let p_center = self.base.object_to_world.transform_point(&Point3f::zero());
        // Sample uniformly on sphere if $\pt{}$ is inside it
        let p_origin = offset_ray_origin(
            &inter.get_p(),
            &inter.get_p_error(),
            &inter.get_n(),
            &(p_center - inter.get_p()),
        );
        let radius = self.radius;
        if Vector3f::distance_squared(&p_origin, &p_center) <= radius * radius {
            let (intr, pdf) = self.sample(u)?;
            let wi = intr.get_p() - inter.get_p();
            if wi.length_squared() == 0.0 {
                return None;
            } else {
                // Convert from area measure returned by Sample() call above to
                // solid angle measure.
                let wi = wi.normalize();
                let pdf = pdf * Vector3f::distance_squared(&intr.get_p(), &inter.get_p())
                    / Vector3f::abs_dot(&intr.get_n(), &-wi);
                if pdf <= 0.0 || pdf.is_infinite() {
                    return None;
                }
                return Some((intr, pdf));
            }
        }
        // Sample sphere uniformly inside subtended cone

        // Compute coordinate system for sphere sampling
        //
        let dc = Vector3f::distance(&inter.get_p(), &p_center);
        let inv_dc = 1.0 / dc;
        let wc = (p_center - inter.get_p()) * inv_dc;
        let (wc_x, wc_y) = coordinate_system(&wc);

        // Compute $\theta$ and $\phi$ values for sample in cone
        let sin_theta_max = radius * inv_dc;
        let sin_theta_max2 = sin_theta_max * sin_theta_max;
        let inv_sin_theta_max = 1.0 / sin_theta_max;
        let cos_theta_max = Float::sqrt(Float::max(0.0, 1.0 - sin_theta_max2));
        let mut one_minus_cos_theta_max = 1.0 - cos_theta_max;

        let mut cos_theta = (cos_theta_max - 1.0) * u[0] + 1.0;
        let mut sin_theta2 = 1.0 - cos_theta * cos_theta;

        if sin_theta_max2 < 0.00068523
        //sin^2(1.5 deg)
        {
            /* Fall back to a Taylor series expansion for small angles, where
            the standard approach suffers from severe cancellation errors */
            sin_theta2 = Float::max(0.0, sin_theta_max2 * u[0]);
            cos_theta = Float::sqrt(1.0 - sin_theta2);
            one_minus_cos_theta_max = sin_theta_max2 * 0.5;
        }
        let pdf = 1.0 / (2.0 * PI * one_minus_cos_theta_max);
        if pdf <= 0.0 || pdf.is_infinite() {
            return None;
        }

        // Compute angle $\alpha$ from center of sphere to sampled point on surface
        let cos_alpha = sin_theta2 * inv_sin_theta_max
            + cos_theta
                * Float::sqrt(Float::max(
                    0.0,
                    1.0 - sin_theta2 * inv_sin_theta_max * inv_sin_theta_max,
                ));
        let sin_alpha = Float::sqrt(Float::max(0.0, 1.0 - cos_alpha * cos_alpha));
        let phi = u[1] * 2.0 * PI;

        // Compute surface normal and sampled point on sphere
        let n_world = spherical_direction_axes(sin_alpha, cos_alpha, phi, &-wc_x, &-wc_y, &-wc);
        let p_world = p_center + radius * Point3f::from(n_world);

        // Return _Interaction_ for sampled point on sphere
        let p = p_world;
        let p_error = GAMMA5 * Vector3::abs(&p_world);
        let mut n = Normal3f::from(n_world);
        if self.base.reverse_orientation {
            n *= -1.0;
        }
        let it = Interaction::from_surface_sample(&p, &p_error, &n);
        return Some((it, pdf));
    }

    pub fn pdf(&self, _inter: &Interaction) -> Float {
        Float::recip(self.area())
    }

    /// v4 `Sphere::PDF(ShapeSampleContext, wi)` (shapes.cpp):
    /// matches the two-branch sampling used by `sample_from`: when the
    /// reference point is inside the sphere we convert the area PDF to
    /// solid-angle measure via `r²/cosθ/area`, but when it is outside
    /// the sample is drawn uniformly within the subtended cone so the
    /// PDF is `1/(2π(1 − cosθ_max))`. Returning the area-converted PDF
    /// in the outside case breaks the MIS denominator when the path
    /// hits the area light by chance (phase / BSDF sampling), since
    /// `sample_li` and `pdf_li` would disagree.
    pub fn pdf_from(&self, inter: &Interaction, wi: &Vector3f) -> Float {
        let p_center = self.base.object_to_world.transform_point(&Point3f::zero());
        let p_origin = offset_ray_origin(
            &inter.get_p(),
            &inter.get_p_error(),
            &inter.get_n(),
            &(p_center - inter.get_p()),
        );
        let radius2 = self.radius * self.radius;
        if Vector3f::distance_squared(&p_origin, &p_center) <= radius2 {
            // Inside the sphere: area sampling, convert to solid-angle measure.
            let ray = inter.spawn_ray(wi);
            if let Some(si) = self.intersect(&ray, Float::INFINITY) {
                let isect_light = si.intr;
                assert!(isect_light.n.length() > 0.0);
                let pdf = Vector3f::distance_squared(&inter.get_p(), &isect_light.p)
                    / (Vector3f::abs_dot(&isect_light.n, &(-*wi)) * self.area());
                if pdf.is_infinite() {
                    return 0.0;
                }
                return pdf;
            }
            return 0.0;
        }

        // Outside the sphere: matches the cone-uniform `sample_from` branch.
        let dist2 = Vector3f::distance_squared(&inter.get_p(), &p_center);
        let sin2_theta_max = radius2 / dist2;
        let cos_theta_max = Float::sqrt(Float::max(0.0, 1.0 - sin2_theta_max));
        let one_minus_cos_theta_max = if sin2_theta_max < 0.00068523 {
            // sin²(1.5°) – Taylor expansion for accuracy at small angles
            sin2_theta_max * 0.5
        } else {
            1.0 - cos_theta_max
        };
        if one_minus_cos_theta_max <= 0.0 {
            return 0.0;
        }
        1.0 / (2.0 * PI * one_minus_cos_theta_max)
    }

    pub fn solid_angle(&self, p: &Point3f, n_samples: i32) -> Float {
        use crate::util::lowdiscrepancy::*;
        let mut it = BaseInteraction::default();
        it.p = *p;
        it.wo = Vector3f::new(0.0, 0.0, 1.0);
        let inter = Interaction::from(it);
        let mut solid_angle = 0.0;
        for i in 0..n_samples {
            let u = Point2f::new(radical_inverse(0, i as u64), radical_inverse(1, i as u64));
            if let Some((p_shape, pdf)) = self.sample_from(&inter, &u) {
                let r = Ray::new(p, &(p_shape.get_p() - *p), 0.999, 0.0);
                if !self.intersect_p(&r, 0.999) {
                    solid_angle += 1.0 / pdf
                }
            }
        }
        return solid_angle / n_samples as Float;
    }
}

#[inline]
fn radians(x: Float) -> Float {
    return x * (PI / 180.0);
}
pub fn create_sphere_shape(
    o2w: &Transform,
    w2o: &Transform,
    reverse_orientation: bool,
    params: &ParameterDictionary,
) -> Result<Sphere, PbrtError> {
    Sphere::create(o2w, w2o, reverse_orientation, params)
}

//------------------
