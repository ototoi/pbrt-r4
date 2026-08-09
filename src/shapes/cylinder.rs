use crate::interaction::*;
use crate::paramdict::*;

use crate::shapes::*;
use crate::util::base::*;
use crate::util::efloat::*;
use crate::util::error::*;
use crate::util::geometry::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

const MACHINE_EPSILON: Float = Float::EPSILON * 0.5;
const GAMMA3: Float = (3.0 * MACHINE_EPSILON) / (1.0 - (3.0 * MACHINE_EPSILON));

#[inline]
fn radians(x: Float) -> Float {
    return x * (PI / 180.0);
}

pub struct Cylinder {
    pub base: BaseShape,
    pub radius: Float,
    pub z_min: Float,
    pub z_max: Float,
    pub phi_max: Float,
}

impl Cylinder {
    pub fn new(
        o2w: &Transform,
        w2o: &Transform,
        reverse_orientation: bool,
        radius: Float,
        z_min: Float,
        z_max: Float,
        phi_max: Float,
    ) -> Self {
        let phi_max = radians(Float::clamp(phi_max, 0.0, 360.0));
        Cylinder {
            base: BaseShape::new(o2w, w2o, reverse_orientation),
            radius,
            z_min,
            z_max,
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
        let mut zmin = params.get_one_float("zmin", -1.0);
        let mut zmax = params.get_one_float("zmax", 1.0);
        let phimax = params.get_one_float("phimax", 360.0);

        if zmin > zmax {
            std::mem::swap(&mut zmin, &mut zmax);
        }

        if radius == 0.0 {
            let msg = format!(
                "Unable to create cylinder shape: radius={}, zmin={}, zmax={}, phimax={}",
                radius, zmin, zmax, phimax
            );
            Err(PbrtError::error(&msg))
        } else {
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
}

impl Cylinder {
    pub fn object_bound(&self) -> Bounds3f {
        let radius = self.radius;
        let z_min = self.z_min;
        let z_max = self.z_max;
        return Bounds3f::new(
            &Point3f::new(-radius, -radius, z_min),
            &Point3f::new(radius, radius, z_max),
        );
    }
    pub fn world_bound(&self) -> Bounds3f {
        return self
            .base
            .object_to_world
            .transform_bounds(&self.object_bound());
    }

    pub fn normal_bounds(&self) -> DirectionCone {
        DirectionCone::entire_sphere()
    }
    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let (ray, o_err, d_err) = self.base.world_to_object.transform_ray(r);

        // Compute quadratic cylinder coefficients

        // Initialize _EFloat_ ray coordinate values
        let radius = self.radius;
        let z_min = self.z_min;
        let z_max = self.z_max;
        let phi_max = self.phi_max;

        let ox = EFloat::from((ray.o.x, o_err.x));
        let oy = EFloat::from((ray.o.y, o_err.y));
        let dx = EFloat::from((ray.d.x, d_err.x));
        let dy = EFloat::from((ray.d.y, d_err.y));
        let a = dx * dx + dy * dy;
        let b = (dx * ox + dy * oy) * 2.0;
        let c = ox * ox + oy * oy - EFloat::from(radius) * EFloat::from(radius);

        // Solve quadratic equation for _t_ values

        let (t0, t1) = EFloat::quadratic(a, b, c)?;
        if t0.v.is_infinite() || t1.v.is_infinite() {
            return None;
        }

        assert!(t0.v.is_finite());
        assert!(t1.v.is_finite());
        assert!(t0.v <= t1.v);
        // Check quadric shape _t0_ and _t1_ for nearest intersection
        if t0.upper_bound() > t_max || t1.lower_bound() <= 0.0 {
            return None;
        }

        // Compute cylinder inverse mapping
        let mut t_shape_hit = t0;
        if t_shape_hit.lower_bound() <= 0.0 {
            t_shape_hit = t1;
            if t_max < t_shape_hit.upper_bound() {
                return None;
            }
        }

        // Compute cylinder hit point and $\phi$
        let mut p_hit = ray.o + ray.d * Float::from(t_shape_hit);

        // Refine cylinder intersection point
        {
            let hit_rad = Float::sqrt(p_hit.x * p_hit.x + p_hit.y * p_hit.y);
            p_hit.x *= radius / hit_rad;
            p_hit.y *= radius / hit_rad;
        }
        let mut phi = Float::atan2(p_hit.y, p_hit.x);
        if phi < 0.0 {
            phi += 2.0 * PI;
        }

        // Test cylinder intersection against clipping parameters
        if p_hit.z < z_min || p_hit.z > z_max || phi > phi_max {
            if t_shape_hit == t1 {
                return None;
            }
            if t1.upper_bound() > t_max {
                return None;
            }
            t_shape_hit = t1;
            // Compute cylinder inverse mapping
            p_hit = ray.o + ray.d * Float::from(t_shape_hit);
            {
                let hit_rad = Float::sqrt(p_hit.x * p_hit.x + p_hit.y * p_hit.y);
                p_hit.x *= radius / hit_rad;
                p_hit.y *= radius / hit_rad;
            }

            phi = Float::atan2(p_hit.y, p_hit.x);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }
            if p_hit.z < z_min || p_hit.z > z_max || phi > phi_max {
                return None;
            }
        }

        // Find parametric representation of cylinder hit
        let u = phi / phi_max;
        let v = (p_hit.z - z_min) / (z_max - z_min);
        // Compute cylinder $\dpdu$ and $\dpdv$
        let dpdu = Vector3f::new(-phi_max * p_hit.y, phi_max * p_hit.x, 0.0);
        let dpdv = Vector3f::new(0.0, 0.0, z_max - z_min);

        // Compute cylinder $\dndu$ and $\dndv$
        let d2pduu = -phi_max * phi_max * Vector3f::new(p_hit.x, p_hit.y, 0.0);
        let d2pduv = Vector3f::new(0.0, 0.0, 0.0);
        let d2pdvv = Vector3f::new(0.0, 0.0, 0.0);

        // Compute coefficients for fundamental forms
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

        // Compute error bounds for intersection computed with ray equation
        let p_error = GAMMA3 * Vector3f::new(p_hit.x, p_hit.y, 0.0).abs();

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

        // Compute quadratic cylinder coefficients

        // Initialize _EFloat_ ray coordinate values
        let radius = self.radius;
        let z_min = self.z_min;
        let z_max = self.z_max;
        let phi_max = self.phi_max;

        let ox = EFloat::from((ray.o.x, o_err.x));
        let oy = EFloat::from((ray.o.y, o_err.y));
        let dx = EFloat::from((ray.d.x, d_err.x));
        let dy = EFloat::from((ray.d.y, d_err.y));
        let a = dx * dx + dy * dy;
        let b = (dx * ox + dy * oy) * 2.0;
        let c = ox * ox + oy * oy - EFloat::from(radius) * EFloat::from(radius);

        // Solve quadratic equation for _t_ values

        if let Some((t0, t1)) = EFloat::quadratic(a, b, c) {
            if t0.v.is_infinite() || t1.v.is_infinite() {
                return false;
            }

            assert!(t0.v.is_finite());
            assert!(t1.v.is_finite());
            assert!(t0.v <= t1.v);
            // Check quadric shape _t0_ and _t1_ for nearest intersection
            if t0.upper_bound() > t_max || t1.lower_bound() <= 0.0 {
                return false;
            }

            // Compute cylinder inverse mapping
            let mut t_shape_hit = t0;
            if t_shape_hit.lower_bound() <= 0.0 {
                t_shape_hit = t1;
                if t_max < t_shape_hit.upper_bound() {
                    return false;
                }
            }

            // Compute cylinder hit point and $\phi$
            let mut p_hit = ray.o + ray.d * Float::from(t_shape_hit);

            // Refine cylinder intersection point
            {
                let hit_rad = Float::sqrt(p_hit.x * p_hit.x + p_hit.y * p_hit.y);
                p_hit.x *= radius / hit_rad;
                p_hit.y *= radius / hit_rad;
            }
            let mut phi = Float::atan2(p_hit.y, p_hit.x);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }

            // Test cylinder intersection against clipping parameters
            if p_hit.z < z_min || p_hit.z > z_max || phi > phi_max {
                if t_shape_hit == t1 {
                    return false;
                }
                if t1.upper_bound() > t_max {
                    return false;
                }
                t_shape_hit = t1;
                // Compute cylinder inverse mapping
                p_hit = ray.o + ray.d * Float::from(t_shape_hit);
                {
                    let hit_rad = Float::sqrt(p_hit.x * p_hit.x + p_hit.y * p_hit.y);
                    p_hit.x *= radius / hit_rad;
                    p_hit.y *= radius / hit_rad;
                }

                phi = Float::atan2(p_hit.y, p_hit.x);
                if phi < 0.0 {
                    phi += 2.0 * PI;
                }
                if p_hit.z < z_min || p_hit.z > z_max || phi > phi_max {
                    return false;
                }
            }

            return true;
        } else {
            return false;
        }
    }

    pub fn area(&self) -> Float {
        let radius = self.radius;
        let z_min = self.z_min;
        let z_max = self.z_max;
        let phi_max = self.phi_max;
        return (z_max - z_min) * radius * phi_max;
    }

    pub fn sample(&self, u: &Point2f) -> Option<(Interaction, Float)> {
        let radius = self.radius;
        let z_min = self.z_min;
        let z_max = self.z_max;
        let phi_max = self.phi_max;

        let z = lerp(u[0], z_min, z_max);
        let phi = u[1] * phi_max;
        let mut p_obj = Point3f::new(radius * Float::cos(phi), radius * Float::sin(phi), z);
        let mut n = self
            .base
            .object_to_world
            .transform_normal(&Normal3f::new(p_obj.x, p_obj.y, 0.0))
            .normalize();
        if self.base.reverse_orientation {
            n *= -1.0;
        }
        // Reproject _pObj_ to cylinder surface and compute _pObjError_
        let hit_rad = Float::sqrt(p_obj.x * p_obj.x + p_obj.y * p_obj.y);
        p_obj.x *= radius / hit_rad;
        p_obj.y *= radius / hit_rad;
        let p_obj_error = GAMMA3 * Vector3f::new(p_obj.x, p_obj.y, 0.0).abs();
        let (p, p_error) = self
            .base
            .object_to_world
            .transform_point_with_abs_error(&p_obj, &p_obj_error);
        let it = Interaction::from_surface_sample(&p, &p_error, &n);
        let pdf = 1.0 / self.area();
        return Some((it, pdf));
    }

    pub fn pdf(&self, _inter: &Interaction) -> Float {
        Float::recip(self.area())
    }

    pub fn sample_from(&self, inter: &Interaction, u: &Point2f) -> Option<(Interaction, Float)> {
        let (intr, pdf) = self.sample(u)?;
        assert!(intr.is_surface_interaction());
        let wi = intr.get_p() - inter.get_p();
        if wi.length_squared() <= 0.0 {
            return None;
        } else {
            assert!(intr.get_n().length() > 0.0);
            let wi = wi.normalize();
            let pdf = pdf * Vector3f::distance_squared(&inter.get_p(), &intr.get_p())
                / Vector3f::abs_dot(&intr.get_n(), &-wi);
            if pdf <= 0.0 || pdf.is_infinite() {
                return None;
            }
            return Some((intr, pdf));
        }
    }

    pub fn pdf_from(&self, inter: &Interaction, wi: &Vector3f) -> Float {
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
        } else {
            return 0.0;
        }
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

pub fn create_cylinder_shape(
    o2w: &Transform,
    w2o: &Transform,
    reverse_orientation: bool,
    params: &ParameterDictionary,
) -> Result<Cylinder, PbrtError> {
    Cylinder::create(o2w, w2o, reverse_orientation, params)
}
