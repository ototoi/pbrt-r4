use crate::interaction::*;
use crate::paramdict::*;

use crate::shapes::*;
use crate::util::base::*;
use crate::util::efloat::*;
use crate::util::error::*;
use crate::util::geometry::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

#[inline]
fn radians(x: Float) -> Float {
    return x * (PI / 180.0);
}

const GAMMA3: Float = (3.0 * MACHINE_EPSILON) / (1.0 - (3.0 * MACHINE_EPSILON));

pub struct Paraboloid {
    pub base: BaseShape,
    pub radius: Float,
    pub z_min: Float,
    pub z_max: Float,
    pub phi_max: Float,
}

impl Paraboloid {
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
        Paraboloid {
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
        let mut zmin = params.get_one_float("zmin", 0.0);
        let mut zmax = params.get_one_float("zmax", 1.0);
        let phimax = params.get_one_float("phimax", 360.0);

        if zmin > zmax {
            std::mem::swap(&mut zmin, &mut zmax);
        }

        if radius == 0.0 {
            let msg = format!(
                "Unable to create paraboloid shape: radius={}, phimax={}",
                radius, phimax
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

impl Paraboloid {
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

        // Compute quadratic paraboloid coefficients

        // Initialize _EFloat_ ray coordinate values
        let radius = self.radius;
        let z_min = self.z_min;
        let z_max = self.z_max;
        let phi_max = self.phi_max;

        let ox = EFloat::from((ray.o.x, o_err.x));
        let oy = EFloat::from((ray.o.y, o_err.y));
        let oz = EFloat::from((ray.o.z, o_err.z));
        let dx = EFloat::from((ray.d.x, d_err.x));
        let dy = EFloat::from((ray.d.y, d_err.y));
        let dz = EFloat::from((ray.d.z, d_err.z));
        let k = EFloat::from(z_max) / EFloat::from(radius) * EFloat::from(radius);
        let a = k * (dx * dx + dy * dy);
        let b = ((dx * ox + dy * oy) * k * 2.0) - dz;
        let c = k * (ox * ox + oy * oy) - oz;

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

        // Compute paraboloid inverse mapping
        let mut t_shape_hit = t0;
        if t_shape_hit.lower_bound() <= 0.0 {
            t_shape_hit = t1;
            if t_max < t_shape_hit.upper_bound() {
                return None;
            }
        }

        // Compute paraboloid inverse mapping
        let mut p_hit = ray.o + ray.d * Float::from(t_shape_hit);
        let mut phi = Float::atan2(p_hit.y, p_hit.x);
        if phi < 0.0 {
            phi += 2.0 * PI;
        }

        // Test paraboloid intersection against clipping parameters
        if p_hit.z < z_min || p_hit.z > z_max || phi > phi_max {
            if t_shape_hit == t1 {
                return None;
            }
            if t1.upper_bound() > t_max {
                return None;
            }
            t_shape_hit = t1;
            // Compute paraboloid inverse mapping
            p_hit = ray.o + ray.d * Float::from(t_shape_hit);
            phi = Float::atan2(p_hit.y, p_hit.x);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }
            if p_hit.z < z_min || p_hit.z > z_max || phi > phi_max {
                return None;
            }
        }

        // Find parametric representation of paraboloid hit
        let u = phi / phi_max;
        let v = (p_hit.z - z_min) / (z_max - z_min);

        // Compute paraboloid $\dpdu$ and $\dpdv$
        let dpdu = Vector3f::new(-phi_max * p_hit.y, phi_max * p_hit.x, 0.0);
        let dpdv = (z_max - z_min)
            * Vector3f::new(p_hit.x / (2.0 * p_hit.z), p_hit.y / (2.0 * p_hit.z), 1.0);

        // Compute paraboloid $\dndu$ and $\dndv$
        let d2pduu = -phi_max * phi_max * Vector3f::new(p_hit.x, p_hit.y, 0.0);
        let d2pduv = (z_max - z_min)
            * phi_max
            * Vector3f::new(-p_hit.y / (2.0 * p_hit.z), p_hit.x / (2.0 * p_hit.z), 0.0);
        let d2pdvv = -(z_max - z_min)
            * (z_max - z_min)
            * Vector3f::new(
                p_hit.x / (4.0 * p_hit.z * p_hit.z),
                p_hit.y / (4.0 * p_hit.z * p_hit.z),
                0.0,
            );

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
        let px = ox + t_shape_hit * dx;
        let py = oy + t_shape_hit * dy;
        let pz = oz + t_shape_hit * dz;
        let p_error = Vector3f::new(
            px.get_absolute_error() as Float,
            py.get_absolute_error() as Float,
            pz.get_absolute_error() as Float,
        );
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

        // Compute quadratic paraboloid coefficients

        // Initialize _EFloat_ ray coordinate values
        let radius = self.radius;
        let z_min = self.z_min;
        let z_max = self.z_max;
        let phi_max = self.phi_max;

        let ox = EFloat::from((ray.o.x, o_err.x));
        let oy = EFloat::from((ray.o.y, o_err.y));
        let oz = EFloat::from((ray.o.z, o_err.z));
        let dx = EFloat::from((ray.d.x, d_err.x));
        let dy = EFloat::from((ray.d.y, d_err.y));
        let dz = EFloat::from((ray.d.z, d_err.z));
        let k = EFloat::from(z_max) / EFloat::from(radius) * EFloat::from(radius);
        let a = k * (dx * dx + dy * dy);
        let b = ((dx * ox + dy * oy) * k * 2.0) - dz;
        let c = k * (ox * ox + oy * oy) - oz;

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

            // Compute paraboloid inverse mapping
            let mut t_shape_hit = t0;
            if t_shape_hit.lower_bound() <= 0.0 {
                t_shape_hit = t1;
                if t_max < t_shape_hit.upper_bound() {
                    return false;
                }
            }

            // Compute paraboloid inverse mapping
            let mut p_hit = ray.o + ray.d * Float::from(t_shape_hit);
            let mut phi = Float::atan2(p_hit.y, p_hit.x);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }

            // Test paraboloid intersection against clipping parameters
            if p_hit.z < z_min || p_hit.z > z_max || phi > phi_max {
                if t_shape_hit == t1 {
                    return false;
                }
                if t1.upper_bound() > t_max {
                    return false;
                }
                t_shape_hit = t1;
                // Compute paraboloid inverse mapping
                p_hit = ray.o + ray.d * Float::from(t_shape_hit);
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
        let radius2 = radius * radius;
        let k = 4.0 * z_max / radius2;

        return (radius2 * radius2 * phi_max / (12.0 * z_max * z_max))
            * (Float::powf(k * z_max + 1.0, 1.5) - Float::powf(k * z_min + 1.0, 1.5));
    }

    pub fn sample(&self, u: &Point2f) -> Option<(Interaction, Float)> {
        let radius = self.radius;
        let z_min = self.z_min;
        let z_max = self.z_max;
        let phi_max = self.phi_max;
        if radius <= 0.0 || z_max <= 0.0 {
            return None;
        }

        let u_phi = Float::clamp(u.x, 0.0, 1.0 - 1e-6);
        let u_area = Float::clamp(u.y, 1e-6, 1.0 - 1e-6);

        let radius2 = radius * radius;
        let k = 4.0 * z_max / radius2;
        if !k.is_finite() || k <= 0.0 {
            return None;
        }

        // Invert area CDF A(z) to sample uniformly over surface area.
        let c0 = Float::powf(k * z_min + 1.0, 1.5);
        let c1 = Float::powf(k * z_max + 1.0, 1.5);
        if !c0.is_finite() || !c1.is_finite() || c1 <= c0 {
            return None;
        }
        let c = lerp(u_area, c0, c1);
        let mut z = (Float::powf(c, 2.0 / 3.0) - 1.0) / k;
        z = Float::clamp(z, z_min, z_max);

        let phi = u_phi * phi_max;
        let r = radius * Float::sqrt(Float::max(0.0, z / z_max));
        let p_obj = Point3f::new(r * Float::cos(phi), r * Float::sin(phi), z);
        let dpdu = Vector3f::new(-phi_max * p_obj.y, phi_max * p_obj.x, 0.0);
        let z_safe = Float::max(p_obj.z, 1e-6);
        let dpdv = (z_max - z_min)
            * Vector3f::new(p_obj.x / (2.0 * z_safe), p_obj.y / (2.0 * z_safe), 1.0);
        let n_obj = self.base.calc_normal(&dpdu, &dpdv);
        let n = self
            .base
            .object_to_world
            .transform_normal(&n_obj)
            .normalize();

        let p_obj_error = GAMMA3 * Vector3f::new(p_obj.x, p_obj.y, p_obj.z).abs();
        let (p, p_error) = self
            .base
            .object_to_world
            .transform_point_with_abs_error(&p_obj, &p_obj_error);
        let pdf = Float::recip(self.area());
        if !pdf.is_finite() || pdf <= 0.0 {
            return None;
        }

        let it = Interaction::from_surface_sample(&p, &p_error, &n);
        Some((it, pdf))
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

pub fn create_paraboloid_shape(
    o2w: &Transform,
    w2o: &Transform,
    reverse_orientation: bool,
    params: &ParameterDictionary,
) -> Result<Paraboloid, PbrtError> {
    Paraboloid::create(o2w, w2o, reverse_orientation, params)
}
