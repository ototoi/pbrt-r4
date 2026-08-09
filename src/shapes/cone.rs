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

pub struct Cone {
    pub base: BaseShape,
    pub radius: Float,
    pub height: Float,
    pub phi_max: Float,
}

impl Cone {
    pub fn new(
        o2w: &Transform,
        w2o: &Transform,
        reverse_orientation: bool,
        height: Float,
        radius: Float,
        phi_max: Float,
    ) -> Self {
        let phi_max = radians(Float::clamp(phi_max, 0.0, 360.0));
        Cone {
            base: BaseShape::new(o2w, w2o, reverse_orientation),
            radius,
            height,
            phi_max,
        }
    }

    pub fn create(
        o2w: &Transform,
        w2o: &Transform,
        reverse_orientation: bool,
        params: &ParameterDictionary,
    ) -> Result<Self, PbrtError> {
        let height = params.get_one_float("height", 1.0);
        let radius = params.get_one_float("radius", 1.0);
        let phimax = params.get_one_float("phimax", 360.0);

        if height == 0.0 || radius == 0.0 {
            let msg = format!(
                "Unable to create cone shape: height={}, radius={}, phimax={}",
                height, radius, phimax
            );
            Err(PbrtError::error(&msg))
        } else {
            Ok(Self::new(
                o2w,
                w2o,
                reverse_orientation,
                height,
                radius,
                phimax,
            ))
        }
    }
}

impl Cone {
    pub fn object_bound(&self) -> Bounds3f {
        let radius = self.radius;
        let height = Float::max(self.height, MACHINE_EPSILON);
        return Bounds3f::new(
            &Point3f::new(-radius, -radius, 0.0),
            &Point3f::new(radius, radius, height),
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

        // Compute quadratic cone coefficients

        // Initialize _EFloat_ ray coordinate values
        let radius = self.radius;
        let height = self.height;
        let phi_max = self.phi_max;

        let ox = EFloat::from((ray.o.x, o_err.x));
        let oy = EFloat::from((ray.o.y, o_err.y));
        let oz = EFloat::from((ray.o.z, o_err.z));
        let dx = EFloat::from((ray.d.x, d_err.x));
        let dy = EFloat::from((ray.d.y, d_err.y));
        let dz = EFloat::from((ray.d.z, d_err.z));
        let k = EFloat::from(radius) / EFloat::from(height);
        let k = k * k;
        let height_ = EFloat::from(height);
        let a = dx * dx + dy * dy - k * dz * dz;
        let b = (dx * ox + dy * oy - k * dz * (oz - height_)) * 2.0;
        let c = ox * ox + oy * oy - k * (oz - height_) * (oz - height_);

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

        // Compute cone inverse mapping
        let mut t_shape_hit = t0;
        if t_shape_hit.lower_bound() <= 0.0 {
            t_shape_hit = t1;
            if t_max < t_shape_hit.upper_bound() {
                return None;
            }
        }

        // Compute cone inverse mapping
        let mut p_hit = ray.o + ray.d * Float::from(t_shape_hit);
        let mut phi = Float::atan2(p_hit.y, p_hit.x);
        if phi < 0.0 {
            phi += 2.0 * PI;
        }

        // Test cone intersection against clipping parameters
        if p_hit.z < 0.0 || p_hit.z > height || phi > phi_max {
            if t_shape_hit == t1 {
                return None;
            }
            if t1.upper_bound() > t_max {
                return None;
            }
            t_shape_hit = t1;
            // Compute cone inverse mapping
            p_hit = ray.o + ray.d * Float::from(t_shape_hit);
            phi = Float::atan2(p_hit.y, p_hit.x);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }
            if p_hit.z < 0.0 || p_hit.z > height || phi > phi_max {
                return None;
            }
        }

        // Find parametric representation of cone hit
        let u = phi / phi_max;
        let v = p_hit.z / height;

        // Compute cone $\dpdu$ and $\dpdv$
        let dpdu = Vector3f::new(-phi_max * p_hit.y, phi_max * p_hit.x, 0.0);
        let dpdv = Vector3f::new(-p_hit.x / (1.0 - v), -p_hit.y / (1.0 - v), height);

        // Compute cone $\dndu$ and $\dndv$
        let d2pduu = -phi_max * phi_max * Vector3f::new(p_hit.x, p_hit.y, 0.0);
        let d2pduv = phi_max / (1.0 - v) * Vector3f::new(p_hit.y, -p_hit.x, 0.0);
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

        // Compute quadratic cone coefficients

        // Initialize _EFloat_ ray coordinate values
        let radius = self.radius;
        let height = self.height;
        let phi_max = self.phi_max;

        let ox = EFloat::from((ray.o.x, o_err.x));
        let oy = EFloat::from((ray.o.y, o_err.y));
        let oz = EFloat::from((ray.o.z, o_err.z));
        let dx = EFloat::from((ray.d.x, d_err.x));
        let dy = EFloat::from((ray.d.y, d_err.y));
        let dz = EFloat::from((ray.d.z, d_err.z));
        let k = EFloat::from(radius) / EFloat::from(height);
        let k = k * k;
        let height_ = EFloat::from(height);
        let a = dx * dx + dy * dy - k * dz * dz;
        let b = (dx * ox + dy * oy - k * dz * (oz - height_)) * 2.0;
        let c = ox * ox + oy * oy - k * (oz - height_) * (oz - height_);

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

            // Compute cone inverse mapping
            let mut t_shape_hit = t0;
            if t_shape_hit.lower_bound() <= 0.0 {
                t_shape_hit = t1;
                if t_max < t_shape_hit.upper_bound() {
                    return false;
                }
            }

            // Compute cone inverse mapping
            let mut p_hit = ray.o + ray.d * Float::from(t_shape_hit);
            let mut phi = Float::atan2(p_hit.y, p_hit.x);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }

            // Test cone intersection against clipping parameters
            if p_hit.z < 0.0 || p_hit.z > height || phi > phi_max {
                if t_shape_hit == t1 {
                    return false;
                }
                if t1.upper_bound() > t_max {
                    return false;
                }
                t_shape_hit = t1;
                // Compute cone inverse mapping
                p_hit = ray.o + ray.d * Float::from(t_shape_hit);
                phi = Float::atan2(p_hit.y, p_hit.x);
                if phi < 0.0 {
                    phi += 2.0 * PI;
                }
                if p_hit.z < 0.0 || p_hit.z > height || phi > phi_max {
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
        let height = self.height;
        let phi_max = self.phi_max;
        return radius * Float::sqrt((height * height) + (radius * radius)) * phi_max / 2.0;
    }

    pub fn sample(&self, u: &Point2f) -> Option<(Interaction, Float)> {
        let radius = self.radius;
        let height = self.height;
        let phi_max = self.phi_max;

        let u_phi = Float::clamp(u.x, 0.0, 1.0 - 1e-6);
        let u_height = Float::clamp(u.y, 0.0, 1.0 - 1e-6);
        let phi = u_phi * phi_max;
        // Area element is proportional to (1 - v), so invert its CDF.
        let v = 1.0 - Float::sqrt(1.0 - u_height);
        let one_minus_v = Float::max(1.0 - v, 1e-6);

        let r = radius * one_minus_v;
        let p_obj = Point3f::new(r * Float::cos(phi), r * Float::sin(phi), v * height);
        let dpdu = Vector3f::new(-phi_max * p_obj.y, phi_max * p_obj.x, 0.0);
        let dpdv = Vector3f::new(-p_obj.x / one_minus_v, -p_obj.y / one_minus_v, height);
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

pub fn create_cone_shape(
    o2w: &Transform,
    w2o: &Transform,
    reverse_orientation: bool,
    params: &ParameterDictionary,
) -> Result<Cone, PbrtError> {
    Cone::create(o2w, w2o, reverse_orientation, params)
}
