use crate::interaction::*;
use crate::paramdict::*;

use crate::shapes::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::sampling::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

#[inline]
fn radians(x: Float) -> Float {
    return x * (PI / 180.0);
}

pub struct Disk {
    pub base: BaseShape,
    pub height: Float,
    pub radius: Float,
    pub inner_radius: Float,
    pub phi_max: Float,
}

impl Disk {
    pub fn new(
        o2w: &Transform,
        w2o: &Transform,
        reverse_orientation: bool,
        height: Float,
        radius: Float,
        inner_radius: Float,
        phi_max: Float,
    ) -> Self {
        let phi_max = radians(Float::clamp(phi_max, 0.0, 360.0));
        Disk {
            base: BaseShape::new(o2w, w2o, reverse_orientation),
            height,
            radius,
            inner_radius,
            phi_max,
        }
    }

    pub fn create(
        o2w: &Transform,
        w2o: &Transform,
        reverse_orientation: bool,
        params: &ParameterDictionary,
    ) -> Result<Self, PbrtError> {
        let height = params.get_one_float("height", 0.0);
        let radius = params.get_one_float("radius", 1.0);
        let inner_radius = params.get_one_float("innerradius", 0.0);
        let phimax = params.get_one_float("phimax", 360.0);

        Ok(Self::new(
            o2w,
            w2o,
            reverse_orientation,
            height,
            radius,
            inner_radius,
            phimax,
        ))
    }
}

impl Disk {
    pub fn object_bound(&self) -> Bounds3f {
        let radius = self.radius;
        let height = self.height;
        return Bounds3f::new(
            &Point3f::new(-radius, -radius, height - 0.001), //originally, Point3f(-radius, -radius, height)
            &Point3f::new(radius, radius, height + 0.001),
        );
    }
    pub fn world_bound(&self) -> Bounds3f {
        return self
            .base
            .object_to_world
            .transform_bounds(&self.object_bound());
    }

    /// pbrt-v4 `Disk::NormalBounds` (shapes.cpp:89). Disks lie on a
    /// single plane so the normal is a single direction (the object-
    /// space +z transformed to render space, optionally flipped by
    /// `reverseOrientation`).
    pub fn normal_bounds(&self) -> DirectionCone {
        use crate::util::base::{Normal3f, Vector3f};
        let mut n: Normal3f = self
            .base
            .object_to_world
            .transform_normal(&Normal3f::new(0.0, 0.0, 1.0))
            .normalize();
        if self.base.reverse_orientation {
            n = -n;
        }
        DirectionCone::from_direction(Vector3f::from(n))
    }
    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let (ray, _o_err, _d_err) = self.base.world_to_object.transform_ray(r);

        // Compute plane intersection for disk

        // Reject disk intersections for rays parallel to the disk's plane
        let height = self.height;
        if ray.d.z == 0.0 {
            return None;
        }
        let t_shape_hit = (height - ray.o.z) / ray.d.z;
        if t_shape_hit <= 0.0 || t_shape_hit >= t_max {
            return None;
        }

        // See if hit point is inside disk radii and $\phimax$
        let radius = self.radius;
        let inner_radius = self.inner_radius;
        let mut p_hit = ray.o + ray.d * t_shape_hit;
        let dist2 = p_hit.x * p_hit.x + p_hit.y * p_hit.y;
        if dist2 > radius * radius || dist2 < inner_radius * inner_radius {
            return None;
        }

        // Test disk $\phi$ value against $\phimax$
        let phi_max = self.phi_max;
        let mut phi = Float::atan2(p_hit.y, p_hit.x);
        if phi < 0.0 {
            phi += 2.0 * PI;
        }
        if phi > phi_max {
            return None;
        }

        let u = phi / phi_max;
        let r_hit = Float::sqrt(dist2);
        let v = (radius - r_hit) / (radius - inner_radius);
        let dpdu = Vector3f::new(-phi_max * p_hit.y, phi_max * p_hit.x, 0.0);
        let dpdv = Vector3f::new(p_hit.x, p_hit.y, 0.0) * ((inner_radius - radius) / r_hit);
        let dndu = Vector3f::new(0.0, 0.0, 0.0);
        let dndv = Vector3f::new(0.0, 0.0, 0.0);

        let n = self.base.calc_normal(&dpdu, &dpdv);

        // Refine disk intersection point
        p_hit.z = height;

        // Compute error bounds for disk intersection
        let p_error = Vector3f::new(0.0, 0.0, 0.0);

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
        return Some(ShapeIntersection::new(isect, t_shape_hit));
    }

    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        let (ray, _o_err, _d_err) = self.base.world_to_object.transform_ray(r);

        // Compute plane intersection for disk

        // Reject disk intersections for rays parallel to the disk's plane
        let height = self.height;
        if ray.d.z == 0.0 {
            return false;
        }
        let t_shape_hit = (height - ray.o.z) / ray.d.z;
        if t_shape_hit <= 0.0 || t_shape_hit >= t_max {
            return false;
        }

        // See if hit point is inside disk radii and $\phimax$
        let radius = self.radius;
        let inner_radius = self.inner_radius;
        let p_hit = ray.o + ray.d * t_shape_hit;
        let dist2 = p_hit.x * p_hit.x + p_hit.y * p_hit.y;
        if dist2 > radius * radius || dist2 < inner_radius * inner_radius {
            return false;
        }

        // Test disk $\phi$ value against $\phimax$
        let phi_max = self.phi_max;
        let mut phi = Float::atan2(p_hit.y, p_hit.x);
        if phi < 0.0 {
            phi += 2.0 * PI;
        }
        if phi > phi_max {
            return false;
        }

        return true;
    }

    pub fn area(&self) -> Float {
        let radius = self.radius;
        let inner_radius = self.inner_radius;
        let phi_max = self.phi_max;
        return phi_max * 0.5 * (radius * radius - inner_radius * inner_radius);
    }

    pub fn sample(&self, u: &Point2f) -> Option<(Interaction, Float)> {
        let radius = self.radius;
        let height = self.height;
        let pd = concentric_sample_disk(u);
        let p_obj = Point3f::new(pd.x * radius, pd.y * radius, height);
        let mut n = self
            .base
            .object_to_world
            .transform_normal(&Normal3f::new(0.0, 0.0, 1.0))
            .normalize();
        if self.base.reverse_orientation {
            n *= -1.0;
        }
        let (p, p_error) = self
            .base
            .object_to_world
            .transform_point_with_abs_error(&p_obj, &Point3f::new(0.0, 0.0, 0.0));
        let pdf = 1.0 / self.area();
        let it = Interaction::from_surface_sample(&p, &p_error, &n);
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

pub fn create_disk_shape(
    o2w: &Transform,
    w2o: &Transform,
    reverse_orientation: bool,
    params: &ParameterDictionary,
) -> Result<Disk, PbrtError> {
    Disk::create(o2w, w2o, reverse_orientation, params)
}
