use super::matrix4x4::Matrix4x4;
use crate::interaction::*;
use crate::util::base::*;
use crate::util::geometry::*;

use std::ops;

const MACHINE_EPSILON: Float = Float::EPSILON * 0.5;
const GAMMA3: Float = (3.0 * MACHINE_EPSILON) / (1.0 - (3.0 * MACHINE_EPSILON));

#[inline]
fn radians(x: Float) -> Float {
    return x * (PI / 180.0);
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Transform {
    pub m: Matrix4x4,
    pub minv: Matrix4x4,
}

impl Transform {
    pub fn new() -> Self {
        Self::identity()
    }

    pub fn identity() -> Self {
        let m = Matrix4x4::identity();
        let minv = Matrix4x4::identity();
        Transform { m, minv }
    }

    pub fn translate(x: Float, y: Float, z: Float) -> Self {
        let m = Matrix4x4::translate(x, y, z);
        let minv = Matrix4x4::translate(-x, -y, -z);
        Transform { m, minv }
    }

    pub fn scale(x: Float, y: Float, z: Float) -> Self {
        let m = Matrix4x4::scale(x, y, z);
        let minv = Matrix4x4::scale(1.0 / x, 1.0 / y, 1.0 / z);
        Transform { m, minv }
    }

    pub fn rotate_x(theta: Float) -> Self {
        let m = Matrix4x4::rotate_x(theta);
        let minv = m.transpose();
        Transform { m, minv }
    }

    pub fn rotate_y(theta: Float) -> Self {
        let m = Matrix4x4::rotate_y(theta);
        let minv = m.transpose();
        Transform { m, minv }
    }

    pub fn rotate_z(theta: Float) -> Self {
        let m = Matrix4x4::rotate_z(theta);
        let minv = m.transpose();
        Transform { m, minv }
    }

    pub fn rotate(theta: Float, x: Float, y: Float, z: Float) -> Transform {
        let a = Vector3f::new(x, y, z).normalize();
        let sin_theta = Float::sin(radians(theta));
        let cos_theta = Float::cos(radians(theta));
        let mut m = Matrix4x4::identity();
        // Compute rotation of first basis vector
        m.m[4 * 0 + 0] = a.x * a.x + (1.0 - a.x * a.x) * cos_theta;
        m.m[4 * 0 + 1] = a.x * a.y * (1.0 - cos_theta) - a.z * sin_theta;
        m.m[4 * 0 + 2] = a.x * a.z * (1.0 - cos_theta) + a.y * sin_theta;
        m.m[4 * 0 + 3] = 0.0;

        // Compute rotations of second and third basis vectors
        m.m[4 * 1 + 0] = a.x * a.y * (1.0 - cos_theta) + a.z * sin_theta;
        m.m[4 * 1 + 1] = a.y * a.y + (1.0 - a.y * a.y) * cos_theta;
        m.m[4 * 1 + 2] = a.y * a.z * (1.0 - cos_theta) - a.x * sin_theta;
        m.m[4 * 1 + 3] = 0.0;

        m.m[4 * 2 + 0] = a.x * a.z * (1.0 - cos_theta) - a.y * sin_theta;
        m.m[4 * 2 + 1] = a.y * a.z * (1.0 - cos_theta) + a.x * sin_theta;
        m.m[4 * 2 + 2] = a.z * a.z + (1.0 - a.z * a.z) * cos_theta;
        m.m[4 * 2 + 3] = 0.0;

        return Transform {
            m,
            minv: m.transpose(),
        };
    }

    /// pbrt-v4 `RotateFromTo(from, to)` implemented via two Householder
    /// reflections. `from` and `to` are expected to be normalized.
    pub fn rotate_from_to(from: Vector3f, to: Vector3f) -> Transform {
        let refl = if Float::abs(from.x) < 0.72 && Float::abs(to.x) < 0.72 {
            Vector3f::new(1.0, 0.0, 0.0)
        } else if Float::abs(from.y) < 0.72 && Float::abs(to.y) < 0.72 {
            Vector3f::new(0.0, 1.0, 0.0)
        } else {
            Vector3f::new(0.0, 0.0, 1.0)
        };

        let u = refl - from;
        let v = refl - to;
        let u_dot_u = Vector3f::dot(&u, &u);
        let v_dot_v = Vector3f::dot(&v, &v);
        let u_dot_v = Vector3f::dot(&u, &v);
        let mut r = Matrix4x4::identity();
        for i in 0..3 {
            for j in 0..3 {
                r.m[4 * i + j] = if i == j { 1.0 } else { 0.0 }
                    - 2.0 / u_dot_u * u[i] * u[j]
                    - 2.0 / v_dot_v * v[i] * v[j]
                    + 4.0 * u_dot_v / (u_dot_u * v_dot_v) * v[i] * u[j];
            }
        }
        Transform {
            m: r,
            minv: r.transpose(),
        }
    }

    pub fn look_at(
        ex: Float,
        ey: Float,
        ez: Float,
        lx: Float,
        ly: Float,
        lz: Float,
        ux: Float,
        uy: Float,
        uz: Float,
    ) -> Self {
        let c2w = Matrix4x4::camera_to_world(ex, ey, ez, lx, ly, lz, ux, uy, uz);
        let w2c = c2w.inverse().unwrap();
        Transform { m: w2c, minv: c2w }
    }

    pub fn orthographic(z_near: Float, z_far: Float) -> Self {
        return Self::scale(1.0, 1.0, 1.0 / (z_far - z_near)) * Self::translate(0.0, 0.0, -z_near);
    }

    pub fn perspective(fov: Float, n: Float, f: Float) -> Self {
        #[rustfmt::skip]
        let persp = Matrix4x4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, f / (f - n), -f * n / (f - n),
            0.0, 0.0, 1.0, 0.0,
        );
        let inv_tan_ang = 1.0 / Float::tan(radians(fov) / 2.0);
        return Self::scale(inv_tan_ang, inv_tan_ang, 1.0) * Transform::from(persp);
    }

    pub fn set(&mut self, other: &Transform) {
        self.m = other.m;
        self.minv = other.minv;
    }

    pub fn inverse(&self) -> Self {
        Transform {
            m: self.minv,
            minv: self.m,
        }
    }

    pub fn swaps_handedness(&self) -> bool {
        let m = &self.m;
        #[rustfmt::skip]
        let det = m.m[4 * 0 + 0] * (m.m[4 * 1 + 1] * m.m[4 * 2 + 2] - m.m[4 * 1 + 2] * m.m[4 * 2 + 1]) -
                       m.m[4 * 0 + 1] * (m.m[4 * 1 + 0] * m.m[4 * 2 + 2] - m.m[4 * 1 + 2] * m.m[4 * 2 + 0]) +
                       m.m[4 * 0 + 2] * (m.m[4 * 1 + 0] * m.m[4 * 2 + 1] - m.m[4 * 1 + 1] * m.m[4 * 2 + 0]);
        return det < 0.0;
    }

    pub fn transform_point(&self, p: &Point3f) -> Point3f {
        return self.m.transform_point(p);
    }

    pub fn transform_vector(&self, p: &Vector3f) -> Vector3f {
        return self.m.transform_vector(p);
    }

    pub fn transform_normal(&self, p: &Normal3f) -> Normal3f {
        return self.minv.transform_normal(p);
    }

    pub fn transform_bounds(&self, bounds: &Bounds3f) -> Bounds3f {
        let min = &bounds.min;
        let max = &bounds.max;
        let p = [
            [min.x, min.y, min.z],
            [min.x, min.y, max.z],
            [min.x, max.y, min.z],
            [min.x, max.y, max.z],
            [max.x, min.y, min.z],
            [max.x, min.y, max.z],
            [max.x, max.y, min.z],
            [max.x, max.y, max.z],
        ];
        let q: Vec<Vector3f> = p
            .iter()
            .map(|v| -> Vector3f {
                return self.transform_point(&Vector3f::new(v[0], v[1], v[2]));
            })
            .collect();
        let min = q
            .iter()
            .map(|v| [v.x, v.y, v.z])
            .reduce(|acc, v| {
                return [
                    Float::min(acc[0], v[0]),
                    Float::min(acc[1], v[1]),
                    Float::min(acc[2], v[2]),
                ];
            })
            .unwrap();
        let max = q
            .iter()
            .map(|v| [v.x, v.y, v.z])
            .reduce(|acc, v| {
                return [
                    Float::max(acc[0], v[0]),
                    Float::max(acc[1], v[1]),
                    Float::max(acc[2], v[2]),
                ];
            })
            .unwrap();
        assert!(min[0] <= max[0]);
        assert!(min[1] <= max[1]);
        assert!(min[2] <= max[2]);
        assert!(q.iter().all(|v| v.x >= min[0] && v.x <= max[0]));
        assert!(q.iter().all(|v| v.y >= min[1] && v.y <= max[1]));
        assert!(q.iter().all(|v| v.z >= min[2] && v.z <= max[2]));
        return Bounds3f::from(((min[0], min[1], min[2]), (max[0], max[1], max[2])));
    }

    pub fn transform_point_with_error(&self, p: &Point3f) -> (Point3f, Vector3f) {
        let x = p.x;
        let y = p.y;
        let z = p.z;
        let p = self.transform_point(p);
        let x_abs_sum = Float::abs(self.m.m[4 * 0 + 0] * x)
            + Float::abs(self.m.m[4 * 0 + 1] * y)
            + Float::abs(self.m.m[4 * 0 + 2] * z)
            + Float::abs(self.m.m[4 * 0 + 3]);
        let y_abs_sum = Float::abs(self.m.m[4 * 1 + 0] * x)
            + Float::abs(self.m.m[4 * 1 + 1] * y)
            + Float::abs(self.m.m[4 * 1 + 2] * z)
            + Float::abs(self.m.m[4 * 1 + 3]);
        let z_abs_sum = Float::abs(self.m.m[4 * 2 + 0] * x)
            + Float::abs(self.m.m[4 * 2 + 1] * y)
            + Float::abs(self.m.m[4 * 2 + 2] * z)
            + Float::abs(self.m.m[4 * 2 + 3]);
        let p_error = GAMMA3 * Vector3f::new(x_abs_sum, y_abs_sum, z_abs_sum);
        return (p, p_error);
    }

    pub fn transform_point_with_abs_error(
        &self,
        p: &Point3f,
        pt_error: &Vector3f,
    ) -> (Point3f, Vector3f) {
        let x = p.x;
        let y = p.y;
        let z = p.z;
        let p = self.transform_point(p);
        let ex = (GAMMA3 + 1.0)
            * (Float::abs(self.m.m[4 * 0 + 0]) * pt_error.x
                + Float::abs(self.m.m[4 * 0 + 1]) * pt_error.y
                + Float::abs(self.m.m[4 * 0 + 2]) * pt_error.z)
            + GAMMA3
                * (Float::abs(self.m.m[4 * 0 + 0] * x)
                    + Float::abs(self.m.m[4 * 0 + 1] * y)
                    + Float::abs(self.m.m[4 * 0 + 2] * z)
                    + Float::abs(self.m.m[4 * 0 + 3]));
        let ey = (GAMMA3 + 1.0)
            * (Float::abs(self.m.m[4 * 1 + 0]) * pt_error.x
                + Float::abs(self.m.m[4 * 1 + 1]) * pt_error.y
                + Float::abs(self.m.m[4 * 1 + 2]) * pt_error.z)
            + GAMMA3
                * (Float::abs(self.m.m[4 * 1 + 0] * x)
                    + Float::abs(self.m.m[4 * 1 + 1] * y)
                    + Float::abs(self.m.m[4 * 1 + 2] * z)
                    + Float::abs(self.m.m[4 * 1 + 3]));
        let ez = (GAMMA3 + 1.0)
            * (Float::abs(self.m.m[4 * 2 + 0]) * pt_error.x
                + Float::abs(self.m.m[4 * 2 + 1]) * pt_error.y
                + Float::abs(self.m.m[4 * 2 + 2]) * pt_error.z)
            + GAMMA3
                * (Float::abs(self.m.m[4 * 2 + 0] * x)
                    + Float::abs(self.m.m[4 * 2 + 1] * y)
                    + Float::abs(self.m.m[4 * 2 + 2] * z)
                    + Float::abs(self.m.m[4 * 2 + 3]));
        let p_error = Vector3f::new(ex, ey, ez);
        return (p, p_error);
    }

    pub fn transform_vector_with_error(&self, v: &Vector3f) -> (Vector3f, Vector3f) {
        let x = v.x;
        let y = v.y;
        let z = v.z;
        let ex = GAMMA3
            * (Float::abs(self.m.m[4 * 0 + 0] * v.x)
                + Float::abs(self.m.m[4 * 0 + 1] * v.y)
                + Float::abs(self.m.m[4 * 0 + 2] * v.z));
        let ey = GAMMA3
            * (Float::abs(self.m.m[4 * 1 + 0] * v.x)
                + Float::abs(self.m.m[4 * 1 + 1] * v.y)
                + Float::abs(self.m.m[4 * 1 + 2] * v.z));
        let ez = GAMMA3
            * (Float::abs(self.m.m[4 * 2 + 0] * v.x)
                + Float::abs(self.m.m[4 * 2 + 1] * v.y)
                + Float::abs(self.m.m[4 * 2 + 2] * v.z));
        let vv = Vector3f::new(
            self.m.m[4 * 0 + 0] * x + self.m.m[4 * 0 + 1] * y + self.m.m[4 * 0 + 2] * z,
            self.m.m[4 * 1 + 0] * x + self.m.m[4 * 1 + 1] * y + self.m.m[4 * 1 + 2] * z,
            self.m.m[4 * 2 + 0] * x + self.m.m[4 * 2 + 1] * y + self.m.m[4 * 2 + 2] * z,
        );
        let ve = Vector3f::new(ex, ey, ez);
        return (vv, ve);
    }

    pub fn transform_ray(&self, r: &Ray) -> (Ray, Vector3f, Vector3f) {
        let (mut o, o_error) = self.transform_point_with_error(&r.o);
        let (d, d_error) = self.transform_vector_with_error(&r.d);
        let length_squared = d.length_squared();
        if length_squared > 0.0 {
            let dt = d.abs().dot(&o_error) / length_squared;
            o += d * dt;
        }
        let mut r2 = Ray::new(&o, &d, Float::INFINITY, r.time);
        r2.medium = r.medium.clone();
        return (r2, o_error, d_error);
    }

    pub fn transform_ray_differential(
        &self,
        r: &RayDifferential,
    ) -> (RayDifferential, Vector3f, Vector3f) {
        let mut ret = r.clone();
        let (tr, o_error, d_error) = self.transform_ray(&r.ray);
        ret.ray = tr;
        ret.has_differentials = r.has_differentials;
        ret.rx_origin = self.transform_point(&r.rx_origin);
        ret.ry_origin = self.transform_point(&r.ry_origin);
        ret.rx_direction = self.transform_vector(&r.rx_direction);
        ret.ry_direction = self.transform_vector(&r.ry_direction);
        return (ret, o_error, d_error);
    }

    pub fn transform_surface_interaction(&self, si: &SurfaceInteraction) -> SurfaceInteraction {
        let mut ret = si.clone();
        let (p, p_error) = self.transform_point_with_abs_error(&si.p, &si.p_error);
        ret.p = p;
        ret.p_error = p_error;
        ret.n = self.transform_normal(&si.n).normalize();
        ret.wo = self.transform_vector(&si.wo).normalize();
        //
        ret.dpdu = self.transform_vector(&ret.dpdu);
        ret.dpdv = self.transform_vector(&ret.dpdv);
        ret.dndu = self.transform_normal(&ret.dndu);
        ret.dndv = self.transform_normal(&ret.dndv);

        ret.shading.n = self.transform_normal(&si.shading.n).normalize();
        ret.shading.dpdu = self.transform_vector(&ret.shading.dpdu);
        ret.shading.dpdv = self.transform_vector(&ret.shading.dpdv);
        ret.shading.dndu = self.transform_normal(&ret.shading.dndu);
        ret.shading.dndv = self.transform_normal(&ret.shading.dndv);

        ret.dpdx = self.transform_vector(&ret.dpdx);
        ret.dpdy = self.transform_vector(&ret.dpdy);

        ret.shading.n = face_forward(&ret.shading.n, &ret.n);
        return ret;
    }

    pub fn is_identity(&self) -> bool {
        self.m == Matrix4x4::identity()
    }

    pub fn has_scale(&self) -> bool {
        let m = &self.m;
        let la2 = m.m[4 * 0 + 0] * m.m[4 * 0 + 0]
            + m.m[4 * 0 + 1] * m.m[4 * 0 + 1]
            + m.m[4 * 0 + 2] * m.m[4 * 0 + 2];
        let lb2 = m.m[4 * 1 + 0] * m.m[4 * 1 + 0]
            + m.m[4 * 1 + 1] * m.m[4 * 1 + 1]
            + m.m[4 * 1 + 2] * m.m[4 * 1 + 2];
        let lc2 = m.m[4 * 2 + 0] * m.m[4 * 2 + 0]
            + m.m[4 * 2 + 1] * m.m[4 * 2 + 1]
            + m.m[4 * 2 + 2] * m.m[4 * 2 + 2];
        let det = m.m[4 * 0 + 0]
            * (m.m[4 * 1 + 1] * m.m[4 * 2 + 2] - m.m[4 * 1 + 2] * m.m[4 * 2 + 1])
            - m.m[4 * 0 + 1] * (m.m[4 * 1 + 0] * m.m[4 * 2 + 2] - m.m[4 * 1 + 2] * m.m[4 * 2 + 0])
            + m.m[4 * 0 + 2] * (m.m[4 * 1 + 0] * m.m[4 * 2 + 1] - m.m[4 * 1 + 1] * m.m[4 * 2 + 0]);
        return Float::abs(det) < 0.01 || (la2 < 0.01 || lb2 < 0.01 || lc2 < 0.01);
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::new()
    }
}

impl ops::Mul<Transform> for Transform {
    type Output = Transform;
    fn mul(self, t2: Transform) -> Transform {
        return Transform {
            m: self.m * t2.m,
            minv: t2.minv * self.minv,
        };
    }
}

impl From<[Float; 16]> for Transform {
    fn from(v: [Float; 16]) -> Self {
        let m = Matrix4x4::from(v);
        let minv = m.inverse().unwrap();
        Transform { m, minv }
    }
}

impl From<Matrix4x4> for Transform {
    fn from(m: Matrix4x4) -> Self {
        Transform {
            m,
            minv: m.inverse().unwrap(),
        }
    }
}

impl From<(Matrix4x4, Matrix4x4)> for Transform {
    fn from(v: (Matrix4x4, Matrix4x4)) -> Self {
        Transform { m: v.0, minv: v.1 }
    }
}
