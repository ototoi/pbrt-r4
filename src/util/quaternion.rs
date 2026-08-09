use crate::util::base::*;
use crate::util::transform::*;
use std::ops;

#[derive(Debug, PartialEq, Default, Copy, Clone)]
pub struct Quaternion {
    pub x: Float,
    pub y: Float,
    pub z: Float,
    pub w: Float,
}

impl Quaternion {
    pub fn new(x: Float, y: Float, z: Float, w: Float) -> Self {
        if w >= 0.0 {
            return Quaternion { x, y, z, w };
        } else {
            return Quaternion {
                x: -x,
                y: -y,
                z: -z,
                w: -w,
            };
        }
    }

    pub fn identity() -> Self {
        Quaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }

    pub fn normalize(&self) -> Self {
        let l = Float::sqrt(Quaternion::dot(self, self));
        return Quaternion::new(self.x / l, self.y / l, self.z / l, self.w / l);
    }

    pub fn dot(&self, q2: &Quaternion) -> Float {
        return (self.x * q2.x) + (self.y * q2.y) + (self.z * q2.z) + (self.w * q2.w);
    }

    pub fn slerp(t: Float, q1: &Quaternion, q2: &Quaternion) -> Quaternion {
        const T: Float = 1.0 - Float::EPSILON;
        let c = Self::dot(q1, q2).clamp(0.0, 1.0);
        if c > T {
            return *q1;
        } else {
            let theta = Float::acos(c);
            let s = Float::recip(Float::sin(theta));
            return ((*q1) * Float::sin((1.0 - t) * theta) + (*q2) * Float::sin(t * theta)) * s;
        }
    }

    pub fn to_matrix(&self) -> Matrix4x4 {
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let w = self.w;

        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = x * w;
        let wy = y * w;
        let wz = z * w;

        let mut m = Matrix4x4::identity();
        m.m[4 * 0 + 0] = 1.0 - 2.0 * (yy + zz);
        m.m[4 * 0 + 1] = 2.0 * (xy + wz);
        m.m[4 * 0 + 2] = 2.0 * (xz - wy);
        m.m[4 * 1 + 0] = 2.0 * (xy - wz);
        m.m[4 * 1 + 1] = 1.0 - 2.0 * (xx + zz);
        m.m[4 * 1 + 2] = 2.0 * (yz + wx);
        m.m[4 * 2 + 0] = 2.0 * (xz + wy);
        m.m[4 * 2 + 1] = 2.0 * (yz - wx);
        m.m[4 * 2 + 2] = 1.0 - 2.0 * (xx + yy);
        return m.transpose();
    }

    pub fn from_angle_axis(theta: Float, axis: &Vector3f) -> Self {
        let theta = theta / 2.0;
        let sin_theta = Float::sin(theta);
        let cos_theta = Float::cos(theta);
        let v = axis.normalize() * sin_theta;
        return Quaternion::new(v.x, v.y, v.z, cos_theta);
    }

    pub fn from_matrix(m: &Matrix4x4) -> Self {
        let trace = m.m[0] + m.m[5] + m.m[10];
        if trace > 0.0 {
            // Compute w from matrix trace, then xyz
            // 4w^2 = m[0][0] + m[1][1] + m[2][2] + m[3][3] (but m[3][3] == 1)
            let s = Float::sqrt(trace + 1.0) * 2.0;
            let x = (m.m[4 * 2 + 1] - m.m[4 * 1 + 2]) / s; //21 12
            let y = (m.m[4 * 0 + 2] - m.m[4 * 2 + 0]) / s; //02 20
            let z = (m.m[4 * 1 + 0] - m.m[4 * 0 + 1]) / s; //10 01
            let w = s / 4.0;
            return Quaternion::new(x, y, z, w);
        } else if m.m[4 * 0 + 0] > m.m[4 * 1 + 1] && m.m[4 * 0 + 0] > m.m[4 * 2 + 2] {
            let s = Float::sqrt(1.0 + m.m[4 * 0 + 0] - m.m[4 * 1 + 1] - m.m[4 * 2 + 2]) * 2.0;
            let x = s / 4.0;
            let y = (m.m[4 * 1 + 0] + m.m[4 * 0 + 1]) / s;
            let z = (m.m[4 * 0 + 2] + m.m[4 * 2 + 0]) / s;
            let w = (m.m[4 * 2 + 1] - m.m[4 * 1 + 2]) / s;
            return Quaternion::new(x, y, z, w);
        } else if m.m[4 * 1 + 1] > m.m[4 * 2 + 2] {
            let s = Float::sqrt(1.0 + m.m[4 * 1 + 1] - m.m[4 * 0 + 0] - m.m[4 * 2 + 2]) * 2.0;
            let x = (m.m[4 * 1 + 0] + m.m[4 * 0 + 1]) / s;
            let y = s / 4.0;
            let z = (m.m[4 * 2 + 1] + m.m[4 * 1 + 2]) / s;
            let w = (m.m[4 * 0 + 2] - m.m[4 * 2 + 0]) / s;
            return Quaternion::new(x, y, z, w);
        } else {
            let s = Float::sqrt(1.0 + m.m[4 * 2 + 2] - m.m[4 * 0 + 0] - m.m[4 * 1 + 1]) * 2.0;
            let x = (m.m[4 * 0 + 2] + m.m[4 * 2 + 0]) / s;
            let y = (m.m[4 * 2 + 1] + m.m[4 * 1 + 2]) / s;
            let z = s / 4.0;
            let w = (m.m[4 * 1 + 0] - m.m[4 * 0 + 1]) / s;
            return Quaternion::new(x, y, z, w);
        }
    }
}

impl ops::Add<Quaternion> for Quaternion {
    type Output = Quaternion;
    fn add(self, rhs: Quaternion) -> Quaternion {
        Quaternion {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }
}

impl ops::Sub<Quaternion> for Quaternion {
    type Output = Quaternion;
    fn sub(self, rhs: Quaternion) -> Quaternion {
        Quaternion {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }
}

impl ops::Mul<Float> for Quaternion {
    type Output = Quaternion;
    fn mul(self, rhs: Float) -> Quaternion {
        Quaternion {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            w: self.w * rhs,
        }
    }
}

impl ops::Neg for Quaternion {
    type Output = Quaternion;
    fn neg(self) -> Quaternion {
        return Quaternion {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        };
    }
}

impl From<Matrix4x4> for Quaternion {
    fn from(m: Matrix4x4) -> Self {
        return Quaternion::from_matrix(&m);
    }
}
