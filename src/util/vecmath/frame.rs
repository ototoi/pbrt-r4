use crate::util::base::*;

/// Frame for coordinate system transformations
#[derive(Debug, Clone)]
pub struct Frame {
    pub x: Vector3f,
    pub y: Vector3f,
    pub z: Vector3f,
}

impl Frame {
    /// Create a frame from X and Z vectors
    pub fn from_xz(x: Vector3f, z: Vector3f) -> Self {
        let x = x.normalize();
        let z = z.normalize();
        let y = Vector3f::cross(&z, &x).normalize();
        Frame { x, y, z }
    }

    /// pbrt-v4 `Frame::FromZ(n)` -- build an orthonormal frame whose
    /// local +Z is `z` (vecmath.h:1063).
    pub fn from_z(z: Vector3f) -> Self {
        let z = z.normalize();
        let (x, y) = coordinate_system(&z);
        Frame { x, y, z }
    }

    /// pbrt-v4 `Frame::FromXY(x, y)` (vecmath.h ~1055) -- build an
    /// orthonormal frame from two orthogonal unit vectors x, y; z is
    /// derived as x × y so the resulting basis is right-handed.
    pub fn from_xy(x: Vector3f, y: Vector3f) -> Self {
        let x = x.normalize();
        let y = y.normalize();
        let z = Vector3f::cross(&x, &y).normalize();
        Frame { x, y, z }
    }

    /// pbrt-v4 `Frame::FromX(x)` (vecmath.h): build an orthonormal
    /// frame whose local +X is `x`, choosing y / z via
    /// `CoordinateSystem(x, &y, &z)`.
    pub fn from_x(x: Vector3f) -> Self {
        let x = x.normalize();
        let (y, z) = coordinate_system(&x);
        Frame { x, y, z }
    }

    /// pbrt-v4 `Frame::FromY(y)` (vecmath.h): build an orthonormal
    /// frame whose local +Y is `y`; v4 calls
    /// `CoordinateSystem(y, &z, &x)` so the resulting (x, y, z) basis
    /// stays right-handed.
    pub fn from_y(y: Vector3f) -> Self {
        let y = y.normalize();
        let (z, x) = coordinate_system(&y);
        Frame { x, y, z }
    }

    /// Transform vector from world to local coordinates
    pub fn to_local(&self, v: Vector3f) -> Vector3f {
        Vector3f::new(
            Vector3f::dot(&v, &self.x),
            Vector3f::dot(&v, &self.y),
            Vector3f::dot(&v, &self.z),
        )
    }

    /// Transform vector from local to world coordinates
    pub fn from_local(&self, v: Vector3f) -> Vector3f {
        self.x * v.x + self.y * v.y + self.z * v.z
    }
}

/// pbrt-v4 `void CoordinateSystem(const Vector3f &v1, Vector3f *v2,
/// Vector3f *v3)` (vecmath.h:1031). Returns `(v2, v3)`.
pub fn coordinate_system(v1: &Vector3f) -> (Vector3f, Vector3f) {
    let sign = (1.0 as Float).copysign(v1.z);
    let a = -1.0 / (sign + v1.z);
    let b = v1.x * v1.y * a;
    let v2 = Vector3f::new(1.0 + sign * v1.x * v1.x * a, sign * b, -sign * v1.x);
    let v3 = Vector3f::new(b, sign + v1.y * v1.y * a, -v1.y);
    (v2, v3)
}
