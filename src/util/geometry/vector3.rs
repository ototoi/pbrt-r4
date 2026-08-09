use super::numeric_traits::*;
use std::ops;

#[derive(Debug, PartialEq, Default, Copy, Clone)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Copy> Vector3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Vector3::<T> { x, y, z }
    }
}

impl<T: Copy + Default> Vector3<T> {
    #[inline]
    pub fn zero() -> Self {
        Vector3::<T> {
            x: T::default(),
            y: T::default(),
            z: T::default(),
        }
    }
}

impl<T: Copy + NumberType> Vector3<T> {
    #[inline]
    pub fn abs(&self) -> Self {
        Vector3::<T> {
            x: NumberType::abs(self.x),
            y: NumberType::abs(self.y),
            z: NumberType::abs(self.z),
        }
    }
}

impl<
        T: Copy
            + PartialEq
            + FloatType
            + NumberType
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::MulAssign<T>,
    > Vector3<T>
{
    #[inline]
    pub fn abs_dot(&self, rhs: &Self) -> T {
        return NumberType::abs(self.dot(rhs));
    }
}

impl<
        T: Copy
            + PartialEq
            + PartialOrd
            + FloatType
            + NumberType
            + Default
            + std::ops::Neg<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::MulAssign<T>,
    > Vector3<T>
{
    pub fn coordinate_system(d1: &Self) -> (Self, Self, Self) {
        let v1 = d1.normalize();
        if T::abs(v1.x) > T::abs(v1.y) {
            let v2 = Self::new(-v1.z, T::default(), v1.x).normalize();
            let v3 = Self::cross(&v1, &v2).normalize();
            return (v1, v2, v3);
        } else {
            let v2 = Self::new(T::default(), v1.z, -v1.y).normalize();
            let v3 = Self::cross(&v1, &v2).normalize();
            return (v1, v2, v3);
        }
    }
}

impl<
        T: Copy
            + PartialEq
            + FloatType
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::MulAssign<T>,
    > Vector3<T>
{
    #[inline]
    pub fn dot(&self, rhs: &Self) -> T {
        return self.x * rhs.x + self.y * rhs.y + self.z * rhs.z;
    }

    #[inline]
    pub fn length_squared(&self) -> T {
        return self.dot(self);
    }

    #[inline]
    pub fn length(&self) -> T {
        return FloatType::sqrt(self.x * self.x + self.y * self.y + self.z * self.z);
    }

    #[inline]
    pub fn normalize(&self) -> Self {
        let l = self.length();
        Vector3::<T> {
            x: self.x / l,
            y: self.y / l,
            z: self.z / l,
        }
    }

    #[inline]
    pub fn distance_squared(a: &Self, b: &Self) -> T {
        let v = *a - *b;
        return v.dot(&v);
    }

    #[inline]
    pub fn distance(a: &Self, b: &Self) -> T {
        return FloatType::sqrt(Self::distance_squared(a, b));
    }

    #[inline]
    pub fn cross(v1: &Self, v2: &Self) -> Self {
        Vector3::<T> {
            x: (v1.y * v2.z) - (v1.z * v2.y),
            y: (v1.z * v2.x) - (v1.x * v2.z),
            z: (v1.x * v2.y) - (v1.y * v2.x),
        }
    }

    #[inline]
    pub fn mul_assign(&mut self, f: T) {
        self.x *= f;
        self.y *= f;
        self.z *= f;
    }
}

// Add
impl<T: std::ops::Add<Output = T>> ops::Add<Vector3<T>> for Vector3<T> {
    type Output = Vector3<T>;
    #[inline]
    fn add(self, rhs: Vector3<T>) -> Vector3<T> {
        return Vector3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        };
    }
}

// Sub
impl<T: std::ops::Sub<Output = T>> ops::Sub<Vector3<T>> for Vector3<T> {
    type Output = Vector3<T>;
    #[inline]
    fn sub(self, rhs: Vector3<T>) -> Vector3<T> {
        return Vector3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        };
    }
}

// Mul
// V x V
impl<T: std::ops::Mul<Output = T>> ops::Mul<Vector3<T>> for Vector3<T> {
    type Output = Vector3<T>;
    #[inline]
    fn mul(self, rhs: Vector3<T>) -> Vector3<T> {
        return Vector3 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        };
    }
}

// Mul Scalar
//V x T
impl<T: std::ops::Mul<Output = T> + Copy> ops::Mul<T> for Vector3<T> {
    type Output = Vector3<T>;
    #[inline]
    fn mul(self, rhs: T) -> Vector3<T> {
        return Vector3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        };
    }
}

impl<T: std::ops::Div<Output = T> + Copy> ops::Div<T> for Vector3<T> {
    type Output = Vector3<T>;
    #[inline]
    fn div(self, rhs: T) -> Vector3<T> {
        return Vector3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        };
    }
}

impl ops::Mul<Vector3<f32>> for f32 {
    type Output = Vector3<f32>;
    #[inline]
    fn mul(self, rhs: Vector3<f32>) -> Vector3<f32> {
        return Vector3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        };
    }
}

impl ops::Mul<Vector3<f64>> for f64 {
    type Output = Vector3<f64>;
    #[inline]
    fn mul(self, rhs: Vector3<f64>) -> Vector3<f64> {
        return Vector3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        };
    }
}

impl ops::Mul<Vector3<i32>> for i32 {
    type Output = Vector3<i32>;
    #[inline]
    fn mul(self, rhs: Vector3<i32>) -> Vector3<i32> {
        return Vector3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        };
    }
}

impl<T: std::ops::Neg<Output = T>> ops::Neg for Vector3<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        return Vector3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        };
    }
}

impl<T: std::ops::AddAssign<T>> ops::AddAssign for Vector3<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl<T: std::ops::SubAssign<T>> ops::SubAssign for Vector3<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl<T: std::ops::MulAssign<T>> ops::MulAssign<Vector3<T>> for Vector3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
    }
}

impl<T: std::ops::MulAssign<T> + Copy> ops::MulAssign<T> for Vector3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl<T: std::ops::DivAssign<T>> ops::DivAssign<Vector3<T>> for Vector3<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl<T: std::ops::DivAssign<T> + Copy> ops::DivAssign<T> for Vector3<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl<T> ops::Index<usize> for Vector3<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        match i {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z,
        }
    }
}

impl<T> ops::IndexMut<usize> for Vector3<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => &mut self.z,
        }
    }
}

impl<T: Copy> From<T> for Vector3<T> {
    #[inline]
    fn from(value: T) -> Self {
        Vector3::<T>::new(value, value, value)
    }
}

impl<T: Copy> From<(T, T, T)> for Vector3<T> {
    #[inline]
    fn from(value: (T, T, T)) -> Self {
        Vector3::<T>::new(value.0, value.1, value.2)
    }
}

impl<T: Copy> From<[T; 3]> for Vector3<T> {
    #[inline]
    fn from(value: [T; 3]) -> Self {
        Vector3::<T>::new(value[0], value[1], value[2])
    }
}

impl<T: Copy> From<&[T]> for Vector3<T> {
    #[inline]
    fn from(value: &[T]) -> Self {
        Vector3::<T>::new(value[0], value[1], value[2])
    }
}
