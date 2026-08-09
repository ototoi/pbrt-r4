use super::numeric_traits::*;
use std::ops;

#[derive(Debug, PartialEq, Default, Copy, Clone)]
pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}

impl<T: Copy> Vector2<T> {
    pub fn new(x: T, y: T) -> Self {
        Vector2::<T> { x, y }
    }
}

impl<T: Default> Vector2<T> {
    #[inline]
    pub fn zero() -> Self {
        Vector2::<T> {
            x: T::default(),
            y: T::default(),
        }
    }
}

impl<T: Copy + NumberType> Vector2<T> {
    #[inline]
    pub fn abs(&self) -> Self {
        Vector2::<T> {
            x: NumberType::abs(self.x),
            y: NumberType::abs(self.y),
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
    > Vector2<T>
{
    #[inline]
    pub fn abs_dot(&self, rhs: &Self) -> T {
        return NumberType::abs(self.dot(rhs));
    }
}

impl<
        T: Copy
            + FloatType
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::MulAssign<T>,
    > Vector2<T>
{
    #[inline]
    pub fn dot(&self, rhs: &Self) -> T {
        return self.x * rhs.x + self.y * rhs.y;
    }

    #[inline]
    pub fn length_squared(&self) -> T {
        return self.dot(self);
    }

    #[inline]
    pub fn length(&self) -> T {
        return FloatType::sqrt(self.x * self.x + self.y * self.y);
    }

    #[inline]
    pub fn normalize(&self) -> Self {
        let l = self.length();
        Vector2::<T> {
            x: self.x / l,
            y: self.y / l,
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
    pub fn mul_assign(&mut self, f: T) {
        self.x *= f;
        self.y *= f;
    }
}

// Add
impl<T: std::ops::Add<Output = T>> ops::Add<Vector2<T>> for Vector2<T> {
    type Output = Vector2<T>;
    #[inline]
    fn add(self, rhs: Vector2<T>) -> Vector2<T> {
        return Vector2 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        };
    }
}

// Sub
impl<T: std::ops::Sub<Output = T>> ops::Sub<Vector2<T>> for Vector2<T> {
    type Output = Vector2<T>;
    #[inline]
    fn sub(self, rhs: Vector2<T>) -> Vector2<T> {
        return Vector2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        };
    }
}

// Mul
// V x V
impl<T: std::ops::Mul<Output = T>> ops::Mul<Vector2<T>> for Vector2<T> {
    type Output = Vector2<T>;
    #[inline]
    fn mul(self, rhs: Vector2<T>) -> Vector2<T> {
        return Vector2 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        };
    }
}

// Mul Scalar
//V x T
impl<T: std::ops::Mul<Output = T> + Copy> ops::Mul<T> for Vector2<T> {
    type Output = Vector2<T>;
    #[inline]
    fn mul(self, rhs: T) -> Vector2<T> {
        return Vector2 {
            x: self.x * rhs,
            y: self.y * rhs,
        };
    }
}

impl<T: std::ops::Div<Output = T> + Copy> ops::Div<T> for Vector2<T> {
    type Output = Vector2<T>;
    #[inline]
    fn div(self, rhs: T) -> Vector2<T> {
        return Vector2 {
            x: self.x / rhs,
            y: self.y / rhs,
        };
    }
}

impl ops::Mul<Vector2<f32>> for f32 {
    type Output = Vector2<f32>;
    #[inline]
    fn mul(self, rhs: Vector2<f32>) -> Vector2<f32> {
        return Vector2 {
            x: self * rhs.x,
            y: self * rhs.y,
        };
    }
}

impl ops::Mul<Vector2<f64>> for f64 {
    type Output = Vector2<f64>;
    #[inline]
    fn mul(self, rhs: Vector2<f64>) -> Vector2<f64> {
        return Vector2 {
            x: self * rhs.x,
            y: self * rhs.y,
        };
    }
}

impl ops::Mul<Vector2<i32>> for i32 {
    type Output = Vector2<i32>;
    #[inline]
    fn mul(self, rhs: Vector2<i32>) -> Vector2<i32> {
        return Vector2 {
            x: self * rhs.x,
            y: self * rhs.y,
        };
    }
}

impl<T: std::ops::Neg<Output = T>> ops::Neg for Vector2<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        return Vector2 {
            x: -self.x,
            y: -self.y,
        };
    }
}

impl<T: std::ops::AddAssign<T>> ops::AddAssign for Vector2<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<T: std::ops::SubAssign<T>> ops::SubAssign for Vector2<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl<T: std::ops::MulAssign<T>> ops::MulAssign for Vector2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
    }
}

impl<T: std::ops::MulAssign<T> + Copy> ops::MulAssign<T> for Vector2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl<T: std::ops::DivAssign<T>> ops::DivAssign<Vector2<T>> for Vector2<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
    }
}

impl<T: std::ops::DivAssign<T> + Copy> ops::DivAssign<T> for Vector2<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl<T> ops::Index<usize> for Vector2<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        match i {
            0 => &self.x,
            _ => &self.y,
        }
    }
}

impl<T> ops::IndexMut<usize> for Vector2<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        match i {
            0 => &mut self.x,
            _ => &mut self.y,
        }
    }
}

impl<T: Copy> From<T> for Vector2<T> {
    #[inline]
    fn from(value: T) -> Self {
        Vector2::<T>::new(value, value)
    }
}

impl<T: Copy> From<(T, T)> for Vector2<T> {
    #[inline]
    fn from(value: (T, T)) -> Self {
        Vector2::<T>::new(value.0, value.1)
    }
}

impl<T: Copy> From<[T; 2]> for Vector2<T> {
    #[inline]
    fn from(value: [T; 2]) -> Self {
        Vector2::<T>::new(value[0], value[1])
    }
}

impl<T: Copy> From<&[T]> for Vector2<T> {
    #[inline]
    fn from(value: &[T]) -> Self {
        Vector2::<T>::new(value[0], value[1])
    }
}
