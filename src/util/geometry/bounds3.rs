use super::intersect::*;
use super::vector3::*;
use crate::util::base::*;
use crate::util::geometry::*;

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Bounds3<T> {
    pub min: Vector3<T>,
    pub max: Vector3<T>,
}

pub type Bounds3f = Bounds3<Float>;
pub type Bounds3i = Bounds3<i32>;

impl<T: Copy + PartialOrd> Bounds3<T> {
    pub fn new(v0: &Vector3<T>, v1: &Vector3<T>) -> Self {
        let min = Vector3::<T>::new(min_(v0.x, v1.x), min_(v0.y, v1.y), min_(v0.z, v1.z));
        let max = Vector3::<T>::new(max_(v0.x, v1.x), max_(v0.y, v1.y), max_(v0.z, v1.z));
        Bounds3::<T> { min, max }
    }

    pub fn corner(&self, i: usize) -> Vector3<T> {
        assert!(i < 8);
        return match i {
            0 => Vector3::<T>::new(self.min.x, self.min.y, self.min.z),
            1 => Vector3::<T>::new(self.max.x, self.min.y, self.min.z),
            2 => Vector3::<T>::new(self.min.x, self.max.y, self.min.z),
            3 => Vector3::<T>::new(self.max.x, self.max.y, self.min.z),
            4 => Vector3::<T>::new(self.min.x, self.min.y, self.max.z),
            5 => Vector3::<T>::new(self.max.x, self.min.y, self.max.z),
            6 => Vector3::<T>::new(self.min.x, self.max.y, self.max.z),
            7 => Vector3::<T>::new(self.max.x, self.max.y, self.max.z),
            _ => unreachable!(),
        };
    }
}

fn min_<T: Copy + PartialOrd>(a: T, b: T) -> T {
    return if a <= b { a } else { b };
}

fn max_<T: Copy + PartialOrd>(a: T, b: T) -> T {
    return if a >= b { a } else { b };
}

impl<
        T: Copy
            + PartialOrd
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    > Bounds3<T>
{
    pub fn area(&self) -> T {
        return (self.max.x - self.min.x) * (self.max.y - self.min.y);
    }
    pub fn diagonal(&self) -> Vector3<T> {
        return self.max - self.min;
    }
    pub fn maximum_extent(&self) -> usize {
        let d = self.diagonal();
        if d.x > d.y && d.x > d.z {
            return 0;
        } else if d.y > d.z {
            return 1;
        } else {
            return 2;
        }
    }

    pub fn offset(&self, p: &Vector3<T>) -> Vector3<T> {
        let mut o = *p - self.min;
        if self.max.x > self.min.x {
            o.x = o.x / (self.max.x - self.min.x);
        }
        if self.max.y > self.min.y {
            o.y = o.y / (self.max.y - self.min.y);
        }
        if self.max.z > self.min.z {
            o.z = o.z / (self.max.z - self.min.z);
        }
        return o;
    }

    pub fn expand(&self, delta: T) -> Self {
        let delta = Vector3::<T>::new(delta, delta, delta);
        return Bounds3 {
            min: self.min - delta,
            max: self.max + delta,
        };
    }

    pub fn union(&self, other: &Self) -> Self {
        let min = Vector3::<T>::new(
            min_(self.min.x, other.min.x),
            min_(self.min.y, other.min.y),
            min_(self.min.z, other.min.z),
        );
        let max = Vector3::<T>::new(
            max_(self.max.x, other.max.x),
            max_(self.max.y, other.max.y),
            max_(self.max.z, other.max.z),
        );
        return Bounds3 { min, max };
    }

    pub fn intersect(&self, other: &Self) -> Self {
        let min = Vector3::<T>::new(
            max_(self.min.x, other.min.x),
            max_(self.min.y, other.min.y),
            max_(self.min.z, other.min.z),
        );
        let max = Vector3::<T>::new(
            min_(self.max.x, other.max.x),
            min_(self.max.y, other.max.y),
            min_(self.max.z, other.max.z),
        );
        return Bounds3 { min, max };
    }

    pub fn union_p(&self, other: &Vector3<T>) -> Self {
        let min = Vector3::<T>::new(
            min_(self.min.x, other.x),
            min_(self.min.y, other.y),
            min_(self.min.z, other.z),
        );
        let max = Vector3::<T>::new(
            max_(self.max.x, other.x),
            max_(self.max.y, other.y),
            max_(self.max.z, other.z),
        );
        return Bounds3 { min, max };
    }

    pub fn inside_exclusive(&self, p: &Vector3<T>) -> bool {
        return p.x >= self.min.x
            && p.x < self.max.x
            && p.y >= self.min.y
            && p.y < self.max.y
            && p.z >= self.min.z
            && p.z < self.max.z;
    }
}

impl Bounds3f {
    pub fn intersect_p(&self, ray: &Ray, t_max: Float) -> Option<(Float, Float)> {
        let (b, tmin, tmax) = intersect_box(&self.min, &self.max, &ray.o, &ray.d, 0.0, t_max);
        if b {
            return Some((tmin, tmax));
        } else {
            return None;
        }
    }

    pub fn lerp(&self, t: &Vector3f) -> Vector3f {
        return Vector3f::new(
            lerp(t.x, self.min.x, self.max.x),
            lerp(t.y, self.min.y, self.max.y),
            lerp(t.z, self.min.z, self.max.z),
        );
    }

    pub fn bounding_sphere(&self) -> (Vector3f, Float) {
        let center = (self.min + self.max) * 0.5;
        let radius = (self.max - self.min).length() * 0.5;
        return (center, radius);
    }

    pub fn surface_area(&self) -> Float {
        let d = self.diagonal();
        return 2.0 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    pub fn distance_squared(&self, p: &Point3f) -> Float {
        let mut d = 0.0;
        for i in 0..3 {
            let delta = max_(0.0, max_(self.min[i] - p[i], p[i] - self.max[i]));
            d += delta * delta;
        }
        return d;
    }

    pub fn distance(&self, p: &Point3f) -> Float {
        return self.distance_squared(p).sqrt();
    }
}

impl<T: Copy + PartialOrd> From<((T, T, T), (T, T, T))> for Bounds3<T> {
    fn from(value: ((T, T, T), (T, T, T))) -> Self {
        let min = Vector3::<T>::from(value.0);
        let max = Vector3::<T>::from(value.1);
        Bounds3::<T>::new(&min, &max)
    }
}

impl<T: Copy> From<(T, T, T)> for Bounds3<T> {
    fn from(value: (T, T, T)) -> Self {
        Bounds3::<T> {
            min: Vector3::<T>::from(value),
            max: Vector3::<T>::from(value),
        }
    }
}

impl Default for Bounds3i {
    fn default() -> Self {
        Bounds3i {
            min: Point3i::new(std::i32::MAX, std::i32::MAX, std::i32::MAX),
            max: Point3i::new(std::i32::MIN, std::i32::MIN, std::i32::MIN),
        }
    }
}

impl Default for Bounds3f {
    fn default() -> Self {
        Bounds3f {
            min: Point3f::new(Float::MAX, Float::MAX, Float::MAX),
            max: Point3f::new(Float::MIN, Float::MIN, Float::MIN),
        }
    }
}
