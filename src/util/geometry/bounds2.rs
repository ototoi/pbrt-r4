use super::Vector2;
use crate::util::base::*;

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Bounds2<T> {
    pub min: Vector2<T>,
    pub max: Vector2<T>,
}

pub type Bounds2f = Bounds2<Float>;
pub type Bounds2i = Bounds2<i32>;

impl<T: Copy + PartialOrd> Bounds2<T> {
    pub fn new(v0: &Vector2<T>, v1: &Vector2<T>) -> Self {
        let min = Vector2::<T>::new(min_(v0.x, v1.x), min_(v0.y, v1.y));
        let max = Vector2::<T>::new(max_(v0.x, v1.x), max_(v0.y, v1.y));
        Bounds2::<T> { min, max }
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
    > Bounds2<T>
{
    pub fn area(&self) -> T {
        return (self.max.x - self.min.x) * (self.max.y - self.min.y);
    }
    pub fn diagonal(&self) -> Vector2<T> {
        return self.max - self.min;
    }
    pub fn offset(&self, p: &Vector2<T>) -> Vector2<T> {
        let mut o = *p - self.min;
        if self.max.x > self.min.x {
            o.x = o.x / (self.max.x - self.min.x);
        }
        if self.max.y > self.min.y {
            o.y = o.y / (self.max.y - self.min.y);
        }
        return o;
    }

    pub fn expand(&self, delta: T) -> Self {
        let delta = Vector2::<T>::new(delta, delta);
        return Bounds2 {
            min: self.min - delta,
            max: self.max + delta,
        };
    }

    pub fn union(&self, other: &Self) -> Self {
        let min = Vector2::<T>::new(min_(self.min.x, other.min.x), min_(self.min.y, other.min.y));
        let max = Vector2::<T>::new(max_(self.max.x, other.max.x), max_(self.max.y, other.max.y));
        return Bounds2 { min, max };
    }

    pub fn intersect(&self, other: &Self) -> Self {
        let min = Vector2::<T>::new(max_(self.min.x, other.min.x), max_(self.min.y, other.min.y));
        let max = Vector2::<T>::new(min_(self.max.x, other.max.x), min_(self.max.y, other.max.y));
        return Bounds2 { min, max };
    }

    pub fn union_p(&self, other: &Vector2<T>) -> Self {
        let min = Vector2::<T>::new(min_(self.min.x, other.x), min_(self.min.y, other.y));
        let max = Vector2::<T>::new(max_(self.max.x, other.x), max_(self.max.y, other.y));
        return Bounds2 { min, max };
    }

    pub fn inside(&self, p: &Vector2<T>) -> bool {
        return p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y;
    }

    pub fn inside_exclusive(&self, p: &Vector2<T>) -> bool {
        return p.x >= self.min.x && p.x < self.max.x && p.y >= self.min.y && p.y < self.max.y;
    }
}

impl Bounds2f {
    pub fn lerp(&self, t: &Vector2f) -> Vector2f {
        return Vector2f::new(
            lerp(t.x, self.min.x, self.max.x),
            lerp(t.y, self.min.y, self.max.y),
        );
    }
}

impl<T: Copy + PartialOrd> From<((T, T), (T, T))> for Bounds2<T> {
    fn from(value: ((T, T), (T, T))) -> Self {
        let min = Vector2::<T>::from(value.0);
        let max = Vector2::<T>::from(value.1);
        Bounds2::<T>::new(&min, &max)
    }
}

impl<T: Copy> From<(T, T)> for Bounds2<T> {
    fn from(value: (T, T)) -> Self {
        Bounds2::<T> {
            min: Vector2::<T>::from(value),
            max: Vector2::<T>::from(value),
        }
    }
}

pub struct Bounds2iIterator<'a> {
    p: Point2i,
    bounds: &'a Bounds2i,
}

impl<'a> Iterator for Bounds2iIterator<'a> {
    type Item = Point2i;

    fn next(&mut self) -> Option<Self::Item> {
        if self.p.y < self.bounds.max.y && self.p.x < self.bounds.max.x {
            let old = self.p;
            self.p.x += 1;
            if self.p.x == self.bounds.max.x {
                self.p.x = self.bounds.min.x;
                self.p.y += 1;
            }
            return Some(old);
        } else {
            return None;
        }
    }
}

impl<'a> IntoIterator for &'a Bounds2i {
    type Item = Point2i;
    type IntoIter = Bounds2iIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Bounds2iIterator {
            p: self.min,
            bounds: self,
        }
    }
}

impl Default for Bounds2i {
    fn default() -> Self {
        Bounds2i {
            min: Point2i::new(std::i32::MAX, std::i32::MAX),
            max: Point2i::new(std::i32::MIN, std::i32::MIN),
        }
    }
}

impl Default for Bounds2f {
    fn default() -> Self {
        Bounds2f {
            min: Point2f::new(Float::MAX, Float::MAX),
            max: Point2f::new(Float::MIN, Float::MIN),
        }
    }
}
