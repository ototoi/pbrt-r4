use super::transform::*;
use crate::util::base::*;

pub const MAX_TRANSFORMS: usize = 2;
pub const START_TRANSFORM_BITS: u32 = 1 << 0; //1
pub const END_TRANSFORM_BITS: u32 = 1 << 1; //2
pub const ALL_TRANSFORM_BITS: u32 = (1 << 2) - 1; //3 = 1 + 2

#[derive(Debug, Default, Copy, Clone)]
pub struct TransformSet {
    pub t: [Transform; MAX_TRANSFORMS],
}

impl TransformSet {
    pub fn new() -> Self {
        let t: [Transform; MAX_TRANSFORMS] = [Transform::identity(); 2];
        TransformSet { t }
    }

    pub fn to_transform(&self) -> Transform {
        return self.t[0];
    }

    pub fn len(&self) -> usize {
        return self.t.len();
    }

    pub fn is_empty(&self) -> bool {
        return self.t.is_empty();
    }

    pub fn inverse(&self) -> Self {
        let v: Vec<Transform> = self.t.iter().map(|t| t.inverse()).collect();
        let it: [Transform; MAX_TRANSFORMS] = v.try_into().unwrap();
        TransformSet { t: it }
    }

    pub fn mul_transform(&mut self, other: &Transform, mask: u32) {
        for i in 0..MAX_TRANSFORMS {
            let active = ((1 << i) & mask) != 0;
            if active {
                self.t[i] = self.t[i] * *other;
            }
        }
    }

    pub fn set_transform(&mut self, other: &Transform, mask: u32) {
        for i in 0..MAX_TRANSFORMS {
            let active = ((1 << i) & mask) != 0;
            if active {
                self.t[i] = *other;
            }
        }
    }

    pub fn set(&mut self, other: &TransformSet, mask: u32) {
        for i in 0..MAX_TRANSFORMS {
            let active = ((1 << i) & mask) != 0;
            if active {
                self.t[i] = other.t[i];
            }
        }
    }

    pub fn is_animated(&self) -> bool {
        for i in 0..(MAX_TRANSFORMS - 1) {
            if self.t[i] != self.t[i + 1] {
                return true;
            }
        }
        return false;
    }
}

impl std::ops::Index<usize> for TransformSet {
    type Output = Transform;
    fn index(&self, index: usize) -> &Self::Output {
        return &self.t[index];
    }
}

impl std::ops::IndexMut<usize> for TransformSet {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return &mut self.t[index];
    }
}

impl std::ops::Mul<TransformSet> for TransformSet {
    type Output = TransformSet;
    fn mul(self, t2: TransformSet) -> TransformSet {
        let v: Vec<Transform> = self
            .t
            .iter()
            .zip(t2.t.iter())
            .map(|(a, b)| *a * *b)
            .collect();
        let t: [Transform; MAX_TRANSFORMS] = v.try_into().unwrap();
        return TransformSet { t };
    }
}

impl From<[Float; 16]> for TransformSet {
    fn from(v: [Float; 16]) -> Self {
        let vv = Transform::from(v);
        let t: [Transform; MAX_TRANSFORMS] = [vv; MAX_TRANSFORMS];
        TransformSet { t }
    }
}
