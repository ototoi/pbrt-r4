//! CPU sampling iterators corresponding to pbrt-v4's `Uniform*` and
//! `Stratified*` generators in `util/sampling.h`.

use crate::util::base::{Float, Point2f, Point3f};
use crate::util::lowdiscrepancy::radical_inverse;
use crate::util::rng::RNG;

pub struct Uniform1D {
    index: usize,
    end: usize,
    rng: RNG,
}

impl Uniform1D {
    pub fn new(n: usize, sequence_index: u64) -> Self {
        Self {
            index: 0,
            end: n,
            rng: RNG::new_sequence(sequence_index),
        }
    }
}

impl Iterator for Uniform1D {
    type Item = Float;

    fn next(&mut self) -> Option<Self::Item> {
        (self.index < self.end).then(|| {
            self.index += 1;
            self.rng.uniform_float()
        })
    }
}

pub struct Uniform2D {
    inner: Uniform1D,
}

impl Uniform2D {
    pub fn new(n: usize, sequence_index: u64) -> Self {
        Self {
            inner: Uniform1D::new(n, sequence_index),
        }
    }
}

impl Iterator for Uniform2D {
    type Item = Point2f;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|x| Point2f::new(x, self.inner.rng.uniform_float()))
    }
}

pub struct Uniform3D {
    inner: Uniform1D,
}

impl Uniform3D {
    pub fn new(n: usize, sequence_index: u64) -> Self {
        Self {
            inner: Uniform1D::new(n, sequence_index),
        }
    }
}

impl Iterator for Uniform3D {
    type Item = Point3f;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|x| {
            Point3f::new(
                x,
                self.inner.rng.uniform_float(),
                self.inner.rng.uniform_float(),
            )
        })
    }
}

pub struct Stratified1D {
    index: usize,
    n: usize,
    rng: RNG,
}

impl Stratified1D {
    pub fn new(n: usize, sequence_index: u64) -> Self {
        assert!(n > 0);
        Self {
            index: 0,
            n,
            rng: RNG::new_sequence(sequence_index),
        }
    }
}

impl Iterator for Stratified1D {
    type Item = Float;

    fn next(&mut self) -> Option<Self::Item> {
        (self.index < self.n).then(|| {
            let value = (self.index as Float + self.rng.uniform_float()) / self.n as Float;
            self.index += 1;
            value
        })
    }
}

pub struct Stratified2D {
    index: usize,
    nx: usize,
    ny: usize,
    rng: RNG,
}

impl Stratified2D {
    pub fn new(nx: usize, ny: usize, sequence_index: u64) -> Self {
        assert!(nx > 0 && ny > 0);
        Self {
            index: 0,
            nx,
            ny,
            rng: RNG::new_sequence(sequence_index),
        }
    }
}

impl Iterator for Stratified2D {
    type Item = Point2f;

    fn next(&mut self) -> Option<Self::Item> {
        (self.index < self.nx * self.ny).then(|| {
            let ix = self.index % self.nx;
            let iy = self.index / self.nx;
            self.index += 1;
            Point2f::new(
                (ix as Float + self.rng.uniform_float()) / self.nx as Float,
                (iy as Float + self.rng.uniform_float()) / self.ny as Float,
            )
        })
    }
}

pub struct Stratified3D {
    index: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    rng: RNG,
}

pub struct Hammersley2D {
    index: u64,
    n: u64,
}

impl Hammersley2D {
    pub fn new(n: u64) -> Self {
        Self { index: 0, n }
    }
}

impl Iterator for Hammersley2D {
    type Item = Point2f;

    fn next(&mut self) -> Option<Self::Item> {
        (self.index < self.n).then(|| {
            let i = self.index;
            self.index += 1;
            Point2f::new(i as Float / self.n as Float, radical_inverse(0, i))
        })
    }
}

pub struct Hammersley3D {
    index: u64,
    n: u64,
}

impl Hammersley3D {
    pub fn new(n: u64) -> Self {
        Self { index: 0, n }
    }
}

impl Iterator for Hammersley3D {
    type Item = Point3f;

    fn next(&mut self) -> Option<Self::Item> {
        (self.index < self.n).then(|| {
            let i = self.index;
            self.index += 1;
            Point3f::new(
                i as Float / self.n as Float,
                radical_inverse(0, i),
                radical_inverse(1, i),
            )
        })
    }
}

impl Stratified3D {
    pub fn new(nx: usize, ny: usize, nz: usize, sequence_index: u64) -> Self {
        assert!(nx > 0 && ny > 0 && nz > 0);
        Self {
            index: 0,
            nx,
            ny,
            nz,
            rng: RNG::new_sequence(sequence_index),
        }
    }
}

impl Iterator for Stratified3D {
    type Item = Point3f;

    fn next(&mut self) -> Option<Self::Item> {
        (self.index < self.nx * self.ny * self.nz).then(|| {
            let ix = self.index % self.nx;
            let iy = (self.index / self.nx) % self.ny;
            let iz = self.index / (self.nx * self.ny);
            self.index += 1;
            Point3f::new(
                (ix as Float + self.rng.uniform_float()) / self.nx as Float,
                (iy as Float + self.rng.uniform_float()) / self.ny as Float,
                (iz as Float + self.rng.uniform_float()) / self.nz as Float,
            )
        })
    }
}
