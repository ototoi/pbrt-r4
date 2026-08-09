use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::spectrum::*;

pub trait SampledGridValue: Clone {
    type Output: Copy;

    fn zero() -> Self::Output;
    fn sample(&self, lambda: Option<&SampledWavelengths>) -> Self::Output;
    fn lerp(t: Float, a: Self::Output, b: Self::Output) -> Self::Output;
    fn max_value(&self) -> Float;
}

impl SampledGridValue for Float {
    type Output = Float;

    fn zero() -> Self::Output {
        0.0
    }

    fn sample(&self, _lambda: Option<&SampledWavelengths>) -> Self::Output {
        *self
    }

    fn lerp(t: Float, a: Self::Output, b: Self::Output) -> Self::Output {
        lerp(t, a, b)
    }

    fn max_value(&self) -> Float {
        *self
    }
}

impl SampledGridValue for RGBUnboundedSpectrum {
    type Output = SampledSpectrum;

    fn zero() -> Self::Output {
        SampledSpectrum::zero()
    }

    fn sample(&self, lambda: Option<&SampledWavelengths>) -> Self::Output {
        let default_lambda;
        let lambda = match lambda {
            Some(lambda) => lambda,
            None => {
                default_lambda = SampledWavelengths::default();
                &default_lambda
            }
        };
        RGBUnboundedSpectrum::sample(self, lambda)
    }

    fn lerp(t: Float, a: Self::Output, b: Self::Output) -> Self::Output {
        a * (1.0 - t) + b * t
    }

    fn max_value(&self) -> Float {
        RGBUnboundedSpectrum::max_value(self)
    }
}

impl SampledGridValue for RGBIlluminantSpectrum {
    type Output = SampledSpectrum;

    fn zero() -> Self::Output {
        SampledSpectrum::zero()
    }

    fn sample(&self, lambda: Option<&SampledWavelengths>) -> Self::Output {
        let default_lambda;
        let lambda = match lambda {
            Some(lambda) => lambda,
            None => {
                default_lambda = SampledWavelengths::default();
                &default_lambda
            }
        };
        RGBIlluminantSpectrum::sample(self, lambda)
    }

    fn lerp(t: Float, a: Self::Output, b: Self::Output) -> Self::Output {
        a * (1.0 - t) + b * t
    }

    fn max_value(&self) -> Float {
        RGBIlluminantSpectrum::max_value(self)
    }
}

#[derive(Debug, Clone)]
pub struct SampledGrid<T: SampledGridValue> {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub values: Vec<T>,
}

impl<T: SampledGridValue> SampledGrid<T> {
    pub fn new(nx: u32, ny: u32, nz: u32, values: Vec<T>) -> Self {
        Self { nx, ny, nz, values }
    }

    pub fn bytes_allocated(&self) -> usize {
        std::mem::size_of::<T>() * self.values.len()
    }

    pub fn lookup(&self, p: &Point3f, lambda: Option<&SampledWavelengths>) -> T::Output {
        let p_samples = Point3f::new(
            p.x * self.nx as Float - 0.5,
            p.y * self.ny as Float - 0.5,
            p.z * self.nz as Float - 0.5,
        );
        let ix = p_samples.x.floor() as i32;
        let iy = p_samples.y.floor() as i32;
        let iz = p_samples.z.floor() as i32;
        let pi = Point3i::new(ix, iy, iz);
        let dx = p_samples.x - ix as Float;
        let dy = p_samples.y - iy as Float;
        let dz = p_samples.z - iz as Float;

        let d00 = T::lerp(
            dx,
            self.sample_value(&(pi + Vector3i::new(0, 0, 0)), lambda),
            self.sample_value(&(pi + Vector3i::new(1, 0, 0)), lambda),
        );
        let d10 = T::lerp(
            dx,
            self.sample_value(&(pi + Vector3i::new(0, 1, 0)), lambda),
            self.sample_value(&(pi + Vector3i::new(1, 1, 0)), lambda),
        );
        let d01 = T::lerp(
            dx,
            self.sample_value(&(pi + Vector3i::new(0, 0, 1)), lambda),
            self.sample_value(&(pi + Vector3i::new(1, 0, 1)), lambda),
        );
        let d11 = T::lerp(
            dx,
            self.sample_value(&(pi + Vector3i::new(0, 1, 1)), lambda),
            self.sample_value(&(pi + Vector3i::new(1, 1, 1)), lambda),
        );
        T::lerp(dz, T::lerp(dy, d00, d10), T::lerp(dy, d01, d11))
    }

    pub fn max_value(&self, bounds: &Bounds3f) -> Float {
        let size = [self.nx as i32, self.ny as i32, self.nz as i32];
        let p0 = Point3f::new(
            bounds.min.x * size[0] as Float - 0.5,
            bounds.min.y * size[1] as Float - 0.5,
            bounds.min.z * size[2] as Float - 0.5,
        );
        let p1 = Point3f::new(
            bounds.max.x * size[0] as Float - 0.5,
            bounds.max.y * size[1] as Float - 0.5,
            bounds.max.z * size[2] as Float - 0.5,
        );
        let min_p = [
            i32::max(p0.x.floor() as i32, 0),
            i32::max(p0.y.floor() as i32, 0),
            i32::max(p0.z.floor() as i32, 0),
        ];
        let max_p = [
            i32::min(p1.x.floor() as i32 + 1, size[0] - 1),
            i32::min(p1.y.floor() as i32 + 1, size[1] - 1),
            i32::min(p1.z.floor() as i32 + 1, size[2] - 1),
        ];
        if min_p[0] > max_p[0] || min_p[1] > max_p[1] || min_p[2] > max_p[2] {
            return 0.0;
        }

        let mut max_value = 0.0;
        for z in min_p[2]..=max_p[2] {
            for y in min_p[1]..=max_p[1] {
                for x in min_p[0]..=max_p[0] {
                    let index = ((z * size[1] + y) * size[0] + x) as usize;
                    max_value = Float::max(max_value, self.values[index].max_value());
                }
            }
        }
        max_value
    }

    fn sample_value(&self, p: &Point3i, lambda: Option<&SampledWavelengths>) -> T::Output {
        self.grid_index(p)
            .map(|i| self.values[i].sample(lambda))
            .unwrap_or_else(T::zero)
    }

    fn grid_index(&self, p: &Point3i) -> Option<usize> {
        let nx = self.nx as i32;
        let ny = self.ny as i32;
        let nz = self.nz as i32;
        let sample_bounds = Bounds3i::new(&Point3i::new(0, 0, 0), &Point3i::new(nx, ny, nz));
        if !sample_bounds.inside_exclusive(p) {
            return None;
        }
        Some(((p.z * ny + p.y) * nx + p.x) as usize)
    }
}
