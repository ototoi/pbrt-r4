// `RayMajorantSegment::sigma_maj` is stored as a `SampledSpectrum`
// (fixed `N_SPECTRUM_SAMPLES` floats, stack-allocated) to match
// pbrt-v4 `base/medium.h:54`. Storing it as `Spectrum` (heap-backed
// `Arc<[Float; 471]>`) caused the DDA loop in `majorant_segments` to
// allocate per voxel-step and pushed RSS to >15 GB on bunny-cloud-class
// scenes. Callers must therefore know the wavelengths up front, so
// `Medium::sample_ray` (and the `sample_t_maj` driver) take
// `&SampledWavelengths` mandatorily, matching v4's `Medium::SampleRay`.

use std::sync::Arc;

use crate::util::base::{Float, Vector3f};
use crate::util::geometry::ray::Ray;
use crate::util::geometry::Bounds3f;
use crate::util::spectrum::*;

/// Mirrors pbrt-v4 `base/medium.h:54`. `sigma_maj` is a
/// `SampledSpectrum` so that DDA loops do not allocate per segment.
#[derive(Debug, Clone)]
pub struct RayMajorantSegment {
    pub t_min: Float,
    pub t_max: Float,
    pub sigma_maj: SampledSpectrum,
}

#[derive(Debug, Clone)]
pub struct HomogeneousMajorantIterator {
    seg: Option<RayMajorantSegment>,
}

impl HomogeneousMajorantIterator {
    pub fn new(t_min: Float, t_max: Float, sigma_maj: SampledSpectrum) -> Self {
        Self {
            seg: Some(RayMajorantSegment {
                t_min,
                t_max,
                sigma_maj,
            }),
        }
    }

    pub fn next(&mut self) -> Option<RayMajorantSegment> {
        self.seg.take()
    }
}

#[derive(Debug, Clone)]
pub struct DDAMajorantIterator {
    t_min: Float,
    t_max: Float,
    sigma_t: SampledSpectrum,
    grid: MajorantGrid,
    next_crossing_t: [Float; 3],
    delta_t: [Float; 3],
    step: [i32; 3],
    voxel_limit: [i32; 3],
    voxel: [i32; 3],
}

#[derive(Debug, Clone)]
pub struct MajorantGrid {
    pub bounds: Bounds3f,
    pub res: [i32; 3],
    pub values: Arc<[Float]>,
}

impl MajorantGrid {
    pub fn new(bounds: Bounds3f, res: [i32; 3], values: Arc<[Float]>) -> Self {
        Self {
            bounds,
            res,
            values,
        }
    }

    pub fn lookup(&self, x: i32, y: i32, z: i32) -> Float {
        self.values[((z * self.res[1] + y) * self.res[0] + x) as usize]
    }
}

impl DDAMajorantIterator {
    pub fn new(
        ray: Ray,
        t_min: Float,
        t_max: Float,
        sigma_t: SampledSpectrum,
        grid: MajorantGrid,
    ) -> Self {
        let diagonal = grid.bounds.diagonal();
        let ray_grid = Ray::new(
            &grid.bounds.offset(&ray.o),
            &Vector3f::new(
                ray.d.x / diagonal.x,
                ray.d.y / diagonal.y,
                ray.d.z / diagonal.z,
            ),
            t_max,
            ray.time,
        );
        let grid_intersect = ray_grid.position(t_min);
        let mut voxel = [0i32; 3];
        let mut next_crossing_t = [Float::INFINITY; 3];
        let mut delta_t = [Float::INFINITY; 3];
        let mut step = [0i32; 3];
        let mut voxel_limit = [0i32; 3];

        for axis in 0..3 {
            voxel[axis] = i32::clamp(
                (grid_intersect[axis] * grid.res[axis] as Float) as i32,
                0,
                grid.res[axis] - 1,
            );
            delta_t[axis] = 1.0 / (ray_grid.d[axis].abs() * grid.res[axis] as Float);
            if ray_grid.d[axis] >= 0.0 {
                let next_voxel_pos = (voxel[axis] + 1) as Float / grid.res[axis] as Float;
                next_crossing_t[axis] =
                    t_min + (next_voxel_pos - grid_intersect[axis]) / ray_grid.d[axis];
                step[axis] = 1;
                voxel_limit[axis] = grid.res[axis];
            } else {
                let next_voxel_pos = voxel[axis] as Float / grid.res[axis] as Float;
                next_crossing_t[axis] =
                    t_min + (next_voxel_pos - grid_intersect[axis]) / ray_grid.d[axis];
                step[axis] = -1;
                voxel_limit[axis] = -1;
            }
        }

        Self {
            t_min,
            t_max,
            sigma_t,
            grid,
            next_crossing_t,
            delta_t,
            step,
            voxel_limit,
            voxel,
        }
    }

    pub fn next(&mut self) -> Option<RayMajorantSegment> {
        if self.t_min >= self.t_max {
            return None;
        }

        let bits = ((self.next_crossing_t[0] < self.next_crossing_t[1]) as usize) << 2
            | ((self.next_crossing_t[0] < self.next_crossing_t[2]) as usize) << 1
            | ((self.next_crossing_t[1] < self.next_crossing_t[2]) as usize);
        const CMP_TO_AXIS: [usize; 8] = [2, 1, 2, 1, 2, 2, 0, 0];
        let step_axis = CMP_TO_AXIS[bits];

        let t_voxel_exit = Float::min(self.t_max, self.next_crossing_t[step_axis]);
        let density_maj = self
            .grid
            .lookup(self.voxel[0], self.voxel[1], self.voxel[2]);
        let seg = RayMajorantSegment {
            t_min: self.t_min,
            t_max: t_voxel_exit,
            sigma_maj: self.sigma_t * density_maj,
        };

        self.t_min = t_voxel_exit;
        if self.next_crossing_t[step_axis] > self.t_max {
            self.t_min = self.t_max;
        }
        self.voxel[step_axis] += self.step[step_axis];
        if self.voxel[step_axis] == self.voxel_limit[step_axis] {
            self.t_min = self.t_max;
        }
        self.next_crossing_t[step_axis] += self.delta_t[step_axis];

        Some(seg)
    }
}
