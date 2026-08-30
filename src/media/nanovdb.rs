//! NanoVDB-backed medium. Mirrors pbrt-v4's `NanoVDBMedium` naming and
//! keeps the `NvdbFile` alive so density lookups can walk the NanoVDB
//! tree directly instead of densifying the active bbox into a flat
//! `Vec<Float>`.
//!
//! Memory footprint of bunny-cloud (`bunny_cloud.nvdb`, ~19M active
//! voxels in a 576×571×437 bbox) drops from ~551 MB densified to roughly
//! the size of the decompressed NanoVDB grid (~80 MB).
//!
//! The implementation follows v4 by retaining the NanoVDB tree and
//! sampling it on demand.

use super::majorant_iterator::DDAMajorantIterator;
use super::majorant_iterator::MajorantGrid;
use super::medium::grid_resolution;
use super::medium::load_ascii_density_grid;
use super::phase_function::HGPhaseFunction;
use super::phase_function::PhaseFunction;
use crate::base::medium::{MediumCoefficients, MediumProperties, MediumSigma};
use crate::paramdict::ParameterDictionary;
use crate::util::base::*;
use crate::util::error::PbrtError;
use crate::util::geometry::*;
use crate::util::rng::RNG;
use crate::util::spectrum::*;
use crate::util::transform::*;

use rayon::prelude::*;
use std::sync::Arc;

use nanovdb_rs::{
    create_sampler1, Grid, GridType, NvdbFile, ReadAccessor, TreeData, ValidatedFloatTreeCache,
};

/// Select a named `Float` grid from an already-open `.nvdb` file for
/// `NanoVDBMedium`, keeping the sparse NanoVDB tree alive.
fn find_grid(file: &Arc<NvdbFile>, filename: &str, grid_name: &str) -> Option<NanoVDBBuffer> {
    let grid_idx = file
        .grids()
        .iter()
        .position(|g| g.name() == grid_name && g.value_type() == GridType::Float)?;
    let grid = &file.grids()[grid_idx];
    let metadata = grid.grid_metadata();
    if !metadata.is_fog_volume() && !metadata.is_unknown_class() {
        log::warn!(
            "nanovdb: {} \"{}\" isn't a FogVolume grid?",
            filename,
            grid_name
        );
        return None;
    }
    let (bmin, bmax) = grid.index_bbox();
    if bmin[0] > bmax[0] || bmin[1] > bmax[1] || bmin[2] > bmax[2] {
        log::warn!("nanovdb: {} has empty index bbox", filename);
        return None;
    }
    let nx = (bmax[0] - bmin[0] + 1).max(1) as u32;
    let ny = (bmax[1] - bmin[1] + 1).max(1) as u32;
    let nz = (bmax[2] - bmin[2] + 1).max(1) as u32;

    log::info!(
        "nanovdb: {} \"{}\" -> {}x{}x{} (active bbox), retained tree only",
        filename,
        grid.name(),
        nx,
        ny,
        nz,
    );
    Some(NanoVDBBuffer::new(Arc::clone(file), grid_idx))
}

/// Open a `.nvdb` file once and select its density and optional temperature
/// grids for `NanoVDBMedium`.
fn read_grids(
    filename: &str,
    density_name: &str,
    temperature_name: &str,
) -> Option<(NanoVDBBuffer, Option<NanoVDBBuffer>)> {
    if filename.is_empty() {
        return None;
    }
    let file = match NvdbFile::open(filename) {
        Ok(file) => Arc::new(file),
        Err(e) => {
            log::warn!("nanovdb: open {} failed: {}", filename, e);
            return None;
        }
    };
    let density_grid = find_grid(&file, filename, density_name)?;
    let temperature_grid = if temperature_name.is_empty() {
        None
    } else {
        find_grid(&file, filename, temperature_name)
    };
    Some((density_grid, temperature_grid))
}

/// Backing storage for a NanoVDB grid. This mirrors pbrt-v4's
/// `NanoVDBBuffer` role: keep the decoded/mapped grid bytes alive for
/// `NanoVDBMedium`. In r4 the actual file and mmap ownership lives in
/// `nanovdb-rs::NvdbFile`, so the buffer records the selected grid.
#[derive(Clone)]
pub struct NanoVDBBuffer {
    file: Arc<NvdbFile>,
    grid_idx: usize,
}

impl NanoVDBBuffer {
    pub fn new(file: Arc<NvdbFile>, grid_idx: usize) -> Self {
        Self { file, grid_idx }
    }

    fn grid(&self) -> &Grid {
        &self.file.grids()[self.grid_idx]
    }

    pub fn raw_bytes(&self) -> &[u8] {
        self.grid().raw_bytes()
    }

    pub fn bytes_allocated(&self) -> usize {
        self.raw_bytes().len()
    }

    fn index_bbox(&self) -> ([i32; 3], [i32; 3]) {
        self.grid().index_bbox()
    }

    fn world_bounds(&self) -> Bounds3f {
        let (world_min, world_max) = self.grid().world_bbox();
        Bounds3f::new(
            &Point3f::new(
                world_min.x as Float,
                world_min.y as Float,
                world_min.z as Float,
            ),
            &Point3f::new(
                world_max.x as Float,
                world_max.y as Float,
                world_max.z as Float,
            ),
        )
    }

    fn world_to_index(&self) -> Option<Transform> {
        let header = self.grid().header()?;
        let map = &header.map;
        let index_to_world = Matrix4x4::new(
            map.mat_d[0][0] as Float,
            map.mat_d[0][1] as Float,
            map.mat_d[0][2] as Float,
            map.vec_d[0] as Float,
            map.mat_d[1][0] as Float,
            map.mat_d[1][1] as Float,
            map.mat_d[1][2] as Float,
            map.vec_d[1] as Float,
            map.mat_d[2][0] as Float,
            map.mat_d[2][1] as Float,
            map.mat_d[2][2] as Float,
            map.vec_d[2] as Float,
            0.0,
            0.0,
            0.0,
            1.0,
        );

        let tx = -(map.inv_mat_d[0][0] * map.vec_d[0]
            + map.inv_mat_d[0][1] * map.vec_d[1]
            + map.inv_mat_d[0][2] * map.vec_d[2]);
        let ty = -(map.inv_mat_d[1][0] * map.vec_d[0]
            + map.inv_mat_d[1][1] * map.vec_d[1]
            + map.inv_mat_d[1][2] * map.vec_d[2]);
        let tz = -(map.inv_mat_d[2][0] * map.vec_d[0]
            + map.inv_mat_d[2][1] * map.vec_d[1]
            + map.inv_mat_d[2][2] * map.vec_d[2]);
        let world_to_index = Matrix4x4::new(
            map.inv_mat_d[0][0] as Float,
            map.inv_mat_d[0][1] as Float,
            map.inv_mat_d[0][2] as Float,
            tx as Float,
            map.inv_mat_d[1][0] as Float,
            map.inv_mat_d[1][1] as Float,
            map.inv_mat_d[1][2] as Float,
            ty as Float,
            map.inv_mat_d[2][0] as Float,
            map.inv_mat_d[2][1] as Float,
            map.inv_mat_d[2][2] as Float,
            tz as Float,
            0.0,
            0.0,
            0.0,
            1.0,
        );
        Some(Transform::from((world_to_index, index_to_world)))
    }

    fn tree_data_and_background(&self) -> Option<(TreeData, f32)> {
        ReadAccessor::parse_tree_data(self.raw_bytes())
    }
}

impl std::fmt::Debug for NanoVDBBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NanoVDBBuffer")
            .field("grid_idx", &self.grid_idx)
            .field("bytes_allocated", &self.bytes_allocated())
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
struct NanoVDBFloatGrid {
    world_to_index: Transform,
    tree_data: TreeData,
    background: f32,
    validated_tree: Option<ValidatedFloatTreeCache>,
}

impl NanoVDBFloatGrid {
    fn new(grid: &NanoVDBBuffer) -> Option<Self> {
        let (tree_data, background) = grid.tree_data_and_background()?;
        let world_to_index = grid.world_to_index()?;
        Some(Self {
            world_to_index,
            tree_data,
            background,
            validated_tree: ValidatedFloatTreeCache::new(grid.raw_bytes()),
        })
    }

    fn sample(&self, grid: &NanoVDBBuffer, p: &Point3f) -> Float {
        let mut accessor =
            ReadAccessor::with_tree_data(grid.raw_bytes(), self.tree_data, self.background);
        self.sample_with_accessor(&mut accessor, grid.raw_bytes(), p)
    }

    fn sample_with_accessor(
        &self,
        accessor: &mut ReadAccessor<'_>,
        bytes: &[u8],
        p: &Point3f,
    ) -> Float {
        let p_index = self.world_to_index.transform_point(p);
        let p_index = [p_index.x, p_index.y, p_index.z];
        if let Some(validated_tree) = self.validated_tree {
            if let Some(value) = validated_tree.sample(
                bytes,
                [p_index[0] as f32, p_index[1] as f32, p_index[2] as f32],
            ) {
                return NanoVDBMedium::sanitize_density_value(value as Float);
            }
        }
        NanoVDBMedium::sanitize_density_value(create_sampler1(accessor).sample_f32([
            p_index[0] as f32,
            p_index[1] as f32,
            p_index[2] as f32,
        ]) as Float)
    }
}

/// NanoVDB-backed medium. This is r4's counterpart to pbrt-v4
/// `NanoVDBMedium`.
pub struct NanoVDBMedium {
    bounds: Bounds3f,
    render_from_medium: Transform,
    sigma_a_spec: DenselySampledSpectrum,
    sigma_s_spec: DenselySampledSpectrum,
    phase: Arc<PhaseFunction>,
    majorant_grid: MajorantGrid,
    density_grid: NanoVDBBuffer,
    temperature_grid: Option<NanoVDBBuffer>,
    density_float_grid: NanoVDBFloatGrid,
    temperature_float_grid: Option<NanoVDBFloatGrid>,
    le_scale: Float,
    temperature_offset: Float,
    temperature_scale: Float,
}

impl std::fmt::Debug for NanoVDBMedium {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NanoVDBMedium")
            .field("bounds", &self.bounds)
            .field("le_scale", &self.le_scale)
            .field("temperature_offset", &self.temperature_offset)
            .field("temperature_scale", &self.temperature_scale)
            .field("sigma_a_spec", &self.sigma_a_spec)
            .field("sigma_s_spec", &self.sigma_s_spec)
            .finish()
    }
}

impl NanoVDBMedium {
    pub fn create(
        parameters: &ParameterDictionary,
        render_from_medium: &Transform,
    ) -> Result<Self, PbrtError> {
        let (nx, ny, nz) = grid_resolution(parameters);
        let expected = (nx * ny * nz) as usize;
        let filename = parameters.get_one_filename("filename", "");
        let grid_name = parameters.get_one_string("gridname", "density");
        let sigma_a =
            parameters.get_one_spectrum_typed("sigma_a", &Spectrum::one(), SpectrumType::Unbounded);
        let sigma_s =
            parameters.get_one_spectrum_typed("sigma_s", &Spectrum::one(), SpectrumType::Unbounded);
        let sigma_scale = parameters.get_one_float("scale", 1.0);
        let g = parameters.get_one_float("g", 0.0);

        if let Some(data) = parameters.get_floats_ref(&grid_name) {
            if expected > 0 && data.len() == expected {
                return Err(PbrtError::error(
                    "NanoVDBMedium received inline density data; use grid/rgbgrid for inline grids.",
                ));
            }
        }
        if load_ascii_density_grid(&filename, expected).is_some() {
            return Err(PbrtError::error(
                "NanoVDBMedium received ASCII density data; use grid/rgbgrid for dense grids.",
            ));
        }

        // Real .nvdb file: by default we keep the NanoVDB tree alive
        // and use `NanoVDBMedium` for on-demand voxel lookups. The
        // because it would create a GridMedium instead.
        let dense_storage = parameters.get_one_bool("dense", false);
        if dense_storage {
            return Err(PbrtError::error(
                "NanoVDBMedium requested dense NanoVDB storage, which cannot create NanoVDBMedium.",
            ));
        }
        let temperature_name = parameters.get_one_string("temperaturename", "temperature");
        let le_scale = parameters.get_one_float("Lescale", 1.0);
        let temperature_offset = parameters.get_one_float(
            "temperatureoffset",
            parameters.get_one_float("temperaturecutoff", 0.0),
        );
        let temperature_scale = parameters.get_one_float("temperaturescale", 1.0);
        if !filename.is_empty() {
            if !dense_storage {
                if let Some((density_grid, temperature_grid)) =
                    read_grids(&filename, &grid_name, &temperature_name)
                {
                    return Self::new(
                        render_from_medium,
                        &sigma_a,
                        &sigma_s,
                        sigma_scale,
                        g,
                        density_grid,
                        temperature_grid,
                        le_scale,
                        temperature_offset,
                        temperature_scale,
                    )
                    .ok_or_else(|| {
                        PbrtError::error(&format!(
                            "NanoVDBMedium failed to parse NanoVDB file {}.",
                            filename
                        ))
                    });
                }
            }
            return Err(PbrtError::error(&format!(
                "NanoVDBMedium failed to load NanoVDB file {}.",
                filename
            )));
        }
        Err(PbrtError::error(
            "NanoVDBMedium requested without filename.",
        ))
    }

    pub fn new(
        render_from_medium: &Transform,
        sigma_a: &Spectrum,
        sigma_s: &Spectrum,
        sigma_scale: Float,
        g: Float,
        density_grid: NanoVDBBuffer,
        temperature_grid: Option<NanoVDBBuffer>,
        le_scale: Float,
        temperature_offset: Float,
        temperature_scale: Float,
    ) -> Option<Self> {
        let mut sigma_a_spec = DenselySampledSpectrum::from_spectrum(sigma_a);
        sigma_a_spec.scale(sigma_scale);
        let mut sigma_s_spec = DenselySampledSpectrum::from_spectrum(sigma_s);
        sigma_s_spec.scale(sigma_scale);

        let density_float_grid = NanoVDBFloatGrid::new(&density_grid)?;
        let mut bounds = density_grid.world_bounds();
        let temperature_float_grid = match temperature_grid.as_ref() {
            None => None,
            Some(grid) => {
                bounds = bounds.union(&grid.world_bounds());
                Some(NanoVDBFloatGrid::new(grid)?)
            }
        };

        let majorant_grid = Self::build_majorant_grid(&density_grid, density_float_grid, bounds);
        Some(NanoVDBMedium {
            bounds,
            render_from_medium: *render_from_medium,
            sigma_a_spec,
            sigma_s_spec,
            phase: Arc::new(PhaseFunction::from(HGPhaseFunction::new(g))),
            majorant_grid,
            density_grid,
            temperature_grid,
            density_float_grid,
            temperature_float_grid,
            le_scale,
            temperature_offset,
            temperature_scale,
        })
    }

    fn build_majorant_grid(
        density_grid: &NanoVDBBuffer,
        density_float_grid: NanoVDBFloatGrid,
        bounds: Bounds3f,
    ) -> MajorantGrid {
        let res: [i32; 3] = [64, 64, 64];
        let mut values = vec![0.0; (res[0] * res[1] * res[2]) as usize];
        let bytes = density_grid.raw_bytes();
        let (bmin, bmax) = density_grid.index_bbox();
        values.par_iter_mut().enumerate().for_each(|(index, out)| {
            let x = index as i32 % res[0];
            let y = (index as i32 / res[0]) % res[1];
            let z = index as i32 / (res[0] * res[1]);

            let p0 = bounds.lerp(&Vector3f::new(
                x as Float / res[0] as Float,
                y as Float / res[1] as Float,
                z as Float / res[2] as Float,
            ));
            let p1 = bounds.lerp(&Vector3f::new(
                (x + 1) as Float / res[0] as Float,
                (y + 1) as Float / res[1] as Float,
                (z + 1) as Float / res[2] as Float,
            ));
            let i0 = density_float_grid.world_to_index.transform_point(&p0);
            let i1 = density_float_grid.world_to_index.transform_point(&p1);
            let bounds_for_axis = |axis: usize| {
                let lo = Float::min(i0[axis], i1[axis]) - 1.0;
                let hi = Float::max(i0[axis], i1[axis]) + 1.0;
                (
                    i32::clamp(lo as i32, bmin[axis], bmax[axis]),
                    i32::clamp(hi as i32, bmin[axis], bmax[axis]),
                )
            };
            let (x0, x1) = bounds_for_axis(0);
            let (y0, y1) = bounds_for_axis(1);
            let (z0, z1) = bounds_for_axis(2);
            let mut accessor = ReadAccessor::with_tree_data(
                bytes,
                density_float_grid.tree_data,
                density_float_grid.background,
            );
            let mut max_value = 0.0;
            for iz in z0..=z1 {
                for iy in y0..=y1 {
                    for ix in x0..=x1 {
                        let v = accessor.get_value([ix, iy, iz]) as Float;
                        if v.is_finite() && v > max_value {
                            max_value = v;
                        }
                    }
                }
            }
            *out = max_value;
        });
        MajorantGrid::new(bounds, res, Arc::from(values))
    }

    fn render_to_medium(&self) -> Transform {
        Transform::inverse(&self.render_from_medium)
    }

    fn apply_inverse_ray(&self, ray: &Ray, ray_t_max: Float) -> (Ray, Float) {
        let (ray_medium, o_error, _) = self.render_to_medium().transform_ray(ray);
        let mut ray_t_max = ray_t_max;
        let length_squared = ray_medium.d.length_squared();
        if length_squared > 0.0 {
            let dt = ray_medium.d.abs().dot(&o_error) / length_squared;
            ray_t_max -= dt;
        }
        (ray_medium, ray_t_max)
    }

    fn sanitize_density_value(v: Float) -> Float {
        if v.is_finite() && v >= 0.0 {
            v
        } else {
            0.0
        }
    }

    /// Trilinear density at a medium-space point. This mirrors v4's
    /// `FloatGrid::worldToIndexF` followed by `SampleFromVoxels`.
    fn density(&self, p: &Point3f) -> Float {
        self.density_float_grid.sample(&self.density_grid, p)
    }

    fn density_with_accessor(&self, accessor: &mut ReadAccessor<'_>, p: &Point3f) -> Float {
        self.density_float_grid
            .sample_with_accessor(accessor, self.density_grid.raw_bytes(), p)
    }

    #[allow(dead_code)]
    pub fn le(&self, p: &Point3f, lambda: &SampledWavelengths) -> SampledSpectrum {
        let p_index = self.render_to_medium().transform_point(p);
        self.le_medium_space(&p_index, lambda)
    }

    fn le_medium_space(&self, p: &Point3f, lambda: &SampledWavelengths) -> SampledSpectrum {
        let Some(grid) = &self.temperature_grid else {
            return SampledSpectrum::zero();
        };
        if self.le_scale <= 0.0 {
            return SampledSpectrum::zero();
        }
        let Some(float_grid) = &self.temperature_float_grid else {
            return SampledSpectrum::zero();
        };
        let temp = float_grid.sample(grid, p);
        let temp = (temp - self.temperature_offset) * self.temperature_scale;
        if temp <= 100.0 {
            return SampledSpectrum::zero();
        }
        Spectrum::Blackbody(BlackbodySpectrum::new(temp, self.le_scale)).sample(lambda)
    }

    #[inline(always)]
    fn sample_t_maj_with_accessor<T, S, F>(
        &self,
        ray: &Ray,
        t_max: Float,
        mut u: Float,
        lambda: &SampledWavelengths,
        rng: &mut RNG,
        mut sample_point: S,
        mut callback: F,
    ) -> SampledSpectrum
    where
        S: FnMut(&Self, &mut ReadAccessor<'_>, &Point3f, &SampledWavelengths) -> T,
        F: FnMut(Point3f, T, SampledSpectrum, SampledSpectrum, &mut RNG) -> bool,
    {
        let ray_length = ray.d.length();
        if ray_length == 0.0 {
            return SampledSpectrum::one();
        }

        let mut local_ray = ray.clone();
        local_ray.d = local_ray.d / ray_length;
        let Some(mut iter) = self.sample_ray(&local_ray, t_max * ray_length, lambda) else {
            return SampledSpectrum::one();
        };
        let bytes = self.density_grid.raw_bytes();
        let mut accessor = ReadAccessor::with_tree_data(
            bytes,
            self.density_float_grid.tree_data,
            self.density_float_grid.background,
        );
        let mut t_maj = SampledSpectrum::one();
        let mut done = false;

        while !done {
            let Some(seg) = iter.next() else {
                return t_maj;
            };

            if seg.sigma_maj[0] == 0.0 {
                let mut dt = seg.t_max - seg.t_min;
                if dt.is_infinite() {
                    dt = Float::MAX;
                }
                t_maj *= (-(seg.sigma_maj) * dt).exp();
                continue;
            }

            let mut t_min = seg.t_min;
            loop {
                let t = t_min - Float::ln(1.0 - u) / seg.sigma_maj[0];
                u = rng.uniform_float();
                if t < seg.t_max {
                    t_maj *= (-(t - t_min) * seg.sigma_maj).exp();
                    let p = local_ray.position(t);
                    let sampled = sample_point(self, &mut accessor, &p, lambda);
                    if !callback(p, sampled, seg.sigma_maj, t_maj, rng) {
                        done = true;
                        break;
                    }
                    t_maj = SampledSpectrum::one();
                    t_min = t;
                } else {
                    let mut dt = seg.t_max - t_min;
                    if dt.is_infinite() {
                        dt = Float::MAX;
                    }
                    t_maj *= (-(seg.sigma_maj) * dt).exp();
                    break;
                }
            }
        }

        SampledSpectrum::one()
    }

    pub fn sample_t_maj_coefficients<F>(
        &self,
        ray: &Ray,
        t_max: Float,
        u: Float,
        lambda: &SampledWavelengths,
        rng: &mut RNG,
        callback: F,
    ) -> SampledSpectrum
    where
        F: FnMut(Point3f, MediumCoefficients, SampledSpectrum, SampledSpectrum, &mut RNG) -> bool,
    {
        self.sample_t_maj_with_accessor(
            ray,
            t_max,
            u,
            lambda,
            rng,
            |medium, accessor, p, lambda| {
                let p_index = medium.render_to_medium().transform_point(p);
                let density = medium.density_with_accessor(accessor, &p_index);
                MediumCoefficients::new(
                    medium.sigma_a_spec.sample(lambda) * density,
                    medium.sigma_s_spec.sample(lambda) * density,
                    medium.le_medium_space(&p_index, lambda),
                )
            },
            callback,
        )
    }

    pub fn sample_t_maj_sigma<F>(
        &self,
        ray: &Ray,
        t_max: Float,
        u: Float,
        lambda: &SampledWavelengths,
        rng: &mut RNG,
        callback: F,
    ) -> SampledSpectrum
    where
        F: FnMut(Point3f, MediumSigma, SampledSpectrum, SampledSpectrum, &mut RNG) -> bool,
    {
        self.sample_t_maj_with_accessor(
            ray,
            t_max,
            u,
            lambda,
            rng,
            |medium, accessor, p, lambda| {
                let p_index = medium.render_to_medium().transform_point(p);
                let density = medium.density_with_accessor(accessor, &p_index);
                MediumSigma::new(
                    medium.sigma_a_spec.sample(lambda) * density,
                    medium.sigma_s_spec.sample(lambda) * density,
                )
            },
            callback,
        )
    }
}

impl NanoVDBMedium {
    pub fn is_emissive(&self) -> bool {
        self.temperature_float_grid.is_some() && self.le_scale > 0.0
    }

    pub fn sample_point(&self, p: &Point3f, lambda: &SampledWavelengths) -> MediumProperties {
        let p_index = self.render_to_medium().transform_point(p);
        let density = self.density(&p_index);
        MediumProperties::new(
            self.sigma_a_spec.sample(lambda) * density,
            self.sigma_s_spec.sample(lambda) * density,
            Arc::clone(&self.phase),
            self.le_medium_space(&p_index, lambda),
        )
    }

    pub fn sample_point_coefficients(
        &self,
        p: &Point3f,
        lambda: &SampledWavelengths,
    ) -> MediumCoefficients {
        let p_index = self.render_to_medium().transform_point(p);
        let density = self.density(&p_index);
        MediumCoefficients::new(
            self.sigma_a_spec.sample(lambda) * density,
            self.sigma_s_spec.sample(lambda) * density,
            self.le_medium_space(&p_index, lambda),
        )
    }

    pub fn sample_point_sigma(&self, p: &Point3f, lambda: &SampledWavelengths) -> MediumSigma {
        let p_index = self.render_to_medium().transform_point(p);
        let density = self.density(&p_index);
        MediumSigma::new(
            self.sigma_a_spec.sample(lambda) * density,
            self.sigma_s_spec.sample(lambda) * density,
        )
    }

    pub fn sample_ray(
        &self,
        ray: &Ray,
        t_max: Float,
        lambda: &SampledWavelengths,
    ) -> Option<DDAMajorantIterator> {
        let (ray_medium, ray_t_max) = self.apply_inverse_ray(ray, t_max);
        let Some((t_min, t_max)) = self.bounds.intersect_p(&ray_medium, ray_t_max) else {
            return None;
        };
        debug_assert!(t_max <= ray_t_max);
        let sigma_t = self.sigma_a_spec.sample(lambda) + self.sigma_s_spec.sample(lambda);
        Some(DDAMajorantIterator::new(
            ray_medium,
            t_min,
            t_max,
            sigma_t,
            self.majorant_grid.clone(),
        ))
    }
}
