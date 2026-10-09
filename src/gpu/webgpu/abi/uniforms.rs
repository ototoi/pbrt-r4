use bytemuck::{Pod, Zeroable};

use crate::gpu::flat;
use crate::util::error::PbrtError;

use super::constants::{INVALID_INDEX, LIGHT_SAMPLER_KIND_UNIFORM};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub camera_to_world: [[f32; 4]; 4],
    pub raster_to_camera: [[f32; 4]; 4],
    pub world_to_camera: [[f32; 4]; 4],
    pub min_dir_differential_x: [f32; 4],
    pub min_dir_differential_y: [f32; 4],
    pub medium_id: u32,
    pub disable_texture_filtering: u32,
    pub disable_pixel_jitter: u32,
    pub padding: u32,
    pub lens_radius: f32,
    pub focal_distance: f32,
    pub lens_padding: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ViewportUniform {
    /// Full output image resolution. Fixed for the whole render; the camera
    /// raster transform and sampler pixel seeding are keyed to this, not to
    /// `tile_width`/`tile_height`.
    pub full_width: u32,
    pub full_height: u32,
    /// Offset and extent of the rendered region (cropwindow/pixelbounds)
    /// within the full image; matches the `Film` framebuffer's actual
    /// dimensions. Equal to the full image when rendering the whole image.
    pub region_x: u32,
    pub region_y: u32,
    pub region_width: u32,
    pub region_height: u32,
    /// Offset and extent of the tile currently being dispatched, in full
    /// image coordinates. Wavefront queues are sized and indexed by
    /// `tile_width * tile_height`, not by the full image or region size.
    pub tile_x: u32,
    pub tile_y: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub sample_index: u32,
    pub max_depth: u32,
    pub seed: u32,
    pub disable_wavelength_jitter: u32,
    pub medium_scattering_enabled: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MaterialTableUniform {
    pub material_offset_words: u32,
    pub material_node_count: u32,
    pub debug_material_kind: u32,
    pub attributes_eval_stride: u32,
    pub texture_eval_stride: u32,
    pub measured_texture_base: u32,
    pub measured_texture_width: u32,
    pub measured_texture_height: u32,
    pub measured_texture_count: u32,
    pub have_subsurface: u32,
    pub reserved: [u32; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightTableUniform {
    pub light_record_offset_words: u32,
    pub light_count: u32,
    pub light_sampler_kind: u32,
    pub light_sampler_data_offset: u32,
    pub light_bvh_node_offset: u32,
    pub light_bvh_node_count: u32,
    pub light_leaf_offset: u32,
    pub light_leaf_count: u32,
    pub finite_light_count: u32,
    pub infinite_light_count: u32,
    pub reserved: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FilmUniform {
    pub sensor_response: [u32; 4],
    pub imaging_ratio: f32,
    pub max_sample_luminance: f32,
    pub mode: u32,
    pub padding: u32,
}

pub fn film_uniform(film: &flat::Film) -> FilmUniform {
    FilmUniform {
        sensor_response: [
            film.sensor_response[0],
            film.sensor_response[1],
            film.sensor_response[2],
            0,
        ],
        imaging_ratio: film.imaging_ratio,
        max_sample_luminance: film.max_sample_luminance,
        mode: 0,
        padding: 0,
    }
}

pub fn viewport_uniform(
    viewport: &flat::Viewport,
    settings: &flat::RenderSettings,
) -> Result<ViewportUniform, PbrtError> {
    let [width, height] = viewport.resolution;
    if width == 0 || height == 0 {
        return Err(PbrtError::error(
            "WebGPU viewport resolution must be positive.",
        ));
    }
    if u64::from(width) * u64::from(height) > u64::from(u32::MAX) {
        return Err(PbrtError::error(
            "WebGPU viewport pixel count must fit in u32.",
        ));
    }
    let [region_x, region_y] = viewport.region_offset;
    let [region_width, region_height] = viewport.region_resolution;
    if region_width == 0
        || region_height == 0
        || region_x
            .checked_add(region_width)
            .is_none_or(|edge| edge > width)
        || region_y
            .checked_add(region_height)
            .is_none_or(|edge| edge > height)
    {
        return Err(PbrtError::error(
            "WebGPU rendered region must be positive and fit within the full image.",
        ));
    }
    Ok(ViewportUniform {
        full_width: width,
        full_height: height,
        region_x,
        region_y,
        region_width,
        region_height,
        tile_x: region_x,
        tile_y: region_y,
        tile_width: region_width,
        tile_height: region_height,
        sample_index: 0,
        max_depth: settings.max_depth,
        seed: settings.seed,
        disable_wavelength_jitter: u32::from(settings.disable_wavelength_jitter),
        medium_scattering_enabled: 0,
    })
}

pub fn material_table_uniform(
    material_node_count: usize,
) -> Result<MaterialTableUniform, PbrtError> {
    let to_u32 = |value: usize, label: &str| {
        u32::try_from(value)
            .map_err(|_| PbrtError::error(&format!("WebGPU {label} does not fit in u32.")))
    };
    Ok(MaterialTableUniform {
        material_offset_words: 0,
        material_node_count: to_u32(material_node_count, "material node count")?,
        debug_material_kind: INVALID_INDEX,
        attributes_eval_stride: 0,
        texture_eval_stride: 0,
        measured_texture_base: 0,
        measured_texture_width: 0,
        measured_texture_height: 0,
        measured_texture_count: 0,
        have_subsurface: 0,
        reserved: [0; 8],
    })
}

pub fn light_table_uniform(
    finite_light_count: usize,
    infinite_light_count: usize,
    light_record_offset_words: usize,
) -> Result<LightTableUniform, PbrtError> {
    let to_u32 = |value: usize, label: &str| {
        u32::try_from(value)
            .map_err(|_| PbrtError::error(&format!("WebGPU {label} does not fit in u32.")))
    };
    Ok(LightTableUniform {
        light_record_offset_words: to_u32(light_record_offset_words, "light-record offset")?,
        light_count: to_u32(
            finite_light_count
                .checked_add(infinite_light_count)
                .ok_or_else(|| PbrtError::error("WebGPU light count overflow."))?,
            "light count",
        )?,
        light_sampler_kind: LIGHT_SAMPLER_KIND_UNIFORM,
        light_sampler_data_offset: INVALID_INDEX,
        light_bvh_node_offset: INVALID_INDEX,
        light_bvh_node_count: 0,
        light_leaf_offset: INVALID_INDEX,
        light_leaf_count: 0,
        finite_light_count: to_u32(finite_light_count, "finite light count")?,
        infinite_light_count: to_u32(infinite_light_count, "infinite light count")?,
        reserved: [0; 2],
    })
}
