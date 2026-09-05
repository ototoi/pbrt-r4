use bytemuck::{Pod, Zeroable};

use crate::gpu::ir::flat;
use crate::util::error::PbrtError;

pub const WORKGROUP_SIZE: u32 = 8;
pub const RAY_T_MIN: f32 = 0.0;
pub const RAY_T_MAX: f32 = f32::MAX;
pub const LIGHT_KIND_POINT: u32 = 0;
pub const LIGHT_KIND_AREA: u32 = 1;
pub const LIGHT_SAMPLER_KIND_UNIFORM: u32 = 0;
pub const LIGHT_SAMPLER_KIND_BVH: u32 = 1;
pub const INVALID_INDEX: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub camera_to_world: [[f32; 4]; 4],
    pub raster_to_camera: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ViewportUniform {
    pub width: u32,
    pub height: u32,
    pub sample_index: u32,
    pub max_depth: u32,
    pub seed: u32,
    pub padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SceneUniform {
    pub material_offset_words: u32,
    pub material_count: u32,
    pub light_record_offset_words: u32,
    pub light_count: u32,
    pub point_light_offset_words: u32,
    pub point_light_count: u32,
    pub area_light_offset_words: u32,
    pub area_light_count: u32,
    pub light_sampler_kind: u32,
    pub light_sampler_data_offset: u32,
    pub light_bvh_node_offset: u32,
    pub light_bvh_node_count: u32,
    pub light_leaf_offset: u32,
    pub light_leaf_count: u32,
    pub scene_data_words: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 4],
    pub normal: [f32; 4],
    pub tangent: [f32; 4],
    pub uv: [f32; 2],
    pub padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Geometry {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Instance {
    pub geometry: u32,
    pub material: u32,
    pub area_light: u32,
    pub orientation_flags: u32,
    pub world_from_object: [[f32; 4]; 4],
    pub normal_from_object: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Material {
    pub kind_tag: u32,
    pub padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RayWorkItem {
    pub origin: [f32; 4],
    pub direction: [f32; 4],
    pub throughput: [f32; 4],
    pub prev_position: [f32; 4],
    pub prev_position_error: [f32; 4],
    pub prev_geometric_normal: [f32; 4],
    pub prev_shading_normal: [f32; 4],
    pub pixel_index: u32,
    pub depth: u32,
    pub inv_w_u: f32,
    pub inv_w_l: f32,
    pub prev_pdf: f32,
    pub padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ShadowRayWorkItem {
    pub origin: [f32; 4],
    pub direction: [f32; 4],
    pub max_t: f32,
    pub padding: [u32; 3],
    pub direct: [f32; 4],
    pub pixel_index: u32,
    pub reserved: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SurfaceWorkItem {
    pub t: f32,
    pub hit: u32,
    pub instance_custom_data: u32,
    pub primitive_index: u32,
    pub barycentric: [f32; 4],
    pub position: [f32; 4],
    pub position_error: [f32; 4],
    pub normal: [f32; 4],
    pub geometric_normal: [f32; 4],
    pub material: u32,
    pub flags: u32,
    pub padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PointLight {
    pub position: [f32; 4],
    pub intensity: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightRecord {
    pub kind: u32,
    pub payload: u32,
    pub padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AreaLight {
    pub instance: u32,
    pub distribution_offset_words: u32,
    pub distribution_count: u32,
    pub total_area: f32,
    pub emission: [f32; 3],
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TriangleDistributionEntry {
    pub primitive: u32,
    pub cdf: f32,
    pub area: f32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct QueueState {
    pub count: u32,
    pub capacity: u32,
    pub overflow: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PixelSampleState {
    pub radiance: [f32; 4],
    pub pixel_index: u32,
    pub sample_index: u32,
    pub error: u32,
    pub padding: u32,
}

pub fn camera_uniform(
    camera: &flat::Camera,
    viewport: &flat::Viewport,
) -> Result<CameraUniform, PbrtError> {
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
    if !camera.fov.is_finite() || camera.fov <= 0.0 || camera.fov >= 180.0 {
        return Err(PbrtError::error(
            "WebGPU camera fov must be finite and in (0, 180).",
        ));
    }
    let [xmin, xmax, ymin, ymax] = camera.screen_window;
    if ![xmin, xmax, ymin, ymax]
        .iter()
        .all(|value| value.is_finite())
        || xmin >= xmax
        || ymin >= ymax
    {
        return Err(PbrtError::error("WebGPU camera screen window is invalid."));
    }
    let camera_to_world = row_major_to_columns(camera.camera_to_world);
    if !camera_to_world
        .iter()
        .flatten()
        .all(|value| value.is_finite())
    {
        return Err(PbrtError::error(
            "WebGPU camera transform contains a non-finite value.",
        ));
    }
    validate_affine(camera.camera_to_world, "Camera")?;

    let tan_half_fov = (camera.fov.to_radians() * 0.5).tan();
    if !tan_half_fov.is_finite() {
        return Err(PbrtError::error(
            "WebGPU camera fov produced a non-finite tangent.",
        ));
    }
    // `screen_window` already incorporates the frame aspect ratio. Do not
    // apply the viewport aspect ratio a second time here.
    let dx = (xmax - xmin) / width as f32;
    let dy = (ymax - ymin) / height as f32;
    let raster_to_camera = row_major_to_columns([
        dx * tan_half_fov,
        0.0,
        0.0,
        (xmin + 0.5 * dx) * tan_half_fov,
        0.0,
        -dy * tan_half_fov,
        0.0,
        (ymax - 0.5 * dy) * tan_half_fov,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]);
    Ok(CameraUniform {
        camera_to_world,
        raster_to_camera,
    })
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
    Ok(ViewportUniform {
        width,
        height,
        sample_index: 0,
        max_depth: settings.max_depth,
        seed: settings.seed,
        padding: [0; 3],
    })
}

pub fn scene_uniform(
    material_count: usize,
    light_count: usize,
    point_light_count: usize,
    area_light_count: usize,
    light_record_offset_words: usize,
    point_light_offset_words: usize,
    area_light_offset_words: usize,
    scene_data_words: usize,
) -> Result<SceneUniform, PbrtError> {
    let to_u32 = |value: usize, label: &str| {
        u32::try_from(value)
            .map_err(|_| PbrtError::error(&format!("WebGPU {label} does not fit in u32.")))
    };
    Ok(SceneUniform {
        material_offset_words: 0,
        material_count: to_u32(material_count, "material count")?,
        light_record_offset_words: to_u32(light_record_offset_words, "light-record offset")?,
        light_count: to_u32(light_count, "light count")?,
        point_light_offset_words: to_u32(point_light_offset_words, "point-light offset")?,
        point_light_count: to_u32(point_light_count, "point-light count")?,
        area_light_offset_words: to_u32(area_light_offset_words, "area-light offset")?,
        area_light_count: to_u32(area_light_count, "area-light count")?,
        light_sampler_kind: LIGHT_SAMPLER_KIND_UNIFORM,
        light_sampler_data_offset: INVALID_INDEX,
        light_bvh_node_offset: INVALID_INDEX,
        light_bvh_node_count: 0,
        light_leaf_offset: INVALID_INDEX,
        light_leaf_count: 0,
        scene_data_words: to_u32(scene_data_words, "scene-data word count")?,
        reserved: 0,
    })
}

pub fn row_major_to_columns(matrix: [f32; 16]) -> [[f32; 4]; 4] {
    [
        [matrix[0], matrix[4], matrix[8], matrix[12]],
        [matrix[1], matrix[5], matrix[9], matrix[13]],
        [matrix[2], matrix[6], matrix[10], matrix[14]],
        [matrix[3], matrix[7], matrix[11], matrix[15]],
    ]
}

pub fn inverse_transpose_linear(
    matrix: [f32; 16],
    label: &str,
) -> Result<[[f32; 4]; 4], PbrtError> {
    validate_affine(matrix, label)?;
    let [a, b, c, _, d, e, f, _, g, h, i, _, _, _, _, _] = matrix;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(PbrtError::error(&format!(
            "{label} transform has a singular linear part."
        )));
    }
    let inverse_determinant = 1.0 / determinant;
    Ok(row_major_to_columns([
        (e * i - f * h) * inverse_determinant,
        (f * g - d * i) * inverse_determinant,
        (d * h - e * g) * inverse_determinant,
        0.0,
        (c * h - b * i) * inverse_determinant,
        (a * i - c * g) * inverse_determinant,
        (b * g - a * h) * inverse_determinant,
        0.0,
        (b * f - c * e) * inverse_determinant,
        (c * d - a * f) * inverse_determinant,
        (a * e - b * d) * inverse_determinant,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]))
}

pub fn validate_affine(matrix: [f32; 16], label: &str) -> Result<(), PbrtError> {
    if !matrix.iter().all(|value| value.is_finite()) {
        return Err(PbrtError::error(&format!(
            "{label} transform contains a non-finite value."
        )));
    }
    if matrix[12..16] != [0.0, 0.0, 0.0, 1.0] {
        return Err(PbrtError::error(&format!(
            "{label} transform must be affine."
        )));
    }
    let determinant = matrix[0] * (matrix[5] * matrix[10] - matrix[6] * matrix[9])
        - matrix[1] * (matrix[4] * matrix[10] - matrix[6] * matrix[8])
        + matrix[2] * (matrix[4] * matrix[9] - matrix[5] * matrix[8]);
    if determinant == 0.0 {
        return Err(PbrtError::error(&format!(
            "{label} transform is not invertible."
        )));
    }
    Ok(())
}

pub fn row_major_to_tlas_transform(matrix: [f32; 16]) -> [f32; 12] {
    [
        matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5], matrix[6], matrix[7],
        matrix[8], matrix[9], matrix[10], matrix[11],
    ]
}
