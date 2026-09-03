use bytemuck::{Pod, Zeroable};

use crate::gpu::ir::flat;
use crate::util::error::PbrtError;

pub const WORKGROUP_SIZE: u32 = 8;
pub const RAY_T_MIN: f32 = 0.0;
pub const RAY_T_MAX: f32 = f32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub camera_to_world: [[f32; 4]; 4],
    pub raster_to_camera: [[f32; 4]; 4],
    pub width: u32,
    pub height: u32,
    pub padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 4],
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
    pub padding: [u32; 2],
    pub world_from_object: [[f32; 4]; 4],
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
    pub pixel_index: u32,
    pub padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HitRecord {
    pub t: f32,
    pub hit: u32,
    pub instance_custom_data: u32,
    pub primitive_index: u32,
    pub barycentric: [f32; 4],
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
    let dy = (ymax - ymin) / height as f32;
    let dx = dy * width as f32 / height as f32;
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
        width,
        height,
        padding: [0; 2],
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
