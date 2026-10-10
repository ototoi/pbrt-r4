use crate::gpu::flat;
use crate::util::error::PbrtError;

use super::transform::{
    add3, coordinate_system, dot, inverse_affine, normalize, row_major_to_columns, sub3,
    validate_affine,
};
use super::uniforms::CameraUniform;

pub fn camera_uniform(
    camera: &flat::Camera,
    viewport: &flat::Viewport,
) -> Result<CameraUniform, PbrtError> {
    if camera.kind != "perspective" {
        return Err(PbrtError::error(&format!(
            "WebGPU camera kind \"{}\" is unsupported; expected perspective.",
            camera.kind
        )));
    }
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
    let world_to_camera = inverse_affine(camera.camera_to_world, "Camera")?;

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
    let [min_dir_differential_x, min_dir_differential_y] =
        minimum_perspective_camera_direction_differentials(
            camera.screen_window,
            [width, height],
            tan_half_fov,
            camera.lens_radius,
            camera.focal_distance,
        );
    Ok(CameraUniform {
        camera_to_world,
        raster_to_camera,
        world_to_camera,
        min_dir_differential_x: [
            min_dir_differential_x[0],
            min_dir_differential_x[1],
            min_dir_differential_x[2],
            0.0,
        ],
        min_dir_differential_y: [
            min_dir_differential_y[0],
            min_dir_differential_y[1],
            min_dir_differential_y[2],
            0.0,
        ],
        medium_id: camera.medium,
        disable_texture_filtering: u32::from(camera.disable_texture_filtering),
        disable_pixel_jitter: u32::from(camera.disable_pixel_jitter),
        padding: 0,
        lens_radius: camera.lens_radius,
        focal_distance: camera.focal_distance,
        lens_padding: [0.0; 2],
    })
}

fn minimum_perspective_camera_direction_differentials(
    screen_window: [f32; 4],
    resolution: [u32; 2],
    tan_half_fov: f32,
    lens_radius: f32,
    focal_distance: f32,
) -> [[f32; 3]; 2] {
    let [xmin, xmax, ymin, ymax] = screen_window;
    let dx = (xmax - xmin) / resolution[0] as f32;
    let dy = (ymax - ymin) / resolution[1] as f32;
    let delta_x = [dx * tan_half_fov, 0.0, 0.0];
    let delta_y = [0.0, -dy * tan_half_fov, 0.0];
    let mut minimum = [[0.0; 3], [0.0; 3]];
    let mut minimum_length_squared = [f32::INFINITY; 2];

    for sample in 0..512 {
        let t = sample as f32 / 511.0;
        let p_camera = [
            (xmin + dx * (t * resolution[0] as f32)) * tan_half_fov,
            (ymax - dy * (t * resolution[1] as f32)) * tan_half_fov,
            1.0,
        ];
        let direction = normalize(p_camera);
        let (frame_x, frame_y) = coordinate_system(direction);
        let (rx_direction, ry_direction) = if lens_radius > 0.0 {
            let rx_p_camera = add3(p_camera, delta_x);
            let ry_p_camera = add3(p_camera, delta_y);
            let rx_focus = [
                focal_distance * rx_p_camera[0] / rx_p_camera[2],
                focal_distance * rx_p_camera[1] / rx_p_camera[2],
                focal_distance,
            ];
            let ry_focus = [
                focal_distance * ry_p_camera[0] / ry_p_camera[2],
                focal_distance * ry_p_camera[1] / ry_p_camera[2],
                focal_distance,
            ];
            (normalize(rx_focus), normalize(ry_focus))
        } else {
            (
                normalize(add3(p_camera, delta_x)),
                normalize(add3(p_camera, delta_y)),
            )
        };
        let local_x = [
            dot(rx_direction, frame_x),
            dot(rx_direction, frame_y),
            dot(rx_direction, direction),
        ];
        let local_y = [
            dot(ry_direction, frame_x),
            dot(ry_direction, frame_y),
            dot(ry_direction, direction),
        ];
        let dx = sub3(local_x, [0.0, 0.0, 1.0]);
        let dy = sub3(local_y, [0.0, 0.0, 1.0]);
        for (axis, differential) in [dx, dy].into_iter().enumerate() {
            let length_squared = dot(differential, differential);
            if length_squared < minimum_length_squared[axis] {
                minimum_length_squared[axis] = length_squared;
                minimum[axis] = differential;
            }
        }
    }
    minimum
}
