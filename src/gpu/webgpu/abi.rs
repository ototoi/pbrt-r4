use bytemuck::{Pod, Zeroable};

use crate::gpu::flat;
use crate::util::error::PbrtError;

pub const WORKGROUP_SIZE: u32 = 8;
pub const RAY_T_MIN: f32 = 0.0;
pub const RAY_T_MAX: f32 = f32::MAX;
pub const LIGHT_KIND_POINT: u32 = 0;
pub const LIGHT_KIND_AREA: u32 = 1;
pub const LIGHT_KIND_SPOT: u32 = 2;
pub const LIGHT_KIND_DISTANT: u32 = 3;
pub const LIGHT_KIND_UNIFORM_INFINITE: u32 = 4;
pub const LIGHT_KIND_IMAGE_INFINITE: u32 = 5;
pub const LIGHT_KIND_PORTAL_IMAGE_INFINITE: u32 = 6;
pub const LIGHT_SAMPLER_KIND_UNIFORM: u32 = 0;
pub const LIGHT_SAMPLER_KIND_BVH: u32 = 1;
pub const INVALID_INDEX: u32 = u32::MAX;
pub const INSTANCE_ORIENTATION_FLAG_REVERSED: u32 = 1 << 0;
pub const INSTANCE_ORIENTATION_FLAG_TRANSFORM_SWAPS_HANDEDNESS: u32 = 1 << 1;

pub fn instance_orientation_flags(
    reverse_orientation: bool,
    transform_swaps_handedness: bool,
) -> u32 {
    (if reverse_orientation {
        INSTANCE_ORIENTATION_FLAG_REVERSED
    } else {
        0
    }) | (if transform_swaps_handedness {
        INSTANCE_ORIENTATION_FLAG_TRANSFORM_SWAPS_HANDEDNESS
    } else {
        0
    })
}
pub const TEXTURE_OPERATION_IMAGE: u32 = 0;
pub const TEXTURE_OPERATION_CONSTANT: u32 = 1;
pub const TEXTURE_OPERATION_SCALE: u32 = 2;
pub const TEXTURE_OPERATION_MIX: u32 = 3;
pub const TEXTURE_OPERATION_CHECKERBOARD: u32 = 4;
pub const TEXTURE_OPERATION_DIRECTION_MIX: u32 = 5;
pub const TEXTURE_OPERATION_DOTS: u32 = 6;
pub const TEXTURE_OPERATION_FBM: u32 = 7;
pub const TEXTURE_OPERATION_WRINKLED: u32 = 8;
pub const TEXTURE_OPERATION_WINDY: u32 = 9;
pub const TEXTURE_OPERATION_BILERP: u32 = 10;
pub const TEXTURE_OPERATION_MARBLE: u32 = 11;

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
    pub reserved: [u32; 9],
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
pub struct TextureNodeRecord {
    pub kind: u32,
    pub first_child: u32,
    pub child_count: u32,
    pub implementation_hash: u32,
    pub swrap_mode: u32,
    pub twrap_mode: u32,
    pub color_space: u32,
    pub texture_index: u32,
    pub operation: u32,
    pub mapping_kind: u32,
    pub sampler: u32,
    pub image_filter_mode: u32,
    pub constant_value: [f32; 4],
    pub mapping: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Instance {
    pub geometry: u32,
    pub material_root: u32,
    pub area_light: u32,
    pub orientation_flags: u32,
    pub medium_inside: u32,
    pub medium_outside: u32,
    pub padding: [u32; 2],
    pub world_from_object: [[f32; 4]; 4],
    pub normal_from_object: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MediumRecord {
    pub kind: u32,
    pub sigma_a: u32,
    pub sigma_s: u32,
    pub le: u32,
    pub g: f32,
    pub padding: [u32; 3],
    pub medium_to_world: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MaterialNode {
    pub kind: u32,
    pub attribute_offset: u32,
    pub attribute_count: u32,
    pub parent: u32,
    pub parent_slot: u32,
    pub child0: u32,
    pub child1: u32,
    pub displacement_texture_root: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AttributeRef {
    pub kind: u32,
    pub index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeasuredBsdfRecord {
    pub ndf: u32,
    pub sigma: u32,
    pub vndf: u32,
    pub luminance: u32,
    pub spectra: u32,
    pub isotropic: u32,
    pub padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeasuredTableRecord {
    pub size: [u32; 2],
    pub parameter_count: u32,
    pub padding0: u32,
    pub parameter_sizes: [u32; 3],
    pub padding1: u32,
    pub parameter_strides: [u32; 3],
    pub padding2: u32,
    pub parameter_value_offsets: [u32; 3],
    pub padding3: u32,
    pub data_offset: u32,
    pub marginal_cdf_offset: u32,
    pub conditional_cdf_offset: u32,
    pub padding4: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TextureRootRecord {
    pub texture_node: u32,
    pub instruction_count: u32,
    pub result: u32,
    pub spectrum_type: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DenseSpectrum {
    pub samples: [f32; flat::DENSE_SAMPLE_COUNT],
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RayWorkItem {
    pub origin: [f32; 4],
    pub direction: [f32; 4],
    pub beta: [f32; 4],
    pub r_u: [f32; 4],
    pub r_l: [f32; 4],
    pub prev_position: [f32; 4],
    pub prev_position_error: [f32; 4],
    pub prev_geometric_normal: [f32; 4],
    pub prev_shading_normal: [f32; 4],
    pub pixel_index: u32,
    pub depth: u32,
    pub eta_scale: f32,
    pub prev_pdf: f32,
    pub prev_specular: u32,
    pub medium_id: u32,
    pub medium_segment_index: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ShadowRayWorkItem {
    pub origin: [f32; 4],
    pub direction: [f32; 4],
    pub endpoint: [f32; 4],
    pub max_t: f32,
    pub medium_id: u32,
    pub depth: u32,
    pub infinite_distance: u32,
    pub direct: [f32; 4],
    pub r_u: [f32; 4],
    pub r_l: [f32; 4],
    pub transmittance: [f32; 4],
    pub inv_w_u: [f32; 4],
    pub inv_w_l: [f32; 4],
    pub pixel_index: u32,
    pub segment_index: u32,
    pub padding: [u32; 2],
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
    pub tangent: [f32; 4],
    pub dpdu: [f32; 4],
    pub dpdv: [f32; 4],
    pub dndu: [f32; 4],
    pub dndv: [f32; 4],
    pub dpdx: [f32; 4],
    pub dpdy: [f32; 4],
    // Keep the vec4 before the vec2 so WGSL inserts no implicit padding.
    pub uv_differentials: [f32; 4],
    pub uv: [f32; 2],
    pub material_root: u32,
    pub flags: u32,
    pub attributes_eval_work_item: u32,
    pub padding: u32,
    pub tail_padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AttributesEvalWorkItem {
    pub material_node: u32,
    pub child_work_item0: u32,
    pub child_work_item1: u32,
    pub bxdf_kind: u32,
    pub selected_child_work_item: u32,
    pub padding: [u32; 3],
    pub values: [[f32; 4]; 10],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MaterialRoot {
    pub node_offset: u32,
    pub node_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TextureEvalResult {
    pub material_node: u32,
    pub attribute_ordinal: u32,
    pub texture_root: u32,
    pub valid: u32,
    pub value: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightRecord {
    pub kind: u32,
    pub attribute_offset: u32,
    pub attribute_count: u32,
    pub sampling_model: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightSamplingModel {
    pub kind: u32,
    pub geometry_kind: u32,
    pub geometry_index: u32,
    pub direction_index: u32,
    pub distribution_offset_words: u32,
    pub distribution_count: u32,
    pub total_area: f32,
    pub flags: u32,
    pub world_to_light: [[f32; 4]; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PortalImageInfiniteRecord {
    pub portal: [[f32; 4]; 4],
    pub world_to_portal: [[f32; 4]; 3],
    pub distribution_offset: u32,
    pub width: u32,
    pub height: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PortalDistributionTexel {
    pub function: f32,
    pub summed_area: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DirectLightSample {
    pub direction_pdf: [f32; 4],
    pub radiance: [f32; 4],
    pub position: [f32; 3],
    pub light_kind: u32,
    pub position_error: [f32; 3],
    pub use_mis: u32,
    pub normal: [f32; 3],
    pub valid: u32,
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct QueueCounters {
    pub current: QueueState,
    pub next: QueueState,
    pub shadow: QueueState,
    pub material: QueueState,
    pub hit_area: QueueState,
    pub escaped: QueueState,
    pub direct: QueueState,
    pub scatter_diffuse: QueueState,
    pub scatter_diffuse_transmission: QueueState,
    pub scatter_conductor: QueueState,
    pub scatter_dielectric: QueueState,
    pub scatter_thin_dielectric: QueueState,
    pub scatter_measured: QueueState,
    pub scatter_coated: QueueState,
    pub medium_continuation: QueueState,
    pub shadow_continuation: QueueState,
    pub medium_active: QueueState,
    pub shadow_active: QueueState,
}

/// Indirect dispatch arguments, laid out identically to `wgpu::util::DispatchIndirectArgs`.
pub const QUEUE_DISPATCH_SLOT_MATERIAL_EVAL: u64 = 0;
pub const QUEUE_DISPATCH_SLOT_DIRECT_EVAL: u64 = 1;
pub const QUEUE_DISPATCH_SLOT_SCATTER_DIFFUSE: u64 = 2;
pub const QUEUE_DISPATCH_SLOT_SCATTER_DIFFUSE_TRANSMISSION: u64 = 3;
pub const QUEUE_DISPATCH_SLOT_SCATTER_CONDUCTOR: u64 = 4;
pub const QUEUE_DISPATCH_SLOT_SCATTER_DIELECTRIC: u64 = 5;
pub const QUEUE_DISPATCH_SLOT_SCATTER_THIN_DIELECTRIC: u64 = 6;
pub const QUEUE_DISPATCH_SLOT_SCATTER_MEASURED: u64 = 7;
pub const QUEUE_DISPATCH_SLOT_SCATTER_COATED: u64 = 8;
pub const QUEUE_DISPATCH_SLOT_CURRENT_RAY: u64 = 9;
pub const QUEUE_DISPATCH_SLOT_ESCAPED: u64 = 10;
pub const QUEUE_DISPATCH_SLOT_HIT_AREA: u64 = 11;
pub const QUEUE_DISPATCH_SLOT_SHADOW: u64 = 12;
pub const QUEUE_DISPATCH_SLOT_NEXT_RAY: u64 = 13;
pub const QUEUE_DISPATCH_SLOT_COUNT: u64 = 14;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DispatchIndirectArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RenderError {
    pub value: u32,
    pub padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PixelSampleState {
    pub radiance: [f32; 4],
    pub lambda: [f32; 4],
    pub lambda_pdf: [f32; 4],
    pub direct: [f32; 4],
    pub indirect: [f32; 4],
}

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
    })
}

fn inverse_affine(transform: [f32; 16], label: &str) -> Result<[[f32; 4]; 4], PbrtError> {
    let [a, b, c, tx, d, e, f, ty, g, h, i, tz, _, _, _, _] = transform;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(PbrtError::error(&format!(
            "{label} transform has a singular linear part."
        )));
    }
    let inverse_determinant = 1.0 / determinant;
    let inverse_linear = [
        (e * i - f * h) * inverse_determinant,
        (c * h - b * i) * inverse_determinant,
        (b * f - c * e) * inverse_determinant,
        (f * g - d * i) * inverse_determinant,
        (a * i - c * g) * inverse_determinant,
        (c * d - a * f) * inverse_determinant,
        (d * h - e * g) * inverse_determinant,
        (b * g - a * h) * inverse_determinant,
        (a * e - b * d) * inverse_determinant,
    ];
    let inverse_translation = [
        -(inverse_linear[0] * tx + inverse_linear[1] * ty + inverse_linear[2] * tz),
        -(inverse_linear[3] * tx + inverse_linear[4] * ty + inverse_linear[5] * tz),
        -(inverse_linear[6] * tx + inverse_linear[7] * ty + inverse_linear[8] * tz),
    ];
    Ok(row_major_to_columns([
        inverse_linear[0],
        inverse_linear[1],
        inverse_linear[2],
        inverse_translation[0],
        inverse_linear[3],
        inverse_linear[4],
        inverse_linear[5],
        inverse_translation[1],
        inverse_linear[6],
        inverse_linear[7],
        inverse_linear[8],
        inverse_translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ]))
}

fn minimum_perspective_camera_direction_differentials(
    screen_window: [f32; 4],
    resolution: [u32; 2],
    tan_half_fov: f32,
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
        let direction = normalize3(p_camera);
        let (frame_x, frame_y) = coordinate_system3(direction);
        let rx_direction = normalize3(add3(p_camera, delta_x));
        let ry_direction = normalize3(add3(p_camera, delta_y));
        let local_x = [
            dot3(rx_direction, frame_x),
            dot3(rx_direction, frame_y),
            dot3(rx_direction, direction),
        ];
        let local_y = [
            dot3(ry_direction, frame_x),
            dot3(ry_direction, frame_y),
            dot3(ry_direction, direction),
        ];
        let dx = sub3(local_x, [0.0, 0.0, 1.0]);
        let dy = sub3(local_y, [0.0, 0.0, 1.0]);
        for (axis, differential) in [dx, dy].into_iter().enumerate() {
            let length_squared = dot3(differential, differential);
            if length_squared < minimum_length_squared[axis] {
                minimum_length_squared[axis] = length_squared;
                minimum[axis] = differential;
            }
        }
    }
    minimum
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let length = dot3(v, v).sqrt();
    [v[0] / length, v[1] / length, v[2] / length]
}

fn coordinate_system3(z: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let sign = if z[2].is_sign_negative() { -1.0 } else { 1.0 };
    let a = -1.0 / (sign + z[2]);
    let b = z[0] * z[1] * a;
    (
        [1.0 + sign * z[0] * z[0] * a, sign * b, -sign * z[0]],
        [b, sign + z[1] * z[1] * a, -z[1]],
    )
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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
        reserved: [0; 9],
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
