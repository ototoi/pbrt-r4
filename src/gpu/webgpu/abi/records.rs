use bytemuck::{Pod, Zeroable};

use crate::gpu::flat;

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
    pub intersection_normal_kind: u32,
    pub padding: [u32; 3],
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
    pub bssrdf_index: u32,
    pub padding: [u32; 3],
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
pub struct ImageInfiniteSamplingRecord {
    pub distribution_offset: u32,
    pub row_cdf_offset: u32,
    pub width: u32,
    pub height: u32,
    pub light_to_render: [[f32; 4]; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ImageInfiniteDistributionTexel {
    pub weight: f32,
    pub conditional_cdf: f32,
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
    pub medium_scatter: QueueState,
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
pub const QUEUE_DISPATCH_SLOT_MEDIUM_SCATTER: u64 = 14;
pub const QUEUE_DISPATCH_SLOT_COUNT: u64 = 15;

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
