const RAY_STATE_INACTIVE: u32 = 0u;
const RAY_STATE_ACTIVE: u32 = 1u;
const RAY_STATE_HIT: u32 = 2u;
const RAY_STATE_MISS: u32 = 3u;
const RAY_STATE_SURFACE: u32 = 4u;
const RAY_STATE_MATERIAL: u32 = 5u;
const RAY_STATE_SHADOW: u32 = 6u;
const RAY_STATE_VISIBLE: u32 = 7u;
const RAY_STATE_OCCLUDED: u32 = 8u;
const RAY_STATE_BOUNCE: u32 = 9u;

struct RayWorkItem {
    origin: vec4<f32>,
    direction: vec4<f32>,
    hit: vec4<f32>,
    indices: vec4<u32>,
    surface_position: vec4<f32>,
    surface_normal: vec4<f32>,
    surface_error: vec4<f32>,
    material_reflectance: vec4<f32>,
    direct_lighting: vec4<f32>,
    throughput: vec4<f32>,
    radiance: vec4<f32>,
};

struct WavefrontArena {
    sample_index: u32,
    capacity: u32,
    overflow: atomic<u32>,
    _padding: u32,
    rays: array<RayWorkItem>,
};

@group(0) @binding(10)
var<storage, read_write> arena: WavefrontArena;

fn pixel_count() -> u32 {
    return u32(camera.viewport.z) * u32(camera.viewport.w);
}
