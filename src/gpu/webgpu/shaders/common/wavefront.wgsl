const WAVEFRONT_STATE_INACTIVE: u32 = 0u;
const WAVEFRONT_STATE_RAY: u32 = 1u;
const WAVEFRONT_STATE_MISS: u32 = 2u;
const WAVEFRONT_STATE_HIT: u32 = 3u;

struct WavefrontSlot {
    ray_origin: vec4<f32>,
    ray_direction: vec4<f32>,
    intersection_ids: vec4<u32>,
    intersection_data: vec4<f32>,
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
    contribution: vec4<f32>,
    radiance: vec4<f32>,
    throughput: vec4<f32>,
    path_info: vec4<u32>,
};

struct WavefrontArena {
    control: vec4<u32>,
    slots: array<WavefrontSlot>,
};

@group(0) @binding(10)
var<storage, read_write> wavefront: WavefrontArena;
