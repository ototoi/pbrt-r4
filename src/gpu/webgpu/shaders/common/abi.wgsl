struct Camera {
    camera_from_raster: mat4x4<f32>,
    render_from_camera: mat4x4<f32>,
    viewport: vec4<f32>,
    bvh_info: vec4<u32>,
    sampler_info: vec4<u32>,
    filter_info: vec4<f32>,
    camera_info: vec4<f32>,
};

struct Primitive {
    first_vertex: u32,
    first_index: u32,
    material: u32,
    alpha: u32,
    _reserved_alpha: u32,
    flags: u32,
    // Explicit trailing padding keeps the storage-array stride at 32 bytes.
    _padding: vec2<u32>,
};

struct Material {
    reflectance: vec4<f32>,
    texture: u32,
    normal_map: u32,
    displacement: u32,
    flags: u32,
};

struct Vertex {
    position: vec4<f32>,
    uv: vec4<f32>,
    normal: vec4<f32>,
    tangent: vec4<f32>,
};

struct Transform {
    render_from_object: mat4x4<f32>,
    normal_from_object: mat4x4<f32>,
};

struct Light {
    position: vec4<f32>,
    intensity: vec4<f32>,
    kind: u32,
    primitive: u32,
    triangle: u32,
    flags: u32,
};

@group(0) @binding(0)
var<uniform> camera: Camera;
@group(0) @binding(1)
var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(3)
var<storage, read> vertices: array<Vertex>;
@group(0) @binding(4)
var<storage, read> indices: array<u32>;
@group(0) @binding(5)
var<storage, read> primitives: array<Primitive>;
@group(0) @binding(6)
var<storage, read> transforms: array<Transform>;
@group(0) @binding(7)
var<storage, read> materials: array<Material>;
@group(0) @binding(8)
var<storage, read> lights: array<Light>;
@group(0) @binding(9)
var<storage, read> scene_data: array<u32>;
