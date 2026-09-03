enable wgpu_ray_query;

const RAY_T_MAX: f32 = 3.402823466e+38;
const RAY_EPSILON: f32 = 0.0001;
const PI: f32 = 3.141592653589793;
const MATERIAL_KIND_NORMAL: u32 = 0u;
const MATERIAL_KIND_UV: u32 = 1u;
const MATERIAL_KIND_DIFFUSE: u32 = 2u;

struct CameraUniform {
    camera_to_world: mat4x4<f32>,
    raster_to_camera: mat4x4<f32>,
};

struct ViewportUniform {
    width: u32,
    height: u32,
    sample_index: u32,
    max_depth: u32,
    seed: u32,
    light_data_offset: u32,
    light_count: u32,
    _padding: u32,
};

struct Vertex {
    position: vec4<f32>,
    normal: vec4<f32>,
    uv: vec2<f32>,
    _padding: vec2<u32>,
};

struct Geometry {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    index_count: u32,
};

struct Instance {
    geometry: u32,
    material: u32,
    _padding: vec2<u32>,
    world_from_object: mat4x4<f32>,
};

struct RayWorkItem {
    origin: vec4<f32>,
    direction: vec4<f32>,
    throughput: vec4<f32>,
    pixel_index: u32,
    depth: u32,
    is_active: u32,
    _padding: u32,
};

struct SurfaceWorkItem {
    t: f32,
    hit: u32,
    instance_custom_data: u32,
    primitive_index: u32,
    barycentric: vec4<f32>,
    position: vec4<f32>,
    normal: vec4<f32>,
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
    shadow_t: f32,
    shadow_visible: u32,
    material: u32,
    flags: u32,
    direct: vec4<f32>,
};

struct PointLight {
    position: vec4<f32>,
    intensity: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> viewport: ViewportUniform;
@group(0) @binding(2)
var tlas: acceleration_structure;
@group(0) @binding(3)
var<storage, read> vertices: array<Vertex>;
@group(0) @binding(4)
var<storage, read> indices: array<u32>;
@group(0) @binding(5)
var<storage, read> geometries: array<Geometry>;
@group(0) @binding(6)
var<storage, read> instances: array<Instance>;
@group(0) @binding(7)
var<storage, read> material_light_data: array<u32>;
@group(0) @binding(8)
var<storage, read_write> rays: array<RayWorkItem>;
@group(0) @binding(9)
var<storage, read_write> surfaces: array<SurfaceWorkItem>;
@group(0) @binding(10)
var<storage, read_write> framebuffer: array<vec4<f32>>;

fn pixel_count() -> u32 {
    return viewport.width * viewport.height;
}

fn load_material_kind(index: u32) -> u32 {
    return material_light_data[index * 4u];
}

fn load_point_light(index: u32) -> PointLight {
    let base = viewport.light_data_offset + index * 8u;
    return PointLight(
        vec4<f32>(
            bitcast<f32>(material_light_data[base]),
            bitcast<f32>(material_light_data[base + 1u]),
            bitcast<f32>(material_light_data[base + 2u]),
            bitcast<f32>(material_light_data[base + 3u]),
        ),
        vec4<f32>(
            bitcast<f32>(material_light_data[base + 4u]),
            bitcast<f32>(material_light_data[base + 5u]),
            bitcast<f32>(material_light_data[base + 6u]),
            bitcast<f32>(material_light_data[base + 7u]),
        ),
    );
}

fn current_ray_index(pixel_index: u32) -> u32 {
    if (rays[pixel_index].is_active != 0u) {
        return pixel_index;
    }
    return pixel_index + pixel_count();
}

fn hash_u32(value: u32) -> u32 {
    var h = value;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

fn random01(pixel_index: u32, dimension: u32) -> f32 {
    let value = viewport.seed
        ^ (pixel_index * 0x9e3779b9u)
        ^ (viewport.sample_index * 0x85ebca6bu)
        ^ (dimension * 0xc2b2ae35u);
    return f32(hash_u32(value) & 0x00ffffffu) / 16777216.0;
}

fn make_tangent(normal: vec3<f32>) -> vec3<f32> {
    if (abs(normal.x) > 0.1) {
        return normalize(cross(vec3<f32>(0.0, 1.0, 0.0), normal));
    }
    return normalize(cross(vec3<f32>(1.0, 0.0, 0.0), normal));
}
