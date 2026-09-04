enable wgpu_ray_query;

const RAY_T_MAX: f32 = 3.402823466e+38;
const MACHINE_EPSILON: f32 = 1.1920929e-7;
const PI: f32 = 3.141592653589793;
const MATERIAL_KIND_NORMAL: u32 = 0u;
const MATERIAL_KIND_UV: u32 = 1u;
const MATERIAL_KIND_DIFFUSE: u32 = 2u;
const LIGHT_KIND_AREA: u32 = 1u;
const LIGHT_KIND_POINT: u32 = 0u;

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
    area_light_data_offset: u32,
};

struct Vertex {
    position: vec4<f32>,
    normal: vec4<f32>,
    tangent: vec4<f32>,
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
    area_light: u32,
    _padding: u32,
    world_from_object: mat4x4<f32>,
    normal_from_object: mat4x4<f32>,
};

struct RayWorkItem {
    origin: vec4<f32>,
    direction: vec4<f32>,
    throughput: vec4<f32>,
    pixel_index: u32,
    depth: u32,
    _padding: u32,
    prev_pdf: f32,
};

struct SurfaceWorkItem {
    t: f32,
    hit: u32,
    instance_custom_data: u32,
    primitive_index: u32,
    barycentric: vec4<f32>,
    position: vec4<f32>,
    position_error: vec4<f32>,
    normal: vec4<f32>,
    material: u32,
    flags: u32,
    _padding: vec2<u32>,
};

struct PointLight {
    position: vec4<f32>,
    intensity: vec4<f32>,
};

struct QueueState {
    count: atomic<u32>,
    capacity: u32,
    overflow: atomic<u32>,
    _padding: u32,
};

struct LightSelection {
    index: u32,
    pmf: f32,
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
var<storage, read_write> surfaces: array<SurfaceWorkItem>;
@group(0) @binding(9)
var<storage, read_write> framebuffer: array<vec4<f32>>;
@group(0) @binding(10)
var<storage, read_write> wavefront_queue: array<atomic<u32>>;

const CURRENT_COUNT: u32 = 0u;
const CURRENT_OVERFLOW: u32 = 2u;
const NEXT_COUNT: u32 = 4u;
const NEXT_OVERFLOW: u32 = 6u;
const SHADOW_COUNT: u32 = 8u;
const SHADOW_OVERFLOW: u32 = 10u;
const MATERIAL_COUNT: u32 = 12u;
const MATERIAL_OVERFLOW: u32 = 14u;
const HIT_AREA_COUNT: u32 = 16u;
const HIT_AREA_OVERFLOW: u32 = 18u;
const ESCAPED_COUNT: u32 = 20u;
const ESCAPED_OVERFLOW: u32 = 22u;
const RENDER_ERROR: u32 = 23u;
const RAY_WORDS: u32 = 16u;
const SAMPLE_STATE_OFFSET: u32 = 24u;
const SAMPLE_STATE_WORDS: u32 = 8u;

fn sample_state_word(pixel_index: u32, word: u32) -> u32 {
    return SAMPLE_STATE_OFFSET + pixel_index * SAMPLE_STATE_WORDS + word;
}

fn ray_data_offset() -> u32 {
    return SAMPLE_STATE_OFFSET + pixel_count() * SAMPLE_STATE_WORDS;
}

fn load_sample_radiance(pixel_index: u32) -> vec4<f32> {
    return vec4<f32>(
        bitcast<f32>(atomicLoad(&wavefront_queue[sample_state_word(pixel_index, 0u)])),
        bitcast<f32>(atomicLoad(&wavefront_queue[sample_state_word(pixel_index, 1u)])),
        bitcast<f32>(atomicLoad(&wavefront_queue[sample_state_word(pixel_index, 2u)])),
        bitcast<f32>(atomicLoad(&wavefront_queue[sample_state_word(pixel_index, 3u)])),
    );
}

fn store_sample_radiance(pixel_index: u32, radiance: vec4<f32>) {
    if (radiance.x != radiance.x || radiance.y != radiance.y || radiance.z != radiance.z
        || radiance.w != radiance.w || abs(radiance.x) > RAY_T_MAX || abs(radiance.y) > RAY_T_MAX
        || abs(radiance.z) > RAY_T_MAX || abs(radiance.w) > RAY_T_MAX) {
        atomicStore(&wavefront_queue[RENDER_ERROR], 1u);
    }
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 0u)], bitcast<u32>(radiance.x));
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 1u)], bitcast<u32>(radiance.y));
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 2u)], bitcast<u32>(radiance.z));
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 3u)], bitcast<u32>(radiance.w));
}

fn store_sample_metadata(pixel_index: u32) {
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 4u)], pixel_index);
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 5u)], viewport.sample_index);
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 6u)], 0u);
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 7u)], 0u);
}

fn shadow_data_offset() -> u32 {
    return ray_data_offset() + pixel_count() * RAY_WORDS * 2u;
}

fn classification_capacity() -> u32 {
    return pixel_count() * (viewport.max_depth + 1u);
}

fn current_ray_count() -> u32 {
    return atomicLoad(&wavefront_queue[CURRENT_COUNT]);
}

fn next_ray_count() -> u32 {
    return atomicLoad(&wavefront_queue[NEXT_COUNT]);
}

fn shadow_ray_count() -> u32 {
    return atomicLoad(&wavefront_queue[SHADOW_COUNT]);
}

const SHADOW_WORDS: u32 = 20u;

fn shadow_ray_word(index: u32, word: u32) -> u32 {
    return shadow_data_offset() + index * SHADOW_WORDS + word;
}

fn append_shadow_ray(pixel_index: u32, origin: vec3<f32>, direction: vec3<f32>, t: f32, direct: vec3<f32>) {
    let index = atomicAdd(&wavefront_queue[SHADOW_COUNT], 1u);
    if (index < pixel_count()) {
        atomicStore(&wavefront_queue[shadow_ray_word(index, 0u)], bitcast<u32>(origin.x));
        atomicStore(&wavefront_queue[shadow_ray_word(index, 1u)], bitcast<u32>(origin.y));
        atomicStore(&wavefront_queue[shadow_ray_word(index, 2u)], bitcast<u32>(origin.z));
        atomicStore(&wavefront_queue[shadow_ray_word(index, 4u)], bitcast<u32>(direction.x));
        atomicStore(&wavefront_queue[shadow_ray_word(index, 5u)], bitcast<u32>(direction.y));
        atomicStore(&wavefront_queue[shadow_ray_word(index, 6u)], bitcast<u32>(direction.z));
        atomicStore(&wavefront_queue[shadow_ray_word(index, 7u)], 0u);
        atomicStore(&wavefront_queue[shadow_ray_word(index, 8u)], bitcast<u32>(t));
        atomicStore(&wavefront_queue[shadow_ray_word(index, 12u)], bitcast<u32>(direct.x));
        atomicStore(&wavefront_queue[shadow_ray_word(index, 13u)], bitcast<u32>(direct.y));
        atomicStore(&wavefront_queue[shadow_ray_word(index, 14u)], bitcast<u32>(direct.z));
        atomicStore(&wavefront_queue[shadow_ray_word(index, 15u)], 0u);
        atomicStore(&wavefront_queue[shadow_ray_word(index, 16u)], pixel_index);
    } else {
        atomicStore(&wavefront_queue[SHADOW_OVERFLOW], 1u);
    }
}

fn load_shadow_pixel(index: u32) -> u32 {
    return atomicLoad(&wavefront_queue[shadow_ray_word(index, 16u)]);
}

fn load_shadow_vec3(index: u32, word: u32) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(atomicLoad(&wavefront_queue[shadow_ray_word(index, word)])),
        bitcast<f32>(atomicLoad(&wavefront_queue[shadow_ray_word(index, word + 1u)])),
        bitcast<f32>(atomicLoad(&wavefront_queue[shadow_ray_word(index, word + 2u)])),
    );
}

fn load_shadow_t(index: u32) -> f32 {
    return bitcast<f32>(atomicLoad(&wavefront_queue[shadow_ray_word(index, 8u)]));
}

fn load_shadow_direct(index: u32) -> vec3<f32> {
    return load_shadow_vec3(index, 12u);
}

fn classification_word(base: u32, index: u32) -> u32 {
    return shadow_data_offset() + pixel_count() * SHADOW_WORDS + base + index;
}

fn append_material_eval(pixel_index: u32) {
    let index = atomicAdd(&wavefront_queue[MATERIAL_COUNT], 1u);
    if (index < classification_capacity()) {
        atomicStore(&wavefront_queue[classification_word(0u, index)], pixel_index);
    } else {
        atomicStore(&wavefront_queue[MATERIAL_OVERFLOW], 1u);
    }
}

fn material_eval_count() -> u32 {
    return atomicLoad(&wavefront_queue[MATERIAL_COUNT]);
}

fn load_material_eval_pixel(index: u32) -> u32 {
    return atomicLoad(&wavefront_queue[classification_word(0u, index)]);
}

fn find_current_ray_for_pixel(pixel_index: u32) -> u32 {
    let count = current_ray_count();
    for (var index = 0u; index < count; index++) {
        if (load_current_ray(index).pixel_index == pixel_index) {
            return index;
        }
    }
    return 0xffffffffu;
}

fn append_hit_area_light(pixel_index: u32) {
    let index = atomicAdd(&wavefront_queue[HIT_AREA_COUNT], 1u);
    if (index < classification_capacity()) {
        atomicStore(
            &wavefront_queue[classification_word(classification_capacity(), index)],
            pixel_index,
        );
    } else {
        atomicStore(&wavefront_queue[HIT_AREA_OVERFLOW], 1u);
    }
}

fn hit_area_light_count() -> u32 {
    return atomicLoad(&wavefront_queue[HIT_AREA_COUNT]);
}

fn load_hit_area_pixel(index: u32) -> u32 {
    return atomicLoad(&wavefront_queue[classification_word(classification_capacity(), index)]);
}

fn escaped_data_offset() -> u32 {
    return shadow_data_offset()
        + pixel_count() * RAY_WORDS * 2u
        + pixel_count() * SHADOW_WORDS
        + classification_capacity() * 2u;
}

fn escaped_ray_count() -> u32 {
    return atomicLoad(&wavefront_queue[ESCAPED_COUNT]);
}

fn append_escaped_ray(pixel_index: u32) {
    let index = atomicAdd(&wavefront_queue[ESCAPED_COUNT], 1u);
    if (index < classification_capacity()) {
        atomicStore(&wavefront_queue[escaped_data_offset() + index], pixel_index);
    } else {
        atomicStore(&wavefront_queue[ESCAPED_OVERFLOW], 1u);
    }
}

fn current_ray_word(index: u32, word: u32) -> u32 {
    return ray_data_offset() + index * RAY_WORDS + word;
}

fn next_ray_word(index: u32, word: u32) -> u32 {
    return ray_data_offset() + pixel_count() * RAY_WORDS + index * RAY_WORDS + word;
}

fn load_ray(base: u32) -> RayWorkItem {
    return RayWorkItem(
        vec4<f32>(
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 0u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 1u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 2u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 3u])),
        ),
        vec4<f32>(
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 4u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 5u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 6u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 7u])),
        ),
        vec4<f32>(
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 8u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 9u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 10u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 11u])),
        ),
        atomicLoad(&wavefront_queue[base + 12u]),
        atomicLoad(&wavefront_queue[base + 13u]),
        atomicLoad(&wavefront_queue[base + 14u]),
        bitcast<f32>(atomicLoad(&wavefront_queue[base + 15u])),
    );
}

fn store_ray(base: u32, ray: RayWorkItem) {
    atomicStore(&wavefront_queue[base + 0u], bitcast<u32>(ray.origin.x));
    atomicStore(&wavefront_queue[base + 1u], bitcast<u32>(ray.origin.y));
    atomicStore(&wavefront_queue[base + 2u], bitcast<u32>(ray.origin.z));
    atomicStore(&wavefront_queue[base + 3u], bitcast<u32>(ray.origin.w));
    atomicStore(&wavefront_queue[base + 4u], bitcast<u32>(ray.direction.x));
    atomicStore(&wavefront_queue[base + 5u], bitcast<u32>(ray.direction.y));
    atomicStore(&wavefront_queue[base + 6u], bitcast<u32>(ray.direction.z));
    atomicStore(&wavefront_queue[base + 7u], bitcast<u32>(ray.direction.w));
    atomicStore(&wavefront_queue[base + 8u], bitcast<u32>(ray.throughput.x));
    atomicStore(&wavefront_queue[base + 9u], bitcast<u32>(ray.throughput.y));
    atomicStore(&wavefront_queue[base + 10u], bitcast<u32>(ray.throughput.z));
    atomicStore(&wavefront_queue[base + 11u], bitcast<u32>(ray.throughput.w));
    atomicStore(&wavefront_queue[base + 12u], ray.pixel_index);
    atomicStore(&wavefront_queue[base + 13u], ray.depth);
    atomicStore(&wavefront_queue[base + 14u], ray._padding);
    atomicStore(&wavefront_queue[base + 15u], bitcast<u32>(ray.prev_pdf));
}

fn load_current_ray(index: u32) -> RayWorkItem {
    return load_ray(current_ray_word(index, 0u));
}

fn load_next_ray(index: u32) -> RayWorkItem {
    return load_ray(next_ray_word(index, 0u));
}

fn store_current_ray(index: u32, ray: RayWorkItem) {
    store_ray(current_ray_word(index, 0u), ray);
}

fn store_next_ray(index: u32, ray: RayWorkItem) {
    store_ray(next_ray_word(index, 0u), ray);
}

fn load_area_emission(index: u32) -> vec4<f32> {
    let base = viewport.area_light_data_offset + index * 12u;
    return vec4<f32>(
        bitcast<f32>(material_light_data[base + 2u]),
        bitcast<f32>(material_light_data[base + 3u]),
        bitcast<f32>(material_light_data[base + 4u]),
        0.0,
    );
}

fn load_area_word(index: u32, word: u32) -> u32 {
    return material_light_data[viewport.area_light_data_offset + index * 12u + word];
}

fn load_area_instance(index: u32) -> u32 {
    return load_area_word(index, 0u);
}

fn load_area_total(index: u32) -> f32 {
    return bitcast<f32>(load_area_word(index, 6u));
}

fn load_area_distribution_offset(index: u32) -> u32 {
    return load_area_word(index, 7u);
}

fn load_area_distribution_count(index: u32) -> u32 {
    return load_area_word(index, 8u);
}

fn load_area_two_sided(index: u32) -> bool {
    return load_area_word(index, 1u) != 0u;
}

fn load_triangle_primitive(offset: u32, index: u32) -> u32 {
    return material_light_data[offset + index * 4u];
}

fn load_triangle_cdf(offset: u32, index: u32) -> f32 {
    return bitcast<f32>(material_light_data[offset + index * 4u + 1u]);
}

fn pixel_count() -> u32 {
    return viewport.width * viewport.height;
}

fn load_material_kind(index: u32) -> u32 {
    return material_light_data[index * 4u];
}

fn load_point_light(index: u32) -> PointLight {
    let base = viewport.light_data_offset + viewport.light_count * 4u + index * 8u;
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

fn load_light_kind(index: u32) -> u32 {
    return material_light_data[viewport.light_data_offset + index * 4u];
}

fn load_light_payload(index: u32) -> u32 {
    return material_light_data[viewport.light_data_offset + index * 4u + 1u];
}

fn light_pmf_for_area(area_index: u32) -> f32 {
    if (viewport.light_count == 0u) {
        return 0.0;
    }
    for (var index = 0u; index < viewport.light_count; index++) {
        if (load_light_kind(index) == LIGHT_KIND_AREA && load_light_payload(index) == area_index) {
            return 1.0 / f32(viewport.light_count);
        }
    }
    return 0.0;
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

fn sample_uniform_light(pixel_index: u32) -> LightSelection {
    if (viewport.light_count == 0u) {
        return LightSelection(0u, 0.0);
    }
    return LightSelection(
        hash_u32(viewport.seed ^ pixel_index ^ viewport.sample_index) % viewport.light_count,
        1.0 / f32(viewport.light_count),
    );
}

fn gamma(n: f32) -> f32 {
    return (n * MACHINE_EPSILON) / (1.0 - n * MACHINE_EPSILON);
}

fn offset_ray_origin(position: vec3<f32>, error: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let offset = normal * dot(abs(normal), error);
    return position + select(-offset, offset, dot(offset, normal) >= 0.0);
}

fn make_tangent(normal: vec3<f32>) -> vec3<f32> {
    if (abs(normal.x) > 0.1) {
        return normalize(cross(vec3<f32>(0.0, 1.0, 0.0), normal));
    }
    return normalize(cross(vec3<f32>(1.0, 0.0, 0.0), normal));
}
