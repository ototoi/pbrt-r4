enable wgpu_ray_query;

const RAY_T_MAX: f32 = 3.402823466e+38;
const MACHINE_EPSILON: f32 = 1.1920929e-7;
const PI: f32 = 3.141592653589793;
const MATERIAL_KIND_NORMAL: u32 = 0u;
const MATERIAL_KIND_UV: u32 = 1u;
const MATERIAL_KIND_DIFFUSE: u32 = 2u;
const LIGHT_KIND_AREA: u32 = 1u;
const LIGHT_KIND_POINT: u32 = 0u;
const LIGHT_SAMPLER_KIND_BVH: u32 = 1u;

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
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

struct SceneUniform {
    material_offset_words: u32,
    material_count: u32,
    light_record_offset_words: u32,
    light_count: u32,
    point_light_offset_words: u32,
    point_light_count: u32,
    area_light_offset_words: u32,
    area_light_count: u32,
    light_sampler_kind: u32,
    light_sampler_data_offset: u32,
    light_bvh_node_offset: u32,
    light_bvh_node_count: u32,
    light_leaf_offset: u32,
    light_leaf_count: u32,
    scene_data_words: u32,
    _reserved: u32,
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
    first_area_light: u32,
    orientation_flags: u32,
    world_from_object: mat4x4<f32>,
    normal_from_object: mat4x4<f32>,
};

struct RaySamples {
    direct: vec4<f32>,
    indirect: vec4<f32>,
};

struct RayWorkItem {
    origin: vec4<f32>,
    direction: vec4<f32>,
    throughput: vec4<f32>,
    prev_position: vec4<f32>,
    prev_position_error: vec4<f32>,
    prev_geometric_normal: vec4<f32>,
    prev_shading_normal: vec4<f32>,
    pixel_index: u32,
    depth: u32,
    inv_w_u: f32,
    inv_w_l: f32,
    prev_pdf: f32,
    _padding: vec3<u32>,
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
    geometric_normal: vec4<f32>,
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

struct DecodedLightBVHNode {
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
    direction: vec3<f32>,
    phi: f32,
    cos_theta_o: f32,
    cos_theta_e: f32,
    two_sided: bool,
    payload: u32,
    is_leaf: bool,
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
var<storage, read> scene_data: array<u32>;
@group(0) @binding(8)
var<storage, read_write> surfaces: array<SurfaceWorkItem>;
@group(0) @binding(9)
var<storage, read_write> framebuffer: array<vec4<f32>>;
@group(0) @binding(10)
var<storage, read_write> wavefront_queue: array<atomic<u32>>;
@group(0) @binding(11)
var<uniform> scene: SceneUniform;

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
const QUEUE_STATE_WORDS: u32 = 24u;
const RAY_WORDS: u32 = 36u;
const SAMPLE_STATE_OFFSET: u32 = QUEUE_STATE_WORDS;
const SAMPLE_STATE_WORDS: u32 = 16u;

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

fn load_ray_samples(pixel_index: u32) -> RaySamples {
    return RaySamples(
        vec4<f32>(
            bitcast<f32>(atomicLoad(&wavefront_queue[sample_state_word(pixel_index, 8u)])),
            bitcast<f32>(atomicLoad(&wavefront_queue[sample_state_word(pixel_index, 9u)])),
            bitcast<f32>(atomicLoad(&wavefront_queue[sample_state_word(pixel_index, 10u)])),
            bitcast<f32>(atomicLoad(&wavefront_queue[sample_state_word(pixel_index, 11u)])),
        ),
        vec4<f32>(
            bitcast<f32>(atomicLoad(&wavefront_queue[sample_state_word(pixel_index, 12u)])),
            bitcast<f32>(atomicLoad(&wavefront_queue[sample_state_word(pixel_index, 13u)])),
            bitcast<f32>(atomicLoad(&wavefront_queue[sample_state_word(pixel_index, 14u)])),
            bitcast<f32>(atomicLoad(&wavefront_queue[sample_state_word(pixel_index, 15u)])),
        ),
    );
}

fn store_ray_samples(pixel_index: u32, samples: RaySamples) {
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 8u)], bitcast<u32>(samples.direct.x));
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 9u)], bitcast<u32>(samples.direct.y));
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 10u)], bitcast<u32>(samples.direct.z));
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 11u)], bitcast<u32>(samples.direct.w));
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 12u)], bitcast<u32>(samples.indirect.x));
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 13u)], bitcast<u32>(samples.indirect.y));
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 14u)], bitcast<u32>(samples.indirect.z));
    atomicStore(&wavefront_queue[sample_state_word(pixel_index, 15u)], bitcast<u32>(samples.indirect.w));
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
const SHADOW_ORIGIN_WORD: u32 = 0u;
const SHADOW_DIRECTION_WORD: u32 = 4u;
const SHADOW_MAX_T_WORD: u32 = 8u;
const SHADOW_DIRECT_WORD: u32 = 12u;
const SHADOW_PIXEL_INDEX_WORD: u32 = 16u;

fn shadow_ray_word(index: u32, word: u32) -> u32 {
    return shadow_data_offset() + index * SHADOW_WORDS + word;
}

fn append_shadow_ray(pixel_index: u32, origin: vec3<f32>, direction: vec3<f32>, t: f32, direct: vec3<f32>) {
    let index = atomicAdd(&wavefront_queue[SHADOW_COUNT], 1u);
    if (index < pixel_count()) {
        atomicStore(&wavefront_queue[shadow_ray_word(index, SHADOW_ORIGIN_WORD)], bitcast<u32>(origin.x));
        atomicStore(&wavefront_queue[shadow_ray_word(index, SHADOW_ORIGIN_WORD + 1u)], bitcast<u32>(origin.y));
        atomicStore(&wavefront_queue[shadow_ray_word(index, SHADOW_ORIGIN_WORD + 2u)], bitcast<u32>(origin.z));
        atomicStore(&wavefront_queue[shadow_ray_word(index, SHADOW_ORIGIN_WORD + 3u)], 0u);
        atomicStore(&wavefront_queue[shadow_ray_word(index, SHADOW_DIRECTION_WORD)], bitcast<u32>(direction.x));
        atomicStore(&wavefront_queue[shadow_ray_word(index, SHADOW_DIRECTION_WORD + 1u)], bitcast<u32>(direction.y));
        atomicStore(&wavefront_queue[shadow_ray_word(index, SHADOW_DIRECTION_WORD + 2u)], bitcast<u32>(direction.z));
        atomicStore(&wavefront_queue[shadow_ray_word(index, SHADOW_DIRECTION_WORD + 3u)], 0u);
        atomicStore(&wavefront_queue[shadow_ray_word(index, SHADOW_MAX_T_WORD)], bitcast<u32>(t));
        atomicStore(&wavefront_queue[shadow_ray_word(index, SHADOW_DIRECT_WORD)], bitcast<u32>(direct.x));
        atomicStore(&wavefront_queue[shadow_ray_word(index, SHADOW_DIRECT_WORD + 1u)], bitcast<u32>(direct.y));
        atomicStore(&wavefront_queue[shadow_ray_word(index, SHADOW_DIRECT_WORD + 2u)], bitcast<u32>(direct.z));
        atomicStore(&wavefront_queue[shadow_ray_word(index, SHADOW_DIRECT_WORD + 3u)], 0u);
        atomicStore(&wavefront_queue[shadow_ray_word(index, SHADOW_PIXEL_INDEX_WORD)], pixel_index);
    } else {
        atomicStore(&wavefront_queue[SHADOW_OVERFLOW], 1u);
    }
}

fn load_shadow_pixel(index: u32) -> u32 {
    return atomicLoad(&wavefront_queue[shadow_ray_word(index, SHADOW_PIXEL_INDEX_WORD)]);
}

fn load_shadow_vec3(index: u32, word: u32) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(atomicLoad(&wavefront_queue[shadow_ray_word(index, word)])),
        bitcast<f32>(atomicLoad(&wavefront_queue[shadow_ray_word(index, word + 1u)])),
        bitcast<f32>(atomicLoad(&wavefront_queue[shadow_ray_word(index, word + 2u)])),
    );
}

fn load_shadow_t(index: u32) -> f32 {
    return bitcast<f32>(atomicLoad(&wavefront_queue[shadow_ray_word(index, SHADOW_MAX_T_WORD)]));
}

fn load_shadow_direct(index: u32) -> vec3<f32> {
    return load_shadow_vec3(index, SHADOW_DIRECT_WORD);
}

fn load_shadow_origin(index: u32) -> vec3<f32> {
    return load_shadow_vec3(index, SHADOW_ORIGIN_WORD);
}

fn load_shadow_direction(index: u32) -> vec3<f32> {
    return load_shadow_vec3(index, SHADOW_DIRECTION_WORD);
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
        vec4<f32>(
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 12u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 13u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 14u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 15u])),
        ),
        vec4<f32>(
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 16u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 17u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 18u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 19u])),
        ),
        vec4<f32>(
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 20u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 21u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 22u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 23u])),
        ),
        vec4<f32>(
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 24u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 25u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 26u])),
            bitcast<f32>(atomicLoad(&wavefront_queue[base + 27u])),
        ),
        atomicLoad(&wavefront_queue[base + 28u]),
        atomicLoad(&wavefront_queue[base + 29u]),
        bitcast<f32>(atomicLoad(&wavefront_queue[base + 30u])),
        bitcast<f32>(atomicLoad(&wavefront_queue[base + 31u])),
        bitcast<f32>(atomicLoad(&wavefront_queue[base + 32u])),
        vec3<u32>(
            atomicLoad(&wavefront_queue[base + 33u]),
            atomicLoad(&wavefront_queue[base + 34u]),
            atomicLoad(&wavefront_queue[base + 35u]),
        ),
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
    atomicStore(&wavefront_queue[base + 12u], bitcast<u32>(ray.prev_position.x));
    atomicStore(&wavefront_queue[base + 13u], bitcast<u32>(ray.prev_position.y));
    atomicStore(&wavefront_queue[base + 14u], bitcast<u32>(ray.prev_position.z));
    atomicStore(&wavefront_queue[base + 15u], bitcast<u32>(ray.prev_position.w));
    atomicStore(&wavefront_queue[base + 16u], bitcast<u32>(ray.prev_position_error.x));
    atomicStore(&wavefront_queue[base + 17u], bitcast<u32>(ray.prev_position_error.y));
    atomicStore(&wavefront_queue[base + 18u], bitcast<u32>(ray.prev_position_error.z));
    atomicStore(&wavefront_queue[base + 19u], bitcast<u32>(ray.prev_position_error.w));
    atomicStore(&wavefront_queue[base + 20u], bitcast<u32>(ray.prev_geometric_normal.x));
    atomicStore(&wavefront_queue[base + 21u], bitcast<u32>(ray.prev_geometric_normal.y));
    atomicStore(&wavefront_queue[base + 22u], bitcast<u32>(ray.prev_geometric_normal.z));
    atomicStore(&wavefront_queue[base + 23u], bitcast<u32>(ray.prev_geometric_normal.w));
    atomicStore(&wavefront_queue[base + 24u], bitcast<u32>(ray.prev_shading_normal.x));
    atomicStore(&wavefront_queue[base + 25u], bitcast<u32>(ray.prev_shading_normal.y));
    atomicStore(&wavefront_queue[base + 26u], bitcast<u32>(ray.prev_shading_normal.z));
    atomicStore(&wavefront_queue[base + 27u], bitcast<u32>(ray.prev_shading_normal.w));
    atomicStore(&wavefront_queue[base + 28u], ray.pixel_index);
    atomicStore(&wavefront_queue[base + 29u], ray.depth);
    atomicStore(&wavefront_queue[base + 30u], bitcast<u32>(ray.inv_w_u));
    atomicStore(&wavefront_queue[base + 31u], bitcast<u32>(ray.inv_w_l));
    atomicStore(&wavefront_queue[base + 32u], bitcast<u32>(ray.prev_pdf));
    atomicStore(&wavefront_queue[base + 33u], ray._padding.x);
    atomicStore(&wavefront_queue[base + 34u], ray._padding.y);
    atomicStore(&wavefront_queue[base + 35u], ray._padding.z);
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
    let base = scene.area_light_offset_words + index * 12u;
    return vec4<f32>(
        bitcast<f32>(scene_data[base + 2u]),
        bitcast<f32>(scene_data[base + 3u]),
        bitcast<f32>(scene_data[base + 4u]),
        0.0,
    );
}

fn load_area_word(index: u32, word: u32) -> u32 {
    return scene_data[scene.area_light_offset_words + index * 12u + word];
}

fn load_area_instance(index: u32) -> u32 {
    return load_area_word(index, 0u);
}

fn load_area_total(index: u32) -> f32 {
    return bitcast<f32>(load_area_word(index, 6u));
}

fn load_area_primitive(index: u32) -> u32 {
    return load_area_word(index, 7u);
}

fn load_area_two_sided(index: u32) -> bool {
    return load_area_word(index, 1u) != 0u;
}

fn pixel_count() -> u32 {
    return viewport.width * viewport.height;
}

fn load_material_kind(index: u32) -> u32 {
    return scene_data[scene.material_offset_words + index * 4u];
}

fn load_point_light(index: u32) -> PointLight {
    let base = scene.point_light_offset_words + index * 8u;
    return PointLight(
        vec4<f32>(
            bitcast<f32>(scene_data[base]),
            bitcast<f32>(scene_data[base + 1u]),
            bitcast<f32>(scene_data[base + 2u]),
            bitcast<f32>(scene_data[base + 3u]),
        ),
        vec4<f32>(
            bitcast<f32>(scene_data[base + 4u]),
            bitcast<f32>(scene_data[base + 5u]),
            bitcast<f32>(scene_data[base + 6u]),
            bitcast<f32>(scene_data[base + 7u]),
        ),
    );
}

fn load_light_kind(index: u32) -> u32 {
    return scene_data[scene.light_record_offset_words + index * 4u];
}

fn load_light_payload(index: u32) -> u32 {
    return scene_data[scene.light_record_offset_words + index * 4u + 1u];
}

fn uniform_light_pmf_for_handle(light_handle: u32) -> f32 {
    if (light_handle >= scene.light_count || scene.light_leaf_offset == 0xffffffffu) {
        return 0.0;
    }
    if (scene_data[scene.light_leaf_offset + light_handle] == 0xffffffffu) {
        return 0.0;
    }
    var count = 0u;
    for (var handle = 0u; handle < scene.light_leaf_count; handle++) {
        if (scene_data[scene.light_leaf_offset + handle] != 0xffffffffu) {
            count = count + 1u;
        }
    }
    if (count == 0u) {
        return 0.0;
    }
    return 1.0 / f32(count);
}

fn hash_u32(value: u32) -> u32 {
    var h = value;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

fn random01(pixel_index: u32, dimension: u32, depth: u32) -> f32 {
    let value = viewport.seed
        ^ (pixel_index * 0x9e3779b9u)
        ^ (viewport.sample_index * 0x85ebca6bu)
        ^ ((dimension + depth * 8u) * 0xc2b2ae35u);
    return f32(hash_u32(value) & 0x00ffffffu) / 16777216.0;
}

fn generate_ray_samples(pixel_index: u32, depth: u32) -> RaySamples {
    return RaySamples(
        vec4<f32>(
            random01(pixel_index, 2u, depth),
            random01(pixel_index, 3u, depth),
            random01(pixel_index, 4u, depth),
            0.0,
        ),
        vec4<f32>(
            random01(pixel_index, 5u, depth),
            random01(pixel_index, 6u, depth),
            random01(pixel_index, 7u, depth),
            random01(pixel_index, 8u, depth),
        ),
    );
}

fn sample_uniform_light(selector: f32) -> LightSelection {
    if (scene.light_leaf_offset == 0xffffffffu || scene.light_leaf_count == 0u) {
        return LightSelection(0xffffffffu, 0.0);
    }
    var count = 0u;
    for (var handle = 0u; handle < scene.light_leaf_count; handle++) {
        if (scene_data[scene.light_leaf_offset + handle] != 0xffffffffu) {
            count = count + 1u;
        }
    }
    if (count == 0u) {
        return LightSelection(0xffffffffu, 0.0);
    }
    let selected = min(u32(min(selector, 0.99999994) * f32(count)), count - 1u);
    var ordinal = 0u;
    for (var handle = 0u; handle < scene.light_leaf_count; handle++) {
        if (scene_data[scene.light_leaf_offset + handle] != 0xffffffffu) {
            if (ordinal == selected) {
                return LightSelection(handle, 1.0 / f32(count));
            }
            ordinal = ordinal + 1u;
        }
    }
    return LightSelection(0xffffffffu, 0.0);
}

fn light_bvh_word(node_index: u32, word: u32) -> u32 {
    return scene_data[scene.light_bvh_node_offset + node_index * 8u + word];
}

fn decode_light_bvh_node(node_index: u32) -> DecodedLightBVHNode {
    let word0 = light_bvh_word(node_index, 0u);
    let word1 = light_bvh_word(node_index, 1u);
    let word2 = light_bvh_word(node_index, 2u);
    let q_min = vec3<u32>(word0 & 0xffffu, word0 >> 16u, word1 & 0xffffu);
    let q_max = vec3<u32>(word1 >> 16u, word2 & 0xffffu, word2 >> 16u);
    let all_min = vec3<f32>(
        bitcast<f32>(scene_data[scene.light_sampler_data_offset]),
        bitcast<f32>(scene_data[scene.light_sampler_data_offset + 1u]),
        bitcast<f32>(scene_data[scene.light_sampler_data_offset + 2u]),
    );
    let all_max = vec3<f32>(
        bitcast<f32>(scene_data[scene.light_sampler_data_offset + 4u]),
        bitcast<f32>(scene_data[scene.light_sampler_data_offset + 5u]),
        bitcast<f32>(scene_data[scene.light_sampler_data_offset + 6u]),
    );
    let extent = all_max - all_min;
    let bounds_min = all_min + vec3<f32>(q_min) / 65535.0 * extent;
    let bounds_max = all_min + vec3<f32>(q_max) / 65535.0 * extent;
    let direction_word = light_bvh_word(node_index, 3u);
    let encoded = vec2<f32>(
        f32(direction_word & 0xffffu) / 65535.0 * 2.0 - 1.0,
        f32(direction_word >> 16u) / 65535.0 * 2.0 - 1.0,
    );
    var direction = vec3<f32>(encoded.x, encoded.y, 1.0 - abs(encoded.x) - abs(encoded.y));
    if (direction.z < 0.0) {
        direction = vec3<f32>(
            (1.0 - abs(direction.y)) * select(-1.0, 1.0, direction.x >= 0.0),
            (1.0 - abs(direction.x)) * select(-1.0, 1.0, direction.y >= 0.0),
            direction.z,
        );
    }
    direction = normalize(direction);
    let cosine_word = light_bvh_word(node_index, 5u);
    let cosine_o = f32(cosine_word & 0x7fffu) / 32767.0 * 2.0 - 1.0;
    let cosine_e = f32((cosine_word >> 15u) & 0x7fffu) / 32767.0 * 2.0 - 1.0;
    let payload_word = light_bvh_word(node_index, 6u);
    return DecodedLightBVHNode(
        bounds_min,
        bounds_max,
        direction,
        bitcast<f32>(light_bvh_word(node_index, 4u)),
        cosine_o,
        cosine_e,
        (cosine_word & 0x40000000u) != 0u,
        payload_word & 0x7fffffffu,
        (payload_word & 0x80000000u) != 0u,
    );
}

fn light_bvh_importance(node: DecodedLightBVHNode, p: vec3<f32>, n: vec3<f32>) -> f32 {
    let center = (node.bounds_min + node.bounds_max) * 0.5;
    let diagonal = node.bounds_max - node.bounds_min;
    let delta = p - center;
    var d2 = max(dot(delta, delta), length(diagonal) * 0.5);
    if (d2 <= 0.0) {
        return 0.0;
    }
    let radius = 0.5 * length(diagonal);
    // Match pbrt-v4 LightBounds::importance: wi points from the light
    // bound's center toward the reference point.
    let center_to_point = p - center;
    let center_distance_squared = dot(center_to_point, center_to_point);
    var cos_theta_b = -1.0;
    if (center_distance_squared > radius * radius && center_distance_squared > 0.0) {
        cos_theta_b = sqrt(max(0.0, 1.0 - radius * radius / center_distance_squared));
    }
    let wi = normalize(center_to_point);
    var cos_theta_w = dot(node.direction, wi);
    if (node.two_sided) {
        cos_theta_w = abs(cos_theta_w);
    }
    let sin_theta_w = sqrt(max(0.0, 1.0 - cos_theta_w * cos_theta_w));
    let sin_theta_o = sqrt(max(0.0, 1.0 - node.cos_theta_o * node.cos_theta_o));
    let sin_theta_b = sqrt(max(0.0, 1.0 - cos_theta_b * cos_theta_b));
    let cos_theta_x = cos_sub_clamped(
        sin_theta_w,
        cos_theta_w,
        sin_theta_o,
        node.cos_theta_o,
    );
    let sin_theta_x = sin_sub_clamped(
        sin_theta_w,
        cos_theta_w,
        sin_theta_o,
        node.cos_theta_o,
    );
    let cos_theta_p = cos_sub_clamped(sin_theta_x, cos_theta_x, sin_theta_b, cos_theta_b);
    if (cos_theta_p <= node.cos_theta_e) {
        return 0.0;
    }
    var importance = node.phi * cos_theta_p / d2;
    if (dot(n, n) != 0.0) {
        let cos_theta_i = abs(dot(wi, normalize(n)));
        let sin_theta_i = sqrt(max(0.0, 1.0 - cos_theta_i * cos_theta_i));
        importance = importance
            * cos_sub_clamped(sin_theta_i, cos_theta_i, sin_theta_b, cos_theta_b);
    }
    return max(importance, 0.0);
}

fn cos_sub_clamped(sin_a: f32, cos_a: f32, sin_b: f32, cos_b: f32) -> f32 {
    if (cos_a > cos_b) {
        return 1.0;
    }
    return cos_a * cos_b + sin_a * sin_b;
}

fn sin_sub_clamped(sin_a: f32, cos_a: f32, sin_b: f32, cos_b: f32) -> f32 {
    if (cos_a > cos_b) {
        return 0.0;
    }
    return sin_a * cos_b - cos_a * sin_b;
}

fn sample_light_bvh(selector: f32, p: vec3<f32>, n: vec3<f32>) -> LightSelection {
    if (scene.light_bvh_node_count == 0u || scene.light_leaf_count == 0u) {
        return LightSelection(0xffffffffu, 0.0);
    }
    var node_index = 0u;
    var pmf = 1.0;
    var u = min(selector, 0.99999994);
    for (var iteration = 0u; iteration < scene.light_bvh_node_count; iteration++) {
        let node = decode_light_bvh_node(node_index);
        if (node.is_leaf) {
            return LightSelection(node.payload, pmf);
        }
        let left = decode_light_bvh_node(node_index + 1u);
        let right = decode_light_bvh_node(node.payload);
        let left_weight = light_bvh_importance(left, p, n);
        let right_weight = light_bvh_importance(right, p, n);
        let total = left_weight + right_weight;
        if (total <= 0.0) {
            return LightSelection(0xffffffffu, 0.0);
        }
        let left_pmf = left_weight / total;
        if (u < left_pmf) {
            pmf = pmf * left_pmf;
            u = u / max(left_pmf, 1e-7);
            node_index = node_index + 1u;
        } else {
            pmf = pmf * (1.0 - left_pmf);
            u = (u - left_pmf) / max(1.0 - left_pmf, 1e-7);
            node_index = node.payload;
        }
        if (node_index >= scene.light_bvh_node_count) {
            return LightSelection(0xffffffffu, 0.0);
        }
    }
    return LightSelection(0xffffffffu, 0.0);
}

fn light_bvh_pmf_for_handle(light_handle: u32, p: vec3<f32>, n: vec3<f32>) -> f32 {
    if (light_handle >= scene.light_leaf_count) {
        return 0.0;
    }
    let leaf_index = scene_data[scene.light_leaf_offset + light_handle];
    if (leaf_index >= scene.light_bvh_node_count) {
        return 0.0;
    }
    var node_index = leaf_index;
    var pmf = 1.0;
    for (var iteration = 0u; iteration < scene.light_bvh_node_count; iteration++) {
        let parent = light_bvh_word(node_index, 7u);
        if (parent == 0xffffffffu) {
            return pmf;
        }
        if (parent >= scene.light_bvh_node_count) {
            return 0.0;
        }
        let parent_node = decode_light_bvh_node(parent);
        let left_child = parent + 1u;
        let left = decode_light_bvh_node(left_child);
        let right = decode_light_bvh_node(parent_node.payload);
        let left_weight = light_bvh_importance(left, p, n);
        let right_weight = light_bvh_importance(right, p, n);
        let total = left_weight + right_weight;
        if (total <= 0.0) {
            return 0.0;
        }
        if (node_index == left_child) {
            pmf = pmf * left_weight / total;
        } else if (node_index == parent_node.payload) {
            pmf = pmf * right_weight / total;
        } else {
            return 0.0;
        }
        node_index = parent;
    }
    return 0.0;
}

fn light_pmf_for_handle(light_handle: u32, p: vec3<f32>, n: vec3<f32>) -> f32 {
    if (scene.light_sampler_kind == LIGHT_SAMPLER_KIND_BVH) {
        return light_bvh_pmf_for_handle(light_handle, p, n);
    }
    return uniform_light_pmf_for_handle(light_handle);
}

fn sample_scene_light(selector: f32, p: vec3<f32>, n: vec3<f32>) -> LightSelection {
    if (scene.light_sampler_kind == LIGHT_SAMPLER_KIND_BVH) {
        return sample_light_bvh(selector, p, n);
    }
    return sample_uniform_light(selector);
}

fn gamma(n: f32) -> f32 {
    return (n * MACHINE_EPSILON) / (1.0 - n * MACHINE_EPSILON);
}

fn next_float_up(value: f32) -> f32 {
    if (value == 0.0) {
        return bitcast<f32>(1u);
    }
    let bits = bitcast<u32>(value);
    if (value < 0.0) {
        return bitcast<f32>(bits - 1u);
    }
    return bitcast<f32>(bits + 1u);
}

fn next_float_down(value: f32) -> f32 {
    if (value == 0.0) {
        return bitcast<f32>(0x80000001u);
    }
    let bits = bitcast<u32>(value);
    if (value > 0.0) {
        return bitcast<f32>(bits - 1u);
    }
    return bitcast<f32>(bits + 1u);
}

fn offset_ray_origin(position: vec3<f32>, error: vec3<f32>, normal: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    let offset = normal * dot(abs(normal), error);
    let signed_offset = select(-offset, offset, dot(direction, normal) >= 0.0);
    var result = position + signed_offset;
    if (signed_offset.x > 0.0) {
        result.x = next_float_up(result.x);
    } else if (signed_offset.x < 0.0) {
        result.x = next_float_down(result.x);
    }
    if (signed_offset.y > 0.0) {
        result.y = next_float_up(result.y);
    } else if (signed_offset.y < 0.0) {
        result.y = next_float_down(result.y);
    }
    if (signed_offset.z > 0.0) {
        result.z = next_float_up(result.z);
    } else if (signed_offset.z < 0.0) {
        result.z = next_float_down(result.z);
    }
    return result;
}

fn make_tangent(normal: vec3<f32>) -> vec3<f32> {
    if (abs(normal.x) > 0.1) {
        return normalize(cross(vec3<f32>(0.0, 1.0, 0.0), normal));
    }
    return normalize(cross(vec3<f32>(1.0, 0.0, 0.0), normal));
}
