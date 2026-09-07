const RAY_T_MAX: f32 = 3.402823466e+38;
const MACHINE_EPSILON: f32 = 1.1920929e-7;
const PI: f32 = 3.141592653589793;
const MATERIAL_KIND_NORMAL: u32 = 0u;
const MATERIAL_KIND_UV: u32 = 1u;
const MATERIAL_KIND_DIFFUSE: u32 = 2u;
const MATERIAL_KIND_LAMBERT: u32 = 2u;
const MATERIAL_KIND_DIELECTRIC: u32 = 3u;
const MATERIAL_KIND_LAYERED: u32 = 4u;
const MATERIAL_KIND_THIN_DIELECTRIC: u32 = 5u;
const MATERIAL_KIND_CONDUCTOR: u32 = 6u;
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
    disable_wavelength_jitter: u32,
    mode: u32,
    _padding: u32,
};

struct FilmUniform {
    sensor_response: vec4<u32>,
    imaging_ratio: f32,
    max_sample_luminance: f32,
    _padding0: u32,
    _padding1: u32,
};

struct MaterialTableUniform {
    material_offset_words: u32, material_count: u32,
    scattering_model_offset_words: u32, scattering_model_count: u32,
    scattering_node_offset_words: u32, scattering_node_count: u32,
    scattering_child_offset_words: u32, scattering_child_count: u32,
    bssrdf_node_offset_words: u32, bssrdf_node_count: u32,
    debug_scattering_model: u32, scattering_reserved: u32,
    _reserved0: u32, _reserved1: u32, _reserved2: u32, _reserved3: u32,
    _reserved4: u32, _reserved5: u32,
};

struct LightTableUniform {
    light_record_offset_words: u32,
    light_count: u32,
    light_sampler_kind: u32,
    light_sampler_data_offset: u32,
    light_bvh_node_offset: u32,
    light_bvh_node_count: u32,
    light_leaf_offset: u32,
    light_leaf_count: u32,
    _reserved0: u32,
    _reserved1: u32,
    _reserved2: u32,
    _reserved3: u32,
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
    orientation_flags: u32,
    world_from_object: mat4x4<f32>,
    normal_from_object: mat4x4<f32>,
};

struct ScatteringModelRecord {
    surface_root: u32,
    bssrdf_root: u32,
    _padding: vec2<u32>,
};

struct MaterialRecord { kind_tag: u32, attribute_offset: u32, attribute_count: u32, scattering_model: u32, };
struct AttributeRef { kind: u32, index: u32, };

struct ScatteringNodeRecord {
    kind_tag: u32,
    event_flags: u32,
    attribute_offset: u32,
    child_offset: u32,
    child_count: u32,
    attribute_count: u32,
    _padding0: u32,
    _padding1: u32,
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

struct LightRecord {
    kind: u32,
    attribute_offset: u32,
    attribute_count: u32,
    sampling_model: u32,
};

struct LightSamplingModel {
    kind: u32,
    geometry_kind: u32,
    geometry_index: u32,
    distribution_offset_words: u32,
    distribution_count: u32,
    total_area: f32,
    flags: u32,
    reserved: u32,
};

struct TriangleDistributionEntry {
    primitive: u32,
    cdf: f32,
    area: f32,
    _reserved: u32,
};

struct AreaTriangleSelection {
    primitive: u32,
    area: f32,
    pmf: f32,
};

struct LayeredParams {
    thickness: f32,
    g: f32,
    max_depth: u32,
    n_samples: u32,
    albedo: vec4<f32>,
    two_sided: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
};

struct QueueState {
    count: atomic<u32>,
    capacity: u32,
    overflow: atomic<u32>,
    _padding: u32,
};

struct QueueCounters {
    current: QueueState,
    next: QueueState,
    shadow: QueueState,
    material: QueueState,
    hit_area: QueueState,
    escaped: QueueState,
};

struct RenderError {
    value: atomic<u32>,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

struct PixelSampleState {
    radiance: vec4<f32>,
    lambda: vec4<f32>,
    lambda_pdf: vec4<f32>,
    direct: vec4<f32>,
    indirect: vec4<f32>,
};

struct ShadowRayWorkItem {
    origin: vec4<f32>,
    direction: vec4<f32>,
    max_t: f32,
    _padding0: vec3<u32>,
    direct: vec4<f32>,
    pixel_index: u32,
    _padding1: vec3<u32>,
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
