const RAY_T_MAX: f32 = 3.402823466e+38;
const MACHINE_EPSILON: f32 = 1.1920929e-7;
const PI: f32 = 3.141592653589793;
const MATERIAL_KIND_NORMAL: u32 = 0u;
const MATERIAL_KIND_UV: u32 = 1u;
const MATERIAL_KIND_DIFFUSE: u32 = 2u;
const MATERIAL_KIND_LAMBERT: u32 = 2u;
const MATERIAL_KIND_DIELECTRIC: u32 = 3u;
const MATERIAL_KIND_THIN_DIELECTRIC: u32 = 5u;
const MATERIAL_KIND_UNUSED_CONDUCTOR: u32 = 6u;
const MATERIAL_KIND_CONDUCTOR_ETA_K: u32 = 11u;
const MATERIAL_KIND_CONDUCTOR_REFLECTANCE: u32 = 12u;
const MATERIAL_KIND_MIX: u32 = 7u;
const MATERIAL_KIND_COATED_DIFFUSE: u32 = 8u;
const MATERIAL_KIND_COATED_CONDUCTOR: u32 = 9u;
const MATERIAL_KIND_MEASURED: u32 = 10u;
const MATERIAL_KIND_DIFFUSE_TRANSMISSION: u32 = 13u;
const MATERIAL_KIND_ALPHA_MASK: u32 = 14u;
const AREA_LIGHT_FLAG_TWO_SIDED: u32 = 1u;
const AREA_LIGHT_FLAG_ZERO_ALPHA_SAMPLE_ONLY: u32 = 2u;
const INSTANCE_ORIENTATION_FLAG_REVERSED: u32 = 1u;
const INSTANCE_ORIENTATION_FLAG_TRANSFORM_SWAPS_HANDEDNESS: u32 = 2u;
const PORTAL_CANDIDATE_INVALID: u32 = 0u;
const PORTAL_CANDIDATE_SAMPLED: u32 = 1u;
const PORTAL_CANDIDATE_SELECTED: u32 = 2u;

struct TriangleSurfaceData {
    position: vec3<f32>,
    uv: vec2<f32>,
    geometric_normal: vec3<f32>,
    valid: u32,
};

struct AttributesEvalWorkItem {
    material_node: u32,
    child_work_item0: u32,
    child_work_item1: u32,
    bxdf_kind: u32,
    selected_child_work_item: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    values: array<vec4<f32>, 10>,
};

struct MaterialRoot {
    node_offset: u32,
    node_count: u32,
};

struct MaterialNode {
    kind: u32,
    attribute_offset: u32,
    attribute_count: u32,
    parent: u32,
    parent_slot: u32,
    child0: u32,
    child1: u32,
    _padding: u32,
};

struct MeasuredBsdfRecord {
    ndf: u32,
    sigma: u32,
    vndf: u32,
    luminance: u32,
    spectra: u32,
    isotropic: u32,
    _padding0: u32,
    _padding1: u32,
};

struct MeasuredTableRecord {
    size: vec2<u32>,
    parameter_count: u32,
    _padding0: u32,
    parameter_sizes: vec3<u32>,
    _padding1: u32,
    parameter_strides: vec3<u32>,
    _padding2: u32,
    parameter_value_offsets: vec3<u32>,
    _padding3: u32,
    data_offset: u32,
    marginal_cdf_offset: u32,
    conditional_cdf_offset: u32,
    _padding4: u32,
};

struct MeasuredPlSample {
    p: vec2<f32>,
    pdf: f32,
};

struct MeasuredLookup {
    offset: u32,
    weights: array<f32, 6>,
};

struct MeasuredBxdfSample {
    f: vec4<f32>,
    wi: vec3<f32>,
    pdf: f32,
    valid: u32,
};

struct TextureEvalResult {
    material_node: u32,
    attribute_ordinal: u32,
    texture_root: u32,
    valid: u32,
    value: vec4<f32>,
};

struct LayeredParams {
    thickness: f32,
    albedo: vec4<f32>,
    g: f32,
    max_depth: f32,
    n_samples: f32,
};

struct DielectricInterfaceSample {
    f: vec4<f32>,
    wi: vec3<f32>,
    pdf: f32,
    etap: f32,
    valid: u32,
    transmission: u32,
    specular: u32,
};
const LIGHT_KIND_AREA: u32 = 1u;
const LIGHT_KIND_POINT: u32 = 0u;
const LIGHT_KIND_SPOT: u32 = 2u;
const LIGHT_KIND_DISTANT: u32 = 3u;
const LIGHT_KIND_UNIFORM_INFINITE: u32 = 4u;
const LIGHT_KIND_IMAGE_INFINITE: u32 = 5u;
const LIGHT_KIND_PORTAL_IMAGE_INFINITE: u32 = 6u;
const LIGHT_SAMPLER_KIND_BVH: u32 = 1u;
const TEXTURE_OPERATION_IMAGE: u32 = 0u;
const TEXTURE_OPERATION_CONSTANT: u32 = 1u;
const TEXTURE_OPERATION_SCALE: u32 = 2u;
const TEXTURE_OPERATION_MIX: u32 = 3u;
const TEXTURE_OPERATION_CHECKERBOARD: u32 = 4u;
const TEXTURE_OPERATION_DIRECTION_MIX: u32 = 5u;
const TEXTURE_OPERATION_DOTS: u32 = 6u;
const TEXTURE_OPERATION_FBM: u32 = 7u;
const TEXTURE_OPERATION_WRINKLED: u32 = 8u;
const TEXTURE_OPERATION_WINDY: u32 = 9u;
const TEXTURE_OPERATION_BILERP: u32 = 10u;
const TEXTURE_OPERATION_MARBLE: u32 = 11u;
const TEXTURE_PROGRAM_CAPACITY: u32 = 256u;

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

struct SamplerUniform {
    kind: u32,
    randomization: u32,
    table_width: u32,
    dimension_count: u32,
    samples_per_pixel: u32,
    seed: u32,
    _padding: vec2<u32>,
    variant_words: array<vec4<u32>, 2>,
};

struct FilmUniform {
    sensor_response: vec4<u32>,
    imaging_ratio: f32,
    max_sample_luminance: f32,
    mode: u32,
    _padding: u32,
};

struct MaterialTableUniform {
    material_offset_words: u32, material_node_count: u32,
    debug_material_kind: u32,
    attributes_eval_stride: u32, texture_eval_stride: u32,
    measured_texture_base: u32, measured_texture_width: u32,
    measured_texture_height: u32, measured_texture_count: u32,
    _reserved6: u32, _reserved7: u32,
    _reserved8: u32, _reserved9: u32, _reserved10: u32, _reserved11: u32,
    _reserved12: u32, _reserved13: u32, _reserved14: u32,
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
    finite_light_count: u32,
    infinite_light_count: u32,
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
    material_root: u32,
    area_light: u32,
    orientation_flags: u32,
    world_from_object: mat4x4<f32>,
    normal_from_object: mat4x4<f32>,
};

struct TextureNodeRecord {
    kind: u32,
    first_child: u32,
    child_count: u32,
    implementation_hash: u32,
    swrap_mode: u32,
    twrap_mode: u32,
    color_space: u32,
    texture_index: u32,
    operation: u32,
    mapping_kind: u32,
    sampler: u32,
    operation_pad: u32,
    constant_value: vec4<f32>,
    mapping: mat4x4<f32>,
};
struct AttributeRef { kind: u32, index: u32, };
struct TextureRootRecord {
    texture_node: u32,
    instruction_count: u32,
    result: u32,
    spectrum_type: u32,
};

struct DenseSpectrum {
    samples: array<f32, 471>,
    flags: u32,
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
    prev_specular: u32,
    _padding0: u32,
    _padding1: u32,
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
    tangent: vec4<f32>,
    uv: vec2<f32>,
    _uv_padding: vec2<f32>,
    material_root: u32,
    flags: u32,
    attributes_eval_work_item: u32,
    _padding: u32,
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
    direction_index: u32,
    distribution_offset_words: u32,
    distribution_count: u32,
    total_area: f32,
    flags: u32,
    world_to_light0: vec4<f32>,
    world_to_light1: vec4<f32>,
    world_to_light2: vec4<f32>,
};

struct PortalImageInfiniteRecord {
    portal0: vec4<f32>,
    portal1: vec4<f32>,
    portal2: vec4<f32>,
    portal3: vec4<f32>,
    world_to_portal0: vec4<f32>,
    world_to_portal1: vec4<f32>,
    world_to_portal2: vec4<f32>,
    distribution_offset: u32,
    width: u32,
    height: u32,
    reserved: u32,
};

struct PortalDistributionTexel {
    function: f32,
    summed_area: f32,
};

struct PortalLightCandidate {
    position_uv: vec4<f32>,
    sample_direction_pdf: vec4<f32>,
    state: u32,
    light_index: u32,
    _padding0: u32,
    _padding1: u32,
};

struct DirectLightSample {
    direction_pdf: vec4<f32>,
    radiance: vec4<f32>,
    position: vec3<f32>,
    light_kind: u32,
    position_error: vec3<f32>,
    use_mis: u32,
    normal: vec3<f32>,
    valid: u32,
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
    u_remapped: f32,
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
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    direct: vec4<f32>,
    pixel_index: u32,
    _padding3: u32,
    _padding4: u32,
    _padding5: u32,
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
