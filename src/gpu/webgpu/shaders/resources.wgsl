enable wgpu_ray_query;

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
var<uniform> film_params: FilmUniform;
@group(0) @binding(8)
var<storage, read_write> surfaces: array<SurfaceWorkItem>;
@group(0) @binding(9)
var<storage, read_write> framebuffer: array<vec4<f32>>;
@group(0) @binding(10)
var<storage, read_write> queue_counters: QueueCounters;
@group(0) @binding(11)
var<storage, read_write> render_error: RenderError;
@group(0) @binding(12)
var<storage, read_write> pixel_sample_states: array<PixelSampleState>;
@group(0) @binding(13)
var<storage, read_write> current_rays: array<RayWorkItem>;
@group(0) @binding(14)
var<storage, read_write> next_rays: array<RayWorkItem>;
@group(0) @binding(15)
var<storage, read_write> shadow_rays: array<ShadowRayWorkItem>;
@group(0) @binding(16)
var<storage, read_write> material_ray_indices: array<u32>;
@group(0) @binding(17)
var<storage, read_write> hit_area_ray_indices: array<u32>;
@group(0) @binding(18)
var<storage, read_write> escaped_ray_indices: array<u32>;
@group(0) @binding(19)
var<uniform> material_table: MaterialTableUniform;
@group(0) @binding(20)
var<uniform> light_table: LightTableUniform;
@group(0) @binding(21)
var<storage, read> materials: array<MaterialRecord>;
@group(0) @binding(22)
var<storage, read> attribute_refs: array<AttributeRef>;
@group(0) @binding(23)
var<storage, read> scalar_attributes: array<f32>;
@group(0) @binding(24)
var<storage, read> scattering_models: array<ScatteringModelRecord>;
@group(0) @binding(25)
var<storage, read> scattering_nodes: array<ScatteringNodeRecord>;
@group(0) @binding(26)
var<storage, read> scattering_children: array<u32>;
@group(0) @binding(28)
var<storage, read> light_records: array<LightRecord>;
@group(0) @binding(29)
var<storage, read> light_sampling_models: array<LightSamplingModel>;
@group(0) @binding(30)
var<storage, read> triangle_distributions: array<TriangleDistributionEntry>;
@group(0) @binding(31)
var<storage, read> light_bvh_header: array<u32>;
@group(0) @binding(32)
var<storage, read> light_bvh_nodes: array<u32>;
@group(0) @binding(33)
var<storage, read> light_bvh_leaves: array<u32>;
@group(0) @binding(34)
var<storage, read> spectra: array<DenseSpectrum>;
@group(0) @binding(36)
var<storage, read> light_positions: array<vec4<f32>>;
