enable wgpu_ray_query;

const RAY_T_MAX: f32 = 3.402823466e+38;
const MATERIAL_KIND_NORMAL: u32 = 0u;
const MATERIAL_KIND_UV: u32 = 1u;

struct CameraUniform {
    camera_to_world: mat4x4<f32>,
    raster_to_camera: mat4x4<f32>,
    width: u32,
    height: u32,
    _padding: vec2<u32>,
};

struct Vertex {
    position: vec4<f32>,
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

struct Material {
    kind_tag: u32,
    _padding: array<u32, 3>,
};

struct RayWorkItem {
    origin: vec4<f32>,
    direction: vec4<f32>,
    pixel_index: u32,
    _padding: array<u32, 3>,
};

struct HitRecord {
    t: f32,
    hit: u32,
    instance_custom_data: u32,
    primitive_index: u32,
    barycentric: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var tlas: acceleration_structure;
@group(0) @binding(2)
var<storage, read> vertices: array<Vertex>;
@group(0) @binding(3)
var<storage, read> indices: array<u32>;
@group(0) @binding(4)
var<storage, read> geometries: array<Geometry>;
@group(0) @binding(5)
var<storage, read> instances: array<Instance>;
@group(0) @binding(6)
var<storage, read> materials: array<Material>;
@group(0) @binding(7)
var<storage, read_write> rays: array<RayWorkItem>;
@group(0) @binding(8)
var<storage, read_write> hits: array<HitRecord>;
@group(0) @binding(9)
var<storage, read_write> framebuffer: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn generate_primary_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= camera.width || global_id.y >= camera.height) {
        return;
    }
    let pixel_index = global_id.y * camera.width + global_id.x;
    let pixel = vec4<f32>(f32(global_id.x), f32(global_id.y), 1.0, 1.0);
    let camera_point = camera.raster_to_camera * pixel;
    let origin = (camera.camera_to_world * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let direction = normalize((camera.camera_to_world * vec4<f32>(camera_point.xyz, 0.0)).xyz);
    rays[pixel_index] = RayWorkItem(
        vec4<f32>(origin, 1.0),
        vec4<f32>(direction, 0.0),
        pixel_index,
        array<u32, 3>(0u, 0u, 0u),
    );
}

@compute @workgroup_size(8, 8, 1)
fn intersect_primary_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= camera.width || global_id.y >= camera.height) {
        return;
    }
    let pixel_index = global_id.y * camera.width + global_id.x;
    let ray = rays[pixel_index];
    var query: ray_query;
    rayQueryInitialize(
        &query,
        tlas,
        RayDesc(0u, 0xffu, 0.0, RAY_T_MAX, ray.origin.xyz, ray.direction.xyz),
    );
    while (rayQueryProceed(&query)) {
    }
    let intersection = rayQueryGetCommittedIntersection(&query);
    if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
        hits[pixel_index] = HitRecord(0.0, 0u, 0u, 0u, vec4<f32>(0.0));
    } else {
        hits[pixel_index] = HitRecord(
            intersection.t,
            1u,
            intersection.instance_custom_data,
            intersection.primitive_index,
            vec4<f32>(intersection.barycentrics, 0.0, 0.0),
        );
    }
}

@compute @workgroup_size(8, 8, 1)
fn shade_normal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= camera.width || global_id.y >= camera.height) {
        return;
    }
    let pixel_index = global_id.y * camera.width + global_id.x;
    let hit = hits[pixel_index];
    if (hit.hit == 0u) {
        framebuffer[pixel_index] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return;
    }

    let instance = instances[hit.instance_custom_data];
    let geometry = geometries[instance.geometry];
    let material = materials[instance.material];
    let first_index = geometry.index_offset + hit.primitive_index * 3u;
    let i0 = geometry.vertex_offset + indices[first_index];
    let i1 = geometry.vertex_offset + indices[first_index + 1u];
    let i2 = geometry.vertex_offset + indices[first_index + 2u];
    let p0 = (instance.world_from_object * vertices[i0].position).xyz;
    let p1 = (instance.world_from_object * vertices[i1].position).xyz;
    let p2 = (instance.world_from_object * vertices[i2].position).xyz;
    let normal = normalize(cross(p1 - p0, p2 - p0));

    if (material.kind_tag == MATERIAL_KIND_NORMAL) {
        framebuffer[pixel_index] = vec4<f32>(normal * 0.5 + vec3<f32>(0.5), 1.0);
    } else if (material.kind_tag == MATERIAL_KIND_UV) {
        framebuffer[pixel_index] = vec4<f32>(
            f32(hit.primitive_index & 1u),
            hit.barycentric.x,
            hit.barycentric.y,
            1.0,
        );
    } else {
        framebuffer[pixel_index] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
}
