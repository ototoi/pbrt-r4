enable wgpu_ray_query;

struct Camera {
    camera_from_raster: array<vec4<f32>, 4>,
    render_from_camera: array<vec4<f32>, 4>,
    viewport: vec4<f32>,
    bvh_info: vec4<u32>,
};

struct Primitive {
    first_vertex: u32,
    first_index: u32,
    material: u32,
    _padding: u32,
};

struct Material {
    reflectance: vec4<f32>,
};

struct Transform {
    rows: array<vec4<f32>, 4>,
};

struct Light {
    position: vec4<f32>,
    intensity: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;
@group(0) @binding(1)
var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2)
var acceleration: acceleration_structure;
@group(0) @binding(3)
var<storage, read> vertices: array<vec4<f32>>;
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

fn transform(rows: array<vec4<f32>, 4>, value: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        dot(rows[0], value),
        dot(rows[1], value),
        dot(rows[2], value),
        dot(rows[3], value),
    );
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let width = u32(camera.viewport.z);
    let height = u32(camera.viewport.w);
    if (global_id.x >= width || global_id.y >= height) {
        return;
    }

    let pixel = vec2<f32>(global_id.xy) + vec2<f32>(0.5) + camera.viewport.xy;
    let camera_target = transform(
        camera.camera_from_raster,
        vec4<f32>(pixel, 0.0, 1.0),
    );
    let origin = transform(
        camera.render_from_camera,
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    ).xyz;
    let ray_target = transform(camera.render_from_camera, camera_target).xyz;
    let direction = normalize(ray_target - origin);

    var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var query: ray_query;
    rayQueryInitialize(
        &query,
        acceleration,
        RayDesc(0u, 0xFFu, 0.0001, 1.0e30, origin, direction),
    );
    rayQueryProceed(&query);
    let intersection = rayQueryGetCommittedIntersection(&query);
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        let primitive = primitives[intersection.instance_custom_data];
        let index_offset = primitive.first_index + intersection.primitive_index * 3u;
        let i0 = primitive.first_vertex + indices[index_offset];
        let i1 = primitive.first_vertex + indices[index_offset + 1u];
        let i2 = primitive.first_vertex + indices[index_offset + 2u];
        let barycentrics = vec3<f32>(
            1.0 - intersection.barycentrics.x - intersection.barycentrics.y,
            intersection.barycentrics.x,
            intersection.barycentrics.y,
        );
        let object_position = vertices[i0].xyz * barycentrics.x
            + vertices[i1].xyz * barycentrics.y
            + vertices[i2].xyz * barycentrics.z;
        let object_normal = normalize(cross(
            vertices[i1].xyz - vertices[i0].xyz,
            vertices[i2].xyz - vertices[i0].xyz,
        ));
        let transform_table = transforms[intersection.instance_custom_data];
        let position = transform(transform_table.rows, vec4<f32>(object_position, 1.0)).xyz;
        let normal = normalize(transform(transform_table.rows, vec4<f32>(object_normal, 0.0)).xyz);
        let reflectance = materials[primitive.material].reflectance.xyz;
        var radiance = vec3<f32>(0.0, 0.0, 0.0);
        for (var light_index = 0u; light_index < arrayLength(&lights); light_index++) {
            let light = lights[light_index];
            let to_light = light.position.xyz - position;
            let distance_squared = max(dot(to_light, to_light), 1.0e-8);
            let wi = normalize(to_light);
            let cosine = max(dot(normal, wi), 0.0);
            radiance += reflectance * light.intensity.xyz * cosine / distance_squared;
        }
        color = vec4<f32>(radiance, 1.0);
    }

    let output_index = global_id.y * width + global_id.x;
    output[output_index] = color;
}
