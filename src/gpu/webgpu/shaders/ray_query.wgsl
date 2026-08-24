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
    texture: u32,
    flags: u32,
    _padding: vec2<u32>,
};

struct Vertex {
    position: vec4<f32>,
    uv: vec4<f32>,
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

fn transform(rows: array<vec4<f32>, 4>, value: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        dot(rows[0], value),
        dot(rows[1], value),
        dot(rows[2], value),
        dot(rows[3], value),
    );
}

fn sample_texture(texture_id: u32, uv: vec2<f32>) -> vec3<f32> {
    let texture_base = scene_data[2u] + texture_id * 8u;
    let image_id = scene_data[texture_base];
    let su = bitcast<f32>(scene_data[texture_base + 1u]);
    let sv = bitcast<f32>(scene_data[texture_base + 2u]);
    let du = bitcast<f32>(scene_data[texture_base + 3u]);
    let dv = bitcast<f32>(scene_data[texture_base + 4u]);
    let scale = bitcast<f32>(scene_data[texture_base + 5u]);
    let flags = scene_data[texture_base + 6u];
    var st = uv * vec2<f32>(su, sv) + vec2<f32>(du, dv);
    let swrap = (flags >> 1u) & 3u;
    let twrap = (flags >> 3u) & 3u;
    if ((swrap == 0u && (st.x < 0.0 || st.x > 1.0)) ||
        (twrap == 0u && (st.y < 0.0 || st.y > 1.0))) {
        return vec3<f32>(0.0);
    }
    st.x = select(fract(st.x), clamp(st.x, 0.0, 1.0), swrap == 1u);
    st.y = select(fract(st.y), clamp(st.y, 0.0, 1.0), twrap == 1u);
    let image_base = scene_data[0u] + image_id * 8u;
    let width = scene_data[image_base];
    let height = scene_data[image_base + 1u];
    let channels = scene_data[image_base + 2u];
    let texel_base = scene_data[image_base + 3u];
    let x = min(u32(st.x * f32(width)), width - 1u);
    let y = min(u32(st.y * f32(height)), height - 1u);
    let base = texel_base + (y * width + x) * channels;
    let r = bitcast<f32>(scene_data[base]);
    let g = select(r, bitcast<f32>(scene_data[base + 1u]), channels > 1u);
    let b = select(r, bitcast<f32>(scene_data[base + 2u]), channels > 2u);
    var value = vec3<f32>(r, g, b);
    if ((flags & 1u) != 0u) {
        value = vec3<f32>(1.0) - value;
    }
    if (((flags >> 6u) & 3u) == 0u) {
        value = clamp(value, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return value * scale;
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
        let object_position = vertices[i0].position.xyz * barycentrics.x
            + vertices[i1].position.xyz * barycentrics.y
            + vertices[i2].position.xyz * barycentrics.z;
        let object_normal = normalize(cross(
            vertices[i1].position.xyz - vertices[i0].position.xyz,
            vertices[i2].position.xyz - vertices[i0].position.xyz,
        ));
        let transform_table = transforms[intersection.instance_custom_data];
        let position = transform(transform_table.rows, vec4<f32>(object_position, 1.0)).xyz;
        let normal = normalize(transform(transform_table.rows, vec4<f32>(object_normal, 0.0)).xyz);
        let material = materials[primitive.material];
        var reflectance = material.reflectance.xyz;
        if ((material.flags & 1u) != 0u) {
            let uv = vertices[i0].uv.xy * barycentrics.x
                + vertices[i1].uv.xy * barycentrics.y
                + vertices[i2].uv.xy * barycentrics.z;
            reflectance = sample_texture(material.texture, uv);
        }
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
