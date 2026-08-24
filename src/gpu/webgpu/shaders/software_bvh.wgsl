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

fn ray_hits_box(origin: vec3<f32>, inverse_direction: vec3<f32>, bounds_min: vec3<f32>, bounds_max: vec3<f32>, closest: f32) -> bool {
    let t0 = (bounds_min - origin) * inverse_direction;
    let t1 = (bounds_max - origin) * inverse_direction;
    let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), 0.0));
    let far = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return near <= far && near <= closest;
}

fn ray_hits_triangle(origin: vec3<f32>, direction: vec3<f32>, p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> f32 {
    let edge1 = p1 - p0;
    let edge2 = p2 - p0;
    let pvec = cross(direction, edge2);
    let determinant = dot(edge1, pvec);
    if (abs(determinant) < 1.0e-7) {
        return -1.0;
    }
    let inverse_determinant = 1.0 / determinant;
    let tvec = origin - p0;
    let u = dot(tvec, pvec) * inverse_determinant;
    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }
    let qvec = cross(tvec, edge1);
    let v = dot(direction, qvec) * inverse_determinant;
    if (v < 0.0 || u + v > 1.0) {
        return -1.0;
    }
    let distance = dot(edge2, qvec) * inverse_determinant;
    if (distance > 0.0001) {
        return distance;
    }
    return -1.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let width = u32(camera.viewport.z);
    let height = u32(camera.viewport.w);
    if (global_id.x >= width || global_id.y >= height) {
        return;
    }

    let pixel = vec2<f32>(global_id.xy) + vec2<f32>(0.5) + camera.viewport.xy;
    let camera_target = transform(camera.camera_from_raster, vec4<f32>(pixel, 0.0, 1.0));
    let origin = transform(camera.render_from_camera, vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let ray_target = transform(camera.render_from_camera, camera_target).xyz;
    let direction = normalize(ray_target - origin);
    let inverse_direction = 1.0 / direction;

    var closest = 1.0e30;
    var hit_primitive = 0u;
    var hit_triangle = 0u;
    var hit_position = vec3<f32>(0.0);
    var stack: array<u32, 64>;
    var stack_size = 1u;
    stack[0] = 0u;

    loop {
        if (stack_size == 0u) {
            break;
        }
        stack_size -= 1u;
        let node_base = camera.bvh_info.y + stack[stack_size] * 12u;
        let node_bounds_min = vec3<f32>(
            bitcast<f32>(scene_data[node_base]),
            bitcast<f32>(scene_data[node_base + 1u]),
            bitcast<f32>(scene_data[node_base + 2u]),
        );
        let node_bounds_max = vec3<f32>(
            bitcast<f32>(scene_data[node_base + 4u]),
            bitcast<f32>(scene_data[node_base + 5u]),
            bitcast<f32>(scene_data[node_base + 6u]),
        );
        let node_first = scene_data[node_base + 8u];
        let node_count = scene_data[node_base + 9u];
        let node_flags = scene_data[node_base + 10u];
        if (!ray_hits_box(origin, inverse_direction, node_bounds_min, node_bounds_max, closest)) {
            continue;
        }
        if (node_flags == 1u) {
            for (var offset = 0u; offset < node_count; offset += 1u) {
                let reference_base = camera.bvh_info.x + (node_first + offset) * 2u;
                let primitive_index = scene_data[reference_base];
                let triangle_index = scene_data[reference_base + 1u];
                let primitive = primitives[primitive_index];
                let index_offset = primitive.first_index + triangle_index * 3u;
                let i0 = primitive.first_vertex + indices[index_offset];
                let i1 = primitive.first_vertex + indices[index_offset + 1u];
                let i2 = primitive.first_vertex + indices[index_offset + 2u];
                let object_p0 = vertices[i0].position.xyz;
                let object_p1 = vertices[i1].position.xyz;
                let object_p2 = vertices[i2].position.xyz;
                let transform_table = transforms[primitive_index];
                let p0 = transform(transform_table.rows, vec4<f32>(object_p0, 1.0)).xyz;
                let p1 = transform(transform_table.rows, vec4<f32>(object_p1, 1.0)).xyz;
                let p2 = transform(transform_table.rows, vec4<f32>(object_p2, 1.0)).xyz;
                let distance = ray_hits_triangle(origin, direction, p0, p1, p2);
                if (distance > 0.0 && distance < closest) {
                    closest = distance;
                    hit_primitive = primitive_index;
                    hit_triangle = triangle_index;
                    hit_position = origin + distance * direction;
                }
            }
        } else {
            stack[stack_size] = node_first;
            stack[stack_size + 1u] = node_first + 1u;
            stack_size += 2u;
        }
    }

    var color = vec3<f32>(0.0);
    if (closest < 1.0e30) {
        let primitive = primitives[hit_primitive];
        let index_offset = primitive.first_index + hit_triangle * 3u;
        let p0 = vertices[primitive.first_vertex + indices[index_offset]].position.xyz;
        let p1 = vertices[primitive.first_vertex + indices[index_offset + 1u]].position.xyz;
        let p2 = vertices[primitive.first_vertex + indices[index_offset + 2u]].position.xyz;
        let transform_table = transforms[hit_primitive];
        let position = hit_position;
        let normal = normalize(transform(transform_table.rows, vec4<f32>(cross(p1 - p0, p2 - p0), 0.0)).xyz);
        let material = materials[primitive.material];
        var reflectance = material.reflectance.xyz;
        if ((material.flags & 1u) != 0u) {
            let wp0 = transform(transform_table.rows, vec4<f32>(p0, 1.0)).xyz;
            let wp1 = transform(transform_table.rows, vec4<f32>(p1, 1.0)).xyz;
            let wp2 = transform(transform_table.rows, vec4<f32>(p2, 1.0)).xyz;
            let edge1 = wp1 - wp0;
            let edge2 = wp2 - wp0;
            let d = dot(edge1, edge1) * dot(edge2, edge2) - dot(edge1, edge2) * dot(edge1, edge2);
            let rel = hit_position - wp0;
            let beta = (dot(rel, edge2) * dot(edge1, edge1) - dot(rel, edge1) * dot(edge1, edge2)) / d;
            let gamma = (dot(rel, edge1) * dot(edge2, edge2) - dot(rel, edge2) * dot(edge1, edge2)) / d;
            let alpha = 1.0 - beta - gamma;
            let uv = vertices[primitive.first_vertex + indices[index_offset]].uv.xy * alpha
                + vertices[primitive.first_vertex + indices[index_offset + 1u]].uv.xy * beta
                + vertices[primitive.first_vertex + indices[index_offset + 2u]].uv.xy * gamma;
            reflectance = sample_texture(material.texture, uv);
        }
        for (var light_index = 0u; light_index < arrayLength(&lights); light_index += 1u) {
            let light = lights[light_index];
            let to_light = light.position.xyz - position;
            let distance_squared = max(dot(to_light, to_light), 1.0e-8);
            color += reflectance * light.intensity.xyz * max(dot(normal, normalize(to_light)), 0.0) / distance_squared;
        }
    }
    output[global_id.y * width + global_id.x] = vec4<f32>(color, 1.0);
}
