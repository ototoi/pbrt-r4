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
    alpha: u32,
    shadow_alpha: u32,
    _flags: u32,
    _padding: vec2<u32>,
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
    kind: u32,
    _padding: array<u32, 3>,
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

fn image_texel(image_base: u32, x: i32, y: i32, swrap: u32, twrap: u32) -> vec4<f32> {
    let width = scene_data[image_base];
    let height = scene_data[image_base + 1u];
    let channels = scene_data[image_base + 2u];
    let texel_base = scene_data[image_base + 5u];
    var ix = x;
    var iy = y;
    if (swrap == 0u && (ix < 0 || ix >= i32(width))) {
        return vec4<f32>(0.0);
    }
    if (twrap == 0u && (iy < 0 || iy >= i32(height))) {
        return vec4<f32>(0.0);
    }
    if (swrap == 1u) {
        ix = clamp(ix, 0, i32(width) - 1);
    } else {
        ix = ((ix % i32(width)) + i32(width)) % i32(width);
    }
    if (twrap == 1u) {
        iy = clamp(iy, 0, i32(height) - 1);
    } else {
        iy = ((iy % i32(height)) + i32(height)) % i32(height);
    }
    let ix_u = u32(ix);
    let iy_u = u32(iy);
    let base = texel_base + (iy_u * width + ix_u) * channels;
    let r = bitcast<f32>(scene_data[base]);
    var g = r;
    var b = r;
    var a = 1.0;
    if (channels > 1u) {
        g = bitcast<f32>(scene_data[base + 1u]);
    }
    if (channels > 2u) {
        b = bitcast<f32>(scene_data[base + 2u]);
    }
    if (channels > 3u) {
        a = bitcast<f32>(scene_data[base + 3u]);
    }
    return vec4<f32>(r, g, b, a);
}

fn sample_texture(texture_id: u32, uv: vec2<f32>) -> vec3<f32> {
    let texture_base = scene_data[6u] + texture_id * 8u;
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
    let coordinate = st * vec2<f32>(f32(width), f32(height)) - vec2<f32>(0.5);
    let x0 = i32(floor(coordinate.x));
    let y0 = i32(floor(coordinate.y));
    let tx = fract(coordinate.x);
    let ty = fract(coordinate.y);
    let p00 = image_texel(image_base, x0, y0, swrap, twrap);
    var value = p00;
    if (((flags >> 5u) & 1u) != 0u) {
        let p10 = image_texel(image_base, x0 + 1, y0, swrap, twrap);
        let p01 = image_texel(image_base, x0, y0 + 1, swrap, twrap);
        let p11 = image_texel(image_base, x0 + 1, y0 + 1, swrap, twrap);
        value = p00 * (1.0 - tx) * (1.0 - ty)
            + p10 * tx * (1.0 - ty)
            + p01 * (1.0 - tx) * ty
            + p11 * tx * ty;
    }
    if ((flags & 1u) != 0u) {
        value = vec4<f32>(vec3<f32>(1.0) - value.xyz, value.w);
    }
    if (((flags >> 6u) & 3u) == 0u) {
        value = vec4<f32>(clamp(value.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), value.w);
    }
    return value.xyz * scale;
}

fn sample_float_texture(texture_id: u32, uv: vec2<f32>) -> f32 {
    let texture_base = scene_data[4u] + texture_id * 8u;
    let flags = scene_data[texture_base + 6u];
    if (((flags >> 7u) & 1u) != 0u) {
        return bitcast<f32>(scene_data[texture_base + 7u]);
    }
    let image_id = scene_data[texture_base];
    let su = bitcast<f32>(scene_data[texture_base + 1u]);
    let sv = bitcast<f32>(scene_data[texture_base + 2u]);
    let du = bitcast<f32>(scene_data[texture_base + 3u]);
    let dv = bitcast<f32>(scene_data[texture_base + 4u]);
    let scale = bitcast<f32>(scene_data[texture_base + 5u]);
    let swrap = (flags >> 1u) & 3u;
    let twrap = (flags >> 3u) & 3u;
    var st = uv * vec2<f32>(su, sv) + vec2<f32>(du, dv);
    if ((swrap == 0u && (st.x < 0.0 || st.x > 1.0)) ||
        (twrap == 0u && (st.y < 0.0 || st.y > 1.0))) {
        return 0.0;
    }
    st.x = select(fract(st.x), clamp(st.x, 0.0, 1.0), swrap == 1u);
    st.y = select(fract(st.y), clamp(st.y, 0.0, 1.0), twrap == 1u);
    let image_base = scene_data[0u] + image_id * 8u;
    let width = scene_data[image_base];
    let height = scene_data[image_base + 1u];
    let coordinate = st * vec2<f32>(f32(width), f32(height)) - vec2<f32>(0.5);
    let x0 = i32(floor(coordinate.x));
    let y0 = i32(floor(coordinate.y));
    let tx = fract(coordinate.x);
    let ty = fract(coordinate.y);
    let p00 = image_texel(image_base, x0, y0, swrap, twrap);
    var value = p00;
    if (((flags >> 5u) & 1u) != 0u) {
        let p10 = image_texel(image_base, x0 + 1, y0, swrap, twrap);
        let p01 = image_texel(image_base, x0, y0 + 1, swrap, twrap);
        let p11 = image_texel(image_base, x0 + 1, y0 + 1, swrap, twrap);
        value = p00 * (1.0 - tx) * (1.0 - ty)
            + p10 * tx * (1.0 - ty)
            + p01 * (1.0 - tx) * ty
            + p11 * tx * ty;
    }
    var result = select(value.r, value.a, ((flags >> 8u) & 3u) == 1u);
    if (((flags >> 8u) & 3u) == 2u) {
        result = (value.r + value.g + value.b) / 3.0;
    }
    if ((flags & 1u) != 0u) {
        result = 1.0 - result;
    }
    return result * scale;
}

fn u32_mul(left: u32, right: u32) -> vec2<u32> {
    let left0 = left & 0xffffu;
    let left1 = left >> 16u;
    let right0 = right & 0xffffu;
    let right1 = right >> 16u;
    let product0 = left0 * right0;
    let middle0 = (product0 >> 16u) + left0 * right1;
    let middle1 = (middle0 & 0xffffu) + left1 * right0;
    return vec2<u32>(
        (product0 & 0xffffu) | ((middle1 & 0xffffu) << 16u),
        left1 * right1 + (middle0 >> 16u) + (middle1 >> 16u),
    );
}

fn u64_mul(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
    let low_product = u32_mul(left.x, right.x);
    let left_high_product = u32_mul(left.y, right.x);
    let right_high_product = u32_mul(left.x, right.y);
    return vec2<u32>(
        low_product.x,
        low_product.y + left_high_product.x + right_high_product.x,
    );
}

fn u64_shift_right_47(value: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(value.y >> 15u, 0u);
}

fn murmur_hash_float_ray(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    let multiplier = vec2<u32>(0x5bd1e995u, 0xc6a4a793u);
    var hash = u64_mul(vec2<u32>(48u, 0u), multiplier);
    let values = array<u32, 6>(
        bitcast<u32>(origin.x), bitcast<u32>(origin.y), bitcast<u32>(origin.z),
        bitcast<u32>(direction.x), bitcast<u32>(direction.y), bitcast<u32>(direction.z),
    );
    for (var index = 0u; index < 6u; index += 2u) {
        var key = vec2<u32>(values[index], values[index + 1u]);
        key = u64_mul(key, multiplier);
        key = key ^ u64_shift_right_47(key);
        key = u64_mul(key, multiplier);
        hash = u64_mul(hash ^ key, multiplier);
    }
    hash = hash ^ u64_shift_right_47(hash);
    hash = u64_mul(hash, multiplier);
    hash = hash ^ u64_shift_right_47(hash);
    return f32(hash.x) * 2.3283064365386963e-10;
}

fn alpha_accept(alpha: f32, origin: vec3<f32>, direction: vec3<f32>) -> bool {
    if (alpha >= 1.0) {
        return true;
    }
    if (alpha <= 0.0) {
        return false;
    }
    return murmur_hash_float_ray(origin, direction) <= alpha;
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
    while (rayQueryProceed(&query)) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        let primitive = primitives[candidate.instance_custom_data];
        let index_offset = primitive.first_index + candidate.primitive_index * 3u;
        let i0 = primitive.first_vertex + indices[index_offset];
        let i1 = primitive.first_vertex + indices[index_offset + 1u];
        let i2 = primitive.first_vertex + indices[index_offset + 2u];
        let barycentrics = vec3<f32>(
            1.0 - candidate.barycentrics.x - candidate.barycentrics.y,
            candidate.barycentrics.x,
            candidate.barycentrics.y,
        );
        let uv = vertices[i0].uv.xy * barycentrics.x
            + vertices[i1].uv.xy * barycentrics.y
            + vertices[i2].uv.xy * barycentrics.z;
        if (primitive.alpha == 0xffffffffu
            || alpha_accept(
                sample_float_texture(primitive.alpha, uv),
                origin,
                direction,
            )) {
            rayQueryConfirmIntersection(&query);
        }
    }
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
            if (light.kind == 1u) {
                radiance += reflectance * light.intensity.xyz;
            } else {
                let to_light = light.position.xyz - position;
                let distance_squared = max(dot(to_light, to_light), 1.0e-8);
                let wi = normalize(to_light);
                let cosine = max(dot(normal, wi), 0.0);
                radiance += reflectance * light.intensity.xyz * cosine / distance_squared;
            }
        }
        color = vec4<f32>(radiance, 1.0);
    }

    let output_index = global_id.y * width + global_id.x;
    output[output_index] = color;
}
