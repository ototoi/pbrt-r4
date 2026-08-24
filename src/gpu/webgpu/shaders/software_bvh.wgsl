struct Camera {
    camera_from_raster: mat4x4<f32>,
    render_from_camera: mat4x4<f32>,
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
    normal_map: u32,
    displacement: u32,
    flags: u32,
};

struct Vertex {
    position: vec4<f32>,
    uv: vec4<f32>,
    normal: vec4<f32>,
    tangent: vec4<f32>,
};

struct Transform {
    render_from_object: mat4x4<f32>,
    normal_from_object: mat4x4<f32>,
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

fn transform(matrix: mat4x4<f32>, value: vec4<f32>) -> vec4<f32> {
    return matrix * value;
}

fn image_mip_base(image_base: u32, level: u32) -> u32 {
    let levels = scene_data[image_base + 4u];
    let selected = min(level, levels - 1u);
    return scene_data[image_base + 3u] + selected * 4u;
}

fn texture_lod(image_base: u32, differentials: vec4<f32>) -> f32 {
    let levels = scene_data[image_base + 4u];
    let footprint = 2.0 * max(
        max(abs(differentials.x), abs(differentials.y)),
        max(abs(differentials.z), abs(differentials.w)),
    );
    let level = f32(levels - 1u) + log2(max(footprint, 1.0e-8));
    return level;
}

fn image_texel(
    image_base: u32,
    level: u32,
    x: i32,
    y: i32,
    swrap: u32,
    twrap: u32,
) -> vec4<f32> {
    let mip_base = image_mip_base(image_base, level);
    let width = scene_data[mip_base];
    let height = scene_data[mip_base + 1u];
    let channels = scene_data[image_base + 2u];
    var texel_base = scene_data[image_base + 5u];
    if (level > 0u) {
        texel_base = scene_data[mip_base + 2u];
    }
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

fn sample_image_level(
    image_base: u32,
    level: u32,
    st: vec2<f32>,
    swrap: u32,
    twrap: u32,
    bilinear: bool,
) -> vec4<f32> {
    let mip_base = image_mip_base(image_base, level);
    let resolution = vec2<f32>(f32(scene_data[mip_base]), f32(scene_data[mip_base + 1u]));
    let coordinate = st * resolution - vec2<f32>(0.5);
    if (!bilinear) {
        return image_texel(
            image_base,
            level,
            i32(round(coordinate.x)),
            i32(round(coordinate.y)),
            swrap,
            twrap,
        );
    }
    let p = vec2<i32>(floor(coordinate));
    let weight = fract(coordinate);
    let p00 = image_texel(image_base, level, p.x, p.y, swrap, twrap);
    let p10 = image_texel(image_base, level, p.x + 1, p.y, swrap, twrap);
    let p01 = image_texel(image_base, level, p.x, p.y + 1, swrap, twrap);
    let p11 = image_texel(image_base, level, p.x + 1, p.y + 1, swrap, twrap);
    return p00 * (1.0 - weight.x) * (1.0 - weight.y)
        + p10 * weight.x * (1.0 - weight.y)
        + p01 * (1.0 - weight.x) * weight.y
        + p11 * weight.x * weight.y;
}

fn sample_image(
    image_base: u32,
    st: vec2<f32>,
    swrap: u32,
    twrap: u32,
    filter_mode: u32,
    differentials: vec4<f32>,
) -> vec4<f32> {
    let levels = scene_data[image_base + 4u];
    let level = texture_lod(image_base, differentials);
    if (level >= f32(levels - 1u)) {
        return image_texel(image_base, levels - 1u, 0, 0, swrap, twrap);
    }
    let i_level = u32(max(0.0, floor(level)));
    if (filter_mode == 0u) {
        return sample_image_level(image_base, i_level, st, swrap, twrap, false);
    }
    let value = sample_image_level(image_base, i_level, st, swrap, twrap, true);
    if (filter_mode != 2u || i_level == 0u) {
        return value;
    }
    let next = sample_image_level(image_base, i_level + 1u, st, swrap, twrap, true);
    return mix(value, next, level - f32(i_level));
}

fn sample_texture(texture_id: u32, uv: vec2<f32>, differentials: vec4<f32>) -> vec3<f32> {
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
    var value = sample_image(
        image_base,
        st,
        swrap,
        twrap,
        (flags >> 5u) & 3u,
        differentials * vec4<f32>(su, sv, su, sv),
    );
    if ((flags & 1u) != 0u) {
        value = vec4<f32>(vec3<f32>(1.0) - value.xyz, value.w);
    }
    if (((flags >> 7u) & 3u) == 0u) {
        value = vec4<f32>(clamp(value.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), value.w);
    }
    return value.xyz * scale;
}

fn sample_normal_map(image_id: u32, uv: vec2<f32>) -> vec3<f32> {
    var st = vec2<f32>(uv.x, 1.0 - uv.y);
    st = fract(st);
    let image_base = scene_data[0u] + image_id * 8u;
    let width = scene_data[image_base];
    let height = scene_data[image_base + 1u];
    let coordinate = st * vec2<f32>(f32(width), f32(height)) - vec2<f32>(0.5);
    let x0 = i32(floor(coordinate.x));
    let y0 = i32(floor(coordinate.y));
    let tx = fract(coordinate.x);
    let ty = fract(coordinate.y);
    let p00 = image_texel(image_base, 0u, x0, y0, 2u, 2u).xyz;
    let p10 = image_texel(image_base, 0u, x0 + 1, y0, 2u, 2u).xyz;
    let p01 = image_texel(image_base, 0u, x0, y0 + 1, 2u, 2u).xyz;
    let p11 = image_texel(image_base, 0u, x0 + 1, y0 + 1, 2u, 2u).xyz;
    let value = p00 * (1.0 - tx) * (1.0 - ty)
        + p10 * tx * (1.0 - ty)
        + p01 * (1.0 - tx) * ty
        + p11 * tx * ty;
    return normalize(2.0 * value - vec3<f32>(1.0));
}

fn coordinate_tangent(normal: vec3<f32>) -> vec3<f32> {
    if (abs(normal.x) > abs(normal.y)) {
        return normalize(vec3<f32>(-normal.z, 0.0, normal.x));
    }
    return normalize(vec3<f32>(0.0, normal.z, -normal.y));
}

fn triangle_dpdu(
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
    normal: vec3<f32>,
) -> vec3<f32> {
    let dp02 = p0 - p2;
    let dp12 = p1 - p2;
    let duv02 = uv0 - uv2;
    let duv12 = uv1 - uv2;
    let determinant = duv02.x * duv12.y - duv02.y * duv12.x;
    if (abs(determinant) >= 1.0e-9) {
        let inverse = 1.0 / determinant;
        return (dp02 * duv12.y - dp12 * duv02.y) * inverse;
    }
    return coordinate_tangent(normal);
}

fn triangle_dpdv(
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
    normal: vec3<f32>,
) -> vec3<f32> {
    let dp02 = p0 - p2;
    let dp12 = p1 - p2;
    let duv02 = uv0 - uv2;
    let duv12 = uv1 - uv2;
    let determinant = duv02.x * duv12.y - duv02.y * duv12.x;
    if (abs(determinant) >= 1.0e-9) {
        let inverse = 1.0 / determinant;
        return (dp12 * duv02.x - dp02 * duv12.x) * inverse;
    }
    let tangent = coordinate_tangent(normal);
    return cross(normal, tangent);
}

fn normal_derivative_u(
    n0: vec3<f32>,
    n1: vec3<f32>,
    n2: vec3<f32>,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
) -> vec3<f32> {
    let duv02 = uv0 - uv2;
    let duv12 = uv1 - uv2;
    let determinant = duv02.x * duv12.y - duv02.y * duv12.x;
    if (abs(determinant) < 1.0e-9) {
        return vec3<f32>(0.0);
    }
    return ((n0 - n2) * duv12.y - (n1 - n2) * duv02.y) / determinant;
}

fn normal_derivative_v(
    n0: vec3<f32>,
    n1: vec3<f32>,
    n2: vec3<f32>,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
) -> vec3<f32> {
    let duv02 = uv0 - uv2;
    let duv12 = uv1 - uv2;
    let determinant = duv02.x * duv12.y - duv02.y * duv12.x;
    if (abs(determinant) < 1.0e-9) {
        return vec3<f32>(0.0);
    }
    return ((n1 - n2) * duv02.x - (n0 - n2) * duv12.x) / determinant;
}

fn sane_derivative(value: f32) -> f32 {
    if (value != value || abs(value) > 1.0e8) {
        return 0.0;
    }
    return clamp(value, -1.0e8, 1.0e8);
}

fn uv_differentials(
    p: vec3<f32>,
    normal: vec3<f32>,
    dpdu: vec3<f32>,
    dpdv: vec3<f32>,
    rx_origin: vec3<f32>,
    rx_direction: vec3<f32>,
    ry_origin: vec3<f32>,
    ry_direction: vec3<f32>,
) -> vec4<f32> {
    let plane_distance = -dot(normal, p);
    var dpdx = vec3<f32>(0.0);
    var dpdy = vec3<f32>(0.0);
    let x_denominator = dot(normal, rx_direction);
    if (abs(x_denominator) > 1.0e-8) {
        let tx = (-dot(normal, rx_origin) - plane_distance) / x_denominator;
        dpdx = rx_origin + tx * rx_direction - p;
    }
    let y_denominator = dot(normal, ry_direction);
    if (abs(y_denominator) > 1.0e-8) {
        let ty = (-dot(normal, ry_origin) - plane_distance) / y_denominator;
        dpdy = ry_origin + ty * ry_direction - p;
    }
    let ata00 = dot(dpdu, dpdu);
    let ata01 = dot(dpdu, dpdv);
    let ata11 = dot(dpdv, dpdv);
    let determinant = ata00 * ata11 - ata01 * ata01;
    var inverse = 0.0;
    if (abs(determinant) > 1.0e-20) {
        inverse = 1.0 / determinant;
    }
    let atb0x = dot(dpdu, dpdx);
    let atb1x = dot(dpdv, dpdx);
    let atb0y = dot(dpdu, dpdy);
    let atb1y = dot(dpdv, dpdy);
    return vec4<f32>(
        sane_derivative((ata11 * atb0x - ata01 * atb1x) * inverse),
        sane_derivative((ata00 * atb1x - ata01 * atb0x) * inverse),
        sane_derivative((ata11 * atb0y - ata01 * atb1y) * inverse),
        sane_derivative((ata00 * atb1y - ata01 * atb0y) * inverse),
    );
}

fn apply_bump_map(
    texture_id: u32,
    uv: vec2<f32>,
    normal: vec3<f32>,
    dpdu: vec3<f32>,
    dpdv: vec3<f32>,
    dndu: vec3<f32>,
    dndv: vec3<f32>,
    differentials: vec4<f32>,
) -> vec3<f32> {
    var du = 0.5 * (abs(differentials.x) + abs(differentials.z));
    var dv = 0.5 * (abs(differentials.y) + abs(differentials.w));
    if (du == 0.0) {
        du = 0.0005;
    }
    if (dv == 0.0) {
        dv = 0.0005;
    }
    let displacement = sample_float_texture(texture_id, uv, differentials);
    let u_displacement = sample_float_texture(
        texture_id,
        uv + vec2<f32>(du, 0.0),
        differentials,
    );
    let v_displacement = sample_float_texture(
        texture_id,
        uv + vec2<f32>(0.0, dv),
        differentials,
    );
    let bumped_dpdu = dpdu + (u_displacement - displacement) / du * normal + displacement * dndu;
    let bumped_dpdv = dpdv + (v_displacement - displacement) / dv * normal + displacement * dndv;
    return normalize(cross(bumped_dpdu, bumped_dpdv));
}

fn apply_normal_map(
    image_id: u32,
    uv: vec2<f32>,
    object_normal: vec3<f32>,
    object_dpdu: vec3<f32>,
    object_tangent: vec3<f32>,
    render_from_object: mat4x4<f32>,
    normal_from_object: mat4x4<f32>,
) -> vec3<f32> {
    let normal = normalize(transform(normal_from_object, vec4<f32>(object_normal, 0.0)).xyz);
    let tangent_source = select(object_dpdu, object_tangent, dot(object_tangent, object_tangent) > 1.0e-20);
    let transformed_tangent = transform(render_from_object, vec4<f32>(tangent_source, 0.0)).xyz;
    let tangent = normalize(transformed_tangent - normal * dot(normal, transformed_tangent));
    let safe_tangent = select(coordinate_tangent(normal), tangent, dot(tangent, tangent) > 1.0e-20);
    let bitangent = cross(normal, safe_tangent);
    let mapped = sample_normal_map(image_id, uv);
    return normalize(safe_tangent * mapped.x + bitangent * mapped.y + normal * mapped.z);
}

fn sample_float_texture(texture_id: u32, uv: vec2<f32>, differentials: vec4<f32>) -> f32 {
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
    let value = sample_image(
        image_base,
        st,
        swrap,
        twrap,
        (flags >> 5u) & 3u,
        differentials * vec4<f32>(su, sv, su, sv),
    );
    var result = select(value.r, value.a, ((flags >> 9u) & 3u) == 1u);
    if (((flags >> 9u) & 3u) == 2u) {
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
    let x_target = transform(
        camera.camera_from_raster,
        vec4<f32>(pixel + vec2<f32>(1.0, 0.0), 0.0, 1.0),
    );
    let y_target = transform(
        camera.camera_from_raster,
        vec4<f32>(pixel + vec2<f32>(0.0, 1.0), 0.0, 1.0),
    );
    let rx_origin = origin;
    let ry_origin = origin;
    let rx_direction = normalize(
        transform(camera.render_from_camera, x_target).xyz - rx_origin,
    );
    let ry_direction = normalize(
        transform(camera.render_from_camera, y_target).xyz - ry_origin,
    );
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
                let p0 = transform(transform_table.render_from_object, vec4<f32>(object_p0, 1.0)).xyz;
                let p1 = transform(transform_table.render_from_object, vec4<f32>(object_p1, 1.0)).xyz;
                let p2 = transform(transform_table.render_from_object, vec4<f32>(object_p2, 1.0)).xyz;
                let distance = ray_hits_triangle(origin, direction, p0, p1, p2);
                if (distance > 0.0 && distance < closest) {
                    let hit = origin + distance * direction;
                    let edge1 = p1 - p0;
                    let edge2 = p2 - p0;
                    let denominator = dot(edge1, edge1) * dot(edge2, edge2)
                        - dot(edge1, edge2) * dot(edge1, edge2);
                    let relative = hit - p0;
                    let beta = (dot(relative, edge2) * dot(edge1, edge1)
                        - dot(relative, edge1) * dot(edge1, edge2)) / denominator;
                    let gamma = (dot(relative, edge1) * dot(edge2, edge2)
                        - dot(relative, edge2) * dot(edge1, edge2)) / denominator;
                    let barycentric = 1.0 - beta - gamma;
                    let uv = vertices[i0].uv.xy * barycentric
                        + vertices[i1].uv.xy * beta
                        + vertices[i2].uv.xy * gamma;
                    if (primitive.alpha != 0xffffffffu
                        && !alpha_accept(
                            sample_float_texture(primitive.alpha, uv, vec4<f32>(0.0)),
                            origin,
                            direction,
                        )) {
                        continue;
                    }
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
        let wp0 = transform(transform_table.render_from_object, vec4<f32>(p0, 1.0)).xyz;
        let wp1 = transform(transform_table.render_from_object, vec4<f32>(p1, 1.0)).xyz;
        let wp2 = transform(transform_table.render_from_object, vec4<f32>(p2, 1.0)).xyz;
        let edge1 = wp1 - wp0;
        let edge2 = wp2 - wp0;
        let d = dot(edge1, edge1) * dot(edge2, edge2) - dot(edge1, edge2) * dot(edge1, edge2);
        let rel = hit_position - wp0;
        let beta = (dot(rel, edge2) * dot(edge1, edge1) - dot(rel, edge1) * dot(edge1, edge2)) / d;
        let gamma = (dot(rel, edge1) * dot(edge2, edge2) - dot(rel, edge2) * dot(edge1, edge2)) / d;
        let barycentrics = vec3<f32>(1.0 - beta - gamma, beta, gamma);
        let geometric_normal = normalize(cross(p1 - p0, p2 - p0));
        let material = materials[primitive.material];
        let v0 = primitive.first_vertex + indices[index_offset];
        let v1 = primitive.first_vertex + indices[index_offset + 1u];
        let v2 = primitive.first_vertex + indices[index_offset + 2u];
        let interpolated_normal = vertices[v0].normal.xyz * barycentrics.x
            + vertices[v1].normal.xyz * barycentrics.y
            + vertices[v2].normal.xyz * barycentrics.z;
        var object_normal = geometric_normal;
        if (dot(interpolated_normal, interpolated_normal) > 1.0e-20) {
            object_normal = normalize(interpolated_normal);
        }
        let uv0 = vertices[v0].uv.xy;
        let uv1 = vertices[v1].uv.xy;
        let uv2 = vertices[v2].uv.xy;
        let uv = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;
        let object_dpdu = triangle_dpdu(p0, p1, p2, uv0, uv1, uv2, object_normal);
        let object_dpdv = triangle_dpdv(p0, p1, p2, uv0, uv1, uv2, object_normal);
        let object_dndu = normal_derivative_u(
            vertices[v0].normal.xyz,
            vertices[v1].normal.xyz,
            vertices[v2].normal.xyz,
            uv0,
            uv1,
            uv2,
        );
        let object_dndv = normal_derivative_v(
            vertices[v0].normal.xyz,
            vertices[v1].normal.xyz,
            vertices[v2].normal.xyz,
            uv0,
            uv1,
            uv2,
        );
        let object_tangent = vertices[v0].tangent.xyz * barycentrics.x
            + vertices[v1].tangent.xyz * barycentrics.y
            + vertices[v2].tangent.xyz * barycentrics.z;
        var normal = normalize(transform(transform_table.normal_from_object, vec4<f32>(object_normal, 0.0)).xyz);
        let dpdu = transform(transform_table.render_from_object, vec4<f32>(object_dpdu, 0.0)).xyz;
        let dpdv = transform(transform_table.render_from_object, vec4<f32>(object_dpdv, 0.0)).xyz;
        let dndu = transform(transform_table.normal_from_object, vec4<f32>(object_dndu, 0.0)).xyz;
        let dndv = transform(transform_table.normal_from_object, vec4<f32>(object_dndv, 0.0)).xyz;
        let differentials = uv_differentials(
            position,
            normal,
            dpdu,
            dpdv,
            rx_origin,
            rx_direction,
            ry_origin,
            ry_direction,
        );
        if ((material.flags & 2u) != 0u) {
            normal = apply_normal_map(
                material.normal_map,
                uv,
                object_normal,
                object_dpdu,
                object_tangent,
                transform_table.render_from_object,
                transform_table.normal_from_object,
            );
        } else if ((material.flags & 4u) != 0u) {
            normal = apply_bump_map(
                material.displacement,
                uv,
                normalize(transform(transform_table.normal_from_object, vec4<f32>(object_normal, 0.0)).xyz),
                dpdu,
                dpdv,
                dndu,
                dndv,
                differentials,
            );
        }
        var reflectance = material.reflectance.xyz;
        if ((material.flags & 1u) != 0u) {
            reflectance = sample_texture(material.texture, uv, differentials);
        }
        for (var light_index = 0u; light_index < arrayLength(&lights); light_index += 1u) {
            let light = lights[light_index];
            if (light.kind == 1u) {
                color += reflectance * light.intensity.xyz;
            } else {
                let to_light = light.position.xyz - position;
                let distance_squared = max(dot(to_light, to_light), 1.0e-8);
                color += reflectance * light.intensity.xyz * max(dot(normal, normalize(to_light)), 0.0) / distance_squared;
            }
        }
    }
    output[global_id.y * width + global_id.x] = vec4<f32>(color, 1.0);
}
