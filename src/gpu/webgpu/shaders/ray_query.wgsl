enable wgpu_ray_query;

struct Camera {
    camera_from_raster: mat4x4<f32>,
    render_from_camera: mat4x4<f32>,
    viewport: vec4<f32>,
    bvh_info: vec4<u32>,
    sampler_info: vec4<u32>,
    filter_info: vec4<f32>,
    camera_info: vec4<f32>,
};

struct Primitive {
    first_vertex: u32,
    first_index: u32,
    material: u32,
    alpha: u32,
    _reserved_alpha: u32,
    flags: u32,
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

fn sample_image_ewa_level(
    image_base: u32,
    level: u32,
    st: vec2<f32>,
    swrap: u32,
    twrap: u32,
    dst0: vec2<f32>,
    dst1: vec2<f32>,
) -> vec4<f32> {
    let levels = scene_data[image_base + 4u];
    if (level >= levels) {
        return image_texel(image_base, levels - 1u, 0, 0, swrap, twrap);
    }
    let mip_base = image_mip_base(image_base, level);
    let resolution = vec2<f32>(f32(scene_data[mip_base]), f32(scene_data[mip_base + 1u]));
    let texture_st = st * resolution - vec2<f32>(0.5);
    let d0 = dst0 * resolution;
    let d1 = dst1 * resolution;
    var a = d0.y * d0.y + d1.y * d1.y + 1.0;
    var b = -2.0 * (d0.x * d0.y + d1.x * d1.y);
    var c = d0.x * d0.x + d1.x * d1.x + 1.0;
    let inverse_f = 1.0 / (a * c - 0.25 * b * b);
    a *= inverse_f;
    b *= inverse_f;
    c *= inverse_f;
    let determinant = -b * b + 4.0 * a * c;
    let inverse_determinant = 1.0 / determinant;
    let u_sqrt = sqrt(max(0.0, determinant * c));
    let v_sqrt = sqrt(max(0.0, a * determinant));
    let s0 = i32(ceil(texture_st.x - 2.0 * inverse_determinant * u_sqrt));
    let s1 = i32(floor(texture_st.x + 2.0 * inverse_determinant * u_sqrt));
    let t0 = i32(ceil(texture_st.y - 2.0 * inverse_determinant * v_sqrt));
    let t1 = i32(floor(texture_st.y + 2.0 * inverse_determinant * v_sqrt));
    var sum = vec4<f32>(0.0);
    var sum_weights = 0.0;
    for (var t = t0; t <= t1; t++) {
        let tt = f32(t) - texture_st.y;
        for (var s = s0; s <= s1; s++) {
            let ss = f32(s) - texture_st.x;
            let radius_squared = a * ss * ss + b * ss * tt + c * tt * tt;
            if (radius_squared < 1.0) {
                let index = min(u32(radius_squared * 128.0), 127u);
                let weight = bitcast<f32>(scene_data[8u + index]);
                sum += weight * image_texel(image_base, level, s, t, swrap, twrap);
                sum_weights += weight;
            }
        }
    }
    return sum / sum_weights;
}

fn sample_image(
    image_base: u32,
    st: vec2<f32>,
    swrap: u32,
    twrap: u32,
    filter_mode: u32,
    max_anisotropy: f32,
    differentials: vec4<f32>,
) -> vec4<f32> {
    let levels = scene_data[image_base + 4u];
    if (filter_mode == 3u) {
        var dst0 = differentials.xy;
        var dst1 = differentials.zw;
        if (dot(dst0, dst0) < dot(dst1, dst1)) {
            let temporary = dst0;
            dst0 = dst1;
            dst1 = temporary;
        }
        let longer_length = length(dst0);
        var shorter_length = length(dst1);
        if (shorter_length * max_anisotropy < longer_length && shorter_length > 0.0) {
            let scale = longer_length / (shorter_length * max_anisotropy);
            dst1 *= scale;
            shorter_length *= scale;
        }
        if (shorter_length == 0.0) {
            return sample_image_level(image_base, 0u, st, swrap, twrap, true);
        }
        let lod = max(0.0, f32(levels - 1u) + log2(shorter_length));
        let integer_lod = u32(floor(lod));
        let value = sample_image_ewa_level(image_base, integer_lod, st, swrap, twrap, dst0, dst1);
        let next = sample_image_ewa_level(image_base, integer_lod + 1u, st, swrap, twrap, dst0, dst1);
        return mix(value, next, lod - f32(integer_lod));
    }
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
        bitcast<f32>(scene_data[texture_base + 7u]),
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
    normal: vec3<f32>,
    object_dpdu: vec3<f32>,
    object_tangent: vec3<f32>,
    render_from_object: mat4x4<f32>,
) -> vec3<f32> {
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
        bitcast<f32>(scene_data[texture_base + 7u]),
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

fn u64_add(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
    let low = left.x + right.x;
    return vec2<u32>(low, left.y + right.y + select(0u, 1u, low < left.x));
}

fn u64_shift_right(value: vec2<u32>, shift: u32) -> vec2<u32> {
    if (shift == 0u) {
        return value;
    }
    if (shift < 32u) {
        return vec2<u32>(
            (value.x >> shift) | (value.y << (32u - shift)),
            value.y >> shift,
        );
    }
    if (shift < 64u) {
        return vec2<u32>(value.y >> (shift - 32u), 0u);
    }
    return vec2<u32>(0u);
}

fn u64_shift_left_one(value: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(value.x << 1u, (value.y << 1u) | (value.x >> 31u));
}

fn mix_bits(value: vec2<u32>) -> vec2<u32> {
    var mixed = value ^ u64_shift_right(value, 31u);
    mixed = u64_mul(mixed, vec2<u32>(0x728ea185u, 0x7fb5d329u));
    mixed = mixed ^ u64_shift_right(mixed, 27u);
    mixed = u64_mul(mixed, vec2<u32>(0xbc2dd44du, 0x81dadef4u));
    return mixed ^ u64_shift_right(mixed, 33u);
}

fn hash_pixel_seed(pixel: vec2<i32>, seed: u32) -> vec2<u32> {
    let multiplier = vec2<u32>(0x5bd1e995u, 0xc6a4a793u);
    var hash = u64_mul(vec2<u32>(12u, 0u), multiplier);
    var key = vec2<u32>(bitcast<u32>(pixel.x), bitcast<u32>(pixel.y));
    key = u64_mul(key, multiplier);
    key = key ^ u64_shift_right(key, 47u);
    key = u64_mul(key, multiplier);
    hash = u64_mul(hash ^ key, multiplier);
    hash = u64_mul(hash ^ vec2<u32>(seed, 0u), multiplier);
    hash = hash ^ u64_shift_right(hash, 47u);
    hash = u64_mul(hash, multiplier);
    return hash ^ u64_shift_right(hash, 47u);
}

struct IndependentRng {
    state: vec2<u32>,
    increment: vec2<u32>,
};

fn rng_uniform_u32(rng: ptr<function, IndependentRng>) -> u32 {
    let old_state = (*rng).state;
    (*rng).state = u64_add(
        u64_mul(old_state, vec2<u32>(0x4c957f2du, 0x5851f42du)),
        (*rng).increment,
    );
    let xorshifted = u64_shift_right(u64_shift_right(old_state, 18u) ^ old_state, 27u).x;
    let rotation = u64_shift_right(old_state, 59u).x;
    return (xorshifted >> rotation) | (xorshifted << ((0u - rotation) & 31u));
}

fn rng_advance(rng: ptr<function, IndependentRng>, sample_index: u32, dimension: u32) {
    var current_multiplier = vec2<u32>(0x4c957f2du, 0x5851f42du);
    var current_plus = (*rng).increment;
    var accumulated_multiplier = vec2<u32>(1u, 0u);
    var accumulated_plus = vec2<u32>(0u);
    var delta = u64_add(
        vec2<u32>(sample_index << 16u, sample_index >> 16u),
        vec2<u32>(dimension, 0u),
    );
    while (delta.x != 0u || delta.y != 0u) {
        if ((delta.x & 1u) != 0u) {
            accumulated_multiplier = u64_mul(accumulated_multiplier, current_multiplier);
            accumulated_plus = u64_add(u64_mul(accumulated_plus, current_multiplier), current_plus);
        }
        current_plus = u64_mul(u64_add(current_multiplier, vec2<u32>(1u, 0u)), current_plus);
        current_multiplier = u64_mul(current_multiplier, current_multiplier);
        delta = u64_shift_right(delta, 1u);
    }
    (*rng).state = u64_add(
        u64_mul(accumulated_multiplier, (*rng).state),
        accumulated_plus,
    );
}

struct IndependentCameraSample {
    filter_sample: vec2<f32>,
    time: f32,
    lens: vec2<f32>,
};

struct IndependentDirectSample {
    light_selection: f32,
    light_sample: vec2<f32>,
};

struct IndependentIndirectSample {
    component: f32,
    direction: vec2<f32>,
    roulette: f32,
};

struct IndependentRaySample {
    direct: IndependentDirectSample,
    indirect: IndependentIndirectSample,
};

fn uniform_float(rng: ptr<function, IndependentRng>) -> f32 {
    return min(0.9999999403953552, f32(rng_uniform_u32(rng)) * 2.3283064365386963e-10);
}

fn independent_rng(pixel: vec2<i32>, sample_index: u32, dimension: u32) -> IndependentRng {
    let sequence = hash_pixel_seed(pixel, camera.sampler_info.x);
    var rng = IndependentRng(vec2<u32>(0u), u64_shift_left_one(sequence) | vec2<u32>(1u, 0u));
    _ = rng_uniform_u32(&rng);
    rng.state = u64_add(rng.state, mix_bits(sequence));
    _ = rng_uniform_u32(&rng);
    rng_advance(&rng, sample_index, dimension);
    return rng;
}

fn independent_camera_sample(pixel: vec2<i32>, sample_index: u32) -> IndependentCameraSample {
    var rng = independent_rng(pixel, sample_index, 0u);
    return IndependentCameraSample(
        vec2<f32>(uniform_float(&rng), uniform_float(&rng)),
        uniform_float(&rng),
        vec2<f32>(uniform_float(&rng), uniform_float(&rng)),
    );
}

fn independent_ray_sample(pixel: vec2<i32>, sample_index: u32, depth: u32) -> IndependentRaySample {
    var rng = independent_rng(pixel, sample_index, 6u + 7u * depth);
    let direct_selection = uniform_float(&rng);
    let direct_u0 = uniform_float(&rng);
    let direct_u1 = uniform_float(&rng);
    let indirect_component = uniform_float(&rng);
    let indirect_u0 = uniform_float(&rng);
    let indirect_u1 = uniform_float(&rng);
    let roulette = uniform_float(&rng);
    return IndependentRaySample(
        IndependentDirectSample(
            direct_selection,
            vec2<f32>(direct_u0, direct_u1),
        ),
        IndependentIndirectSample(
            indirect_component,
            vec2<f32>(indirect_u0, indirect_u1),
            roulette,
        ),
    );
}

fn sample_uniform_disk_concentric(u: vec2<f32>) -> vec2<f32> {
    let offset = 2.0 * u - vec2<f32>(1.0);
    if (offset.x == 0.0 && offset.y == 0.0) {
        return vec2<f32>(0.0);
    }
    var radius: f32;
    var theta: f32;
    if (abs(offset.x) > abs(offset.y)) {
        radius = offset.x;
        theta = 0.7853981633974483 * offset.y / offset.x;
    } else {
        radius = offset.y;
        theta = 1.5707963267948966 - 0.7853981633974483 * offset.x / offset.y;
    }
    return radius * vec2<f32>(cos(theta), sin(theta));
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

fn next_float_up(value: f32) -> f32 {
    if (value == bitcast<f32>(0x7f800000u)) {
        return value;
    }
    var adjusted = value;
    if (adjusted == -0.0) {
        adjusted = 0.0;
    }
    var bits = bitcast<u32>(adjusted);
    if (adjusted >= 0.0) {
        bits += 1u;
    } else {
        bits -= 1u;
    }
    return bitcast<f32>(bits);
}

fn next_float_down(value: f32) -> f32 {
    if (value == bitcast<f32>(0xff800000u)) {
        return value;
    }
    var adjusted = value;
    if (adjusted == 0.0) {
        adjusted = -0.0;
    }
    var bits = bitcast<u32>(adjusted);
    if (adjusted > 0.0) {
        bits -= 1u;
    } else {
        bits += 1u;
    }
    return bitcast<f32>(bits);
}

fn transformed_position_error(
    matrix: mat4x4<f32>,
    position: vec3<f32>,
    position_error: vec3<f32>,
) -> vec3<f32> {
    let gamma3 = 1.7881397e-7;
    var result = vec3<f32>(0.0);
    for (var row = 0u; row < 3u; row++) {
        let propagated = abs(matrix[0u][row]) * position_error.x
            + abs(matrix[1u][row]) * position_error.y
            + abs(matrix[2u][row]) * position_error.z;
        let rounded = abs(matrix[0u][row] * position.x)
            + abs(matrix[1u][row] * position.y)
            + abs(matrix[2u][row] * position.z)
            + abs(matrix[3u][row]);
        result[row] = (1.0 + gamma3) * propagated + gamma3 * rounded;
    }
    return result;
}

fn offset_ray_origin(
    position: vec3<f32>,
    position_error: vec3<f32>,
    normal: vec3<f32>,
    direction: vec3<f32>,
) -> vec3<f32> {
    let distance = dot(abs(normal), position_error);
    var offset = distance * normal;
    if (dot(direction, normal) < 0.0) {
        offset = -offset;
    }
    var origin = position + offset;
    for (var axis = 0u; axis < 3u; axis++) {
        if (offset[axis] > 0.0) {
            origin[axis] = next_float_up(origin[axis]);
        } else if (offset[axis] < 0.0) {
            origin[axis] = next_float_down(origin[axis]);
        }
    }
    return origin;
}

fn shadow_visible(
    origin: vec3<f32>,
    direction: vec3<f32>,
    source_primitive: u32,
    source_triangle: u32,
) -> bool {
    var query: ray_query;
    rayQueryInitialize(
        &query,
        acceleration,
        RayDesc(0u, 0xFFu, 0.0, 0.9999, origin, direction),
    );
    while (rayQueryProceed(&query)) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        if (candidate.instance_custom_data == source_primitive
            && candidate.primitive_index == source_triangle) {
            continue;
        }
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
                sample_float_texture(primitive.alpha, uv, vec4<f32>(0.0)),
                origin,
                direction,
            )) {
            rayQueryConfirmIntersection(&query);
        }
    }
    return rayQueryGetCommittedIntersection(&query).kind == RAY_QUERY_INTERSECTION_NONE;
}

fn infinite_emission() -> vec3<f32> {
    var emission = vec3<f32>(0.0);
    for (var light_index = 0u; light_index < arrayLength(&lights); light_index += 1u) {
        let light = lights[light_index];
        if (light.kind == 1u) {
            emission += light.intensity.xyz;
        }
    }
    return emission;
}

fn render_sample(
    pixel: vec2<f32>,
    lens_sample: vec2<f32>,
    sample_pixel: vec2<i32>,
    sample_index: u32,
) -> vec3<f32> {
    let camera_target = transform(
        camera.camera_from_raster,
        vec4<f32>(pixel, 0.0, 1.0),
    );
    var camera_origin = vec3<f32>(0.0);
    var camera_direction = normalize(camera_target.xyz);
    if (camera.camera_info.x > 0.0) {
        let lens = camera.camera_info.x * sample_uniform_disk_concentric(lens_sample);
        let focus_t = camera.camera_info.y / camera_direction.z;
        let focus = focus_t * camera_direction;
        camera_origin = vec3<f32>(lens, 0.0);
        camera_direction = normalize(focus - camera_origin);
    }
    var ray_origin = transform(camera.render_from_camera, vec4<f32>(camera_origin, 1.0)).xyz;
    var ray_direction = normalize(transform(camera.render_from_camera, vec4<f32>(camera_direction, 0.0)).xyz);
    let x_target = transform(
        camera.camera_from_raster,
        vec4<f32>(pixel + vec2<f32>(1.0, 0.0), 0.0, 1.0),
    );
    let y_target = transform(
        camera.camera_from_raster,
        vec4<f32>(pixel + vec2<f32>(0.0, 1.0), 0.0, 1.0),
    );
    var rx_camera_origin = vec3<f32>(0.0);
    var ry_camera_origin = vec3<f32>(0.0);
    var rx_camera_direction = normalize(x_target.xyz);
    var ry_camera_direction = normalize(y_target.xyz);
    if (camera.camera_info.x > 0.0) {
        let lens = camera.camera_info.x * sample_uniform_disk_concentric(lens_sample);
        rx_camera_origin = vec3<f32>(lens, 0.0);
        ry_camera_origin = rx_camera_origin;
        let rx_focus = (camera.camera_info.y / rx_camera_direction.z) * rx_camera_direction;
        let ry_focus = (camera.camera_info.y / ry_camera_direction.z) * ry_camera_direction;
        rx_camera_direction = normalize(rx_focus - rx_camera_origin);
        ry_camera_direction = normalize(ry_focus - ry_camera_origin);
    }
    var ray_rx_origin = transform(camera.render_from_camera, vec4<f32>(rx_camera_origin, 1.0)).xyz;
    var ray_ry_origin = transform(camera.render_from_camera, vec4<f32>(ry_camera_origin, 1.0)).xyz;
    var ray_rx_direction = normalize(transform(camera.render_from_camera, vec4<f32>(rx_camera_direction, 0.0)).xyz);
    var ray_ry_direction = normalize(transform(camera.render_from_camera, vec4<f32>(ry_camera_direction, 0.0)).xyz);

    var color = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    for (var depth = 0u; depth <= camera.bvh_info.z; depth += 1u) {
    let origin = ray_origin;
    let direction = ray_direction;
    let rx_origin = ray_rx_origin;
    let ry_origin = ray_ry_origin;
    let rx_direction = ray_rx_direction;
    let ry_direction = ray_ry_direction;
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
                sample_float_texture(primitive.alpha, uv, vec4<f32>(0.0)),
                origin,
                direction,
            )) {
            rayQueryConfirmIntersection(&query);
        }
    }
    let intersection = rayQueryGetCommittedIntersection(&query);
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        if (depth == camera.bvh_info.z) {
            break;
        }
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
        let geometric_normal = normalize(cross(
            vertices[i1].position.xyz - vertices[i0].position.xyz,
            vertices[i2].position.xyz - vertices[i0].position.xyz,
        ));
        let transform_table = transforms[intersection.instance_custom_data];
        let position = transform(transform_table.render_from_object, vec4<f32>(object_position, 1.0)).xyz;
        let object_position_error = 4.172327e-7 * (
            abs(barycentrics.x * vertices[i0].position.xyz)
            + abs(barycentrics.y * vertices[i1].position.xyz)
            + abs(barycentrics.z * vertices[i2].position.xyz)
        );
        let position_error = transformed_position_error(
            transform_table.render_from_object,
            object_position,
            object_position_error,
        );
        let render_p0 = transform(transform_table.render_from_object, vertices[i0].position).xyz;
        let render_p1 = transform(transform_table.render_from_object, vertices[i1].position).xyz;
        let render_p2 = transform(transform_table.render_from_object, vertices[i2].position).xyz;
        var geometric_render_normal = normalize(cross(render_p1 - render_p0, render_p2 - render_p0));
        if ((primitive.flags & 1u) != 0u) {
            geometric_render_normal = -geometric_render_normal;
        }
        let material = materials[primitive.material];
        let interpolated_normal = vertices[i0].normal.xyz * barycentrics.x
            + vertices[i1].normal.xyz * barycentrics.y
            + vertices[i2].normal.xyz * barycentrics.z;
        let orientation_sign = select(1.0, -1.0, (primitive.flags & 1u) != 0u);
        let has_vertex_normal = dot(interpolated_normal, interpolated_normal) > 1.0e-20;
        var object_normal = geometric_normal;
        if (has_vertex_normal) {
            object_normal = normalize(interpolated_normal);
        }
        object_normal *= orientation_sign;
        let uv = vertices[i0].uv.xy * barycentrics.x
            + vertices[i1].uv.xy * barycentrics.y
            + vertices[i2].uv.xy * barycentrics.z;
        let object_dpdu = triangle_dpdu(
            vertices[i0].position.xyz,
            vertices[i1].position.xyz,
            vertices[i2].position.xyz,
            vertices[i0].uv.xy,
            vertices[i1].uv.xy,
            vertices[i2].uv.xy,
            object_normal,
        );
        let object_dpdv = triangle_dpdv(
            vertices[i0].position.xyz,
            vertices[i1].position.xyz,
            vertices[i2].position.xyz,
            vertices[i0].uv.xy,
            vertices[i1].uv.xy,
            vertices[i2].uv.xy,
            object_normal,
        );
        let object_dndu = orientation_sign * normal_derivative_u(
            vertices[i0].normal.xyz,
            vertices[i1].normal.xyz,
            vertices[i2].normal.xyz,
            vertices[i0].uv.xy,
            vertices[i1].uv.xy,
            vertices[i2].uv.xy,
        );
        let object_dndv = orientation_sign * normal_derivative_v(
            vertices[i0].normal.xyz,
            vertices[i1].normal.xyz,
            vertices[i2].normal.xyz,
            vertices[i0].uv.xy,
            vertices[i1].uv.xy,
            vertices[i2].uv.xy,
        );
        let object_tangent = vertices[i0].tangent.xyz * barycentrics.x
            + vertices[i1].tangent.xyz * barycentrics.y
            + vertices[i2].tangent.xyz * barycentrics.z;
        var normal = geometric_render_normal;
        if (has_vertex_normal) {
            normal = normalize(transform(transform_table.normal_from_object, vec4<f32>(object_normal, 0.0)).xyz);
            geometric_render_normal = select(
                -geometric_render_normal,
                geometric_render_normal,
                dot(geometric_render_normal, normal) >= 0.0,
            );
        }
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
                normal,
                object_dpdu,
                object_tangent,
                transform_table.render_from_object,
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
        let ray_sample = independent_ray_sample(sample_pixel, sample_index, depth);
        let direct_sample = ray_sample.direct;
        let light_count = arrayLength(&lights);
        let light_index = min(u32(direct_sample.light_selection * f32(light_count)), light_count - 1u);
        let light = lights[light_index];
        if (light.kind == 0u) {
            let to_light = light.position.xyz - position;
            let distance_squared = max(dot(to_light, to_light), 1.0e-8);
            let wi = normalize(to_light);
            let cosine = abs(dot(normal, wi));
            let same_hemisphere = dot(normal, -direction) * dot(normal, wi) > 0.0;
            let shadow_origin = offset_ray_origin(
                position,
                position_error,
                geometric_render_normal,
                to_light,
            );
            if (same_hemisphere && cosine > 0.0 && shadow_visible(
                shadow_origin,
                to_light,
                intersection.instance_custom_data,
                intersection.primitive_index,
            )) {
                color += throughput * reflectance * light.intensity.xyz * cosine
                    * f32(light_count) / (3.141592653589793 * distance_squared);
            }
        }
        let disk = sample_uniform_disk_concentric(ray_sample.indirect.direction);
        var local_wi = vec3<f32>(disk, sqrt(max(0.0, 1.0 - dot(disk, disk))));
        if (dot(normal, -direction) < 0.0) {
            local_wi.z = -local_wi.z;
        }
        let projected_dpdu = dpdu - normal * dot(normal, dpdu);
        let frame_x = normalize(select(
            coordinate_tangent(normal),
            projected_dpdu,
            dot(projected_dpdu, projected_dpdu) > 1.0e-20,
        ));
        let frame_y = cross(normal, frame_x);
        let next_direction = normalize(
            frame_x * local_wi.x + frame_y * local_wi.y + normal * local_wi.z,
        );
        throughput *= reflectance;
        if (depth >= 1u) {
            let maximum = max(throughput.x, max(throughput.y, throughput.z));
            if (maximum < 1.0) {
                let q = 1.0 - maximum;
                if (ray_sample.indirect.roulette < q) {
                    break;
                }
                throughput /= 1.0 - q;
            }
        }
        ray_origin = offset_ray_origin(
            position,
            position_error,
            geometric_render_normal,
            next_direction,
        );
        ray_direction = next_direction;
        ray_rx_origin = ray_origin;
        ray_ry_origin = ray_origin;
        ray_rx_direction = ray_direction;
        ray_ry_direction = ray_direction;
    } else {
        color += throughput * infinite_emission();
        break;
    }
    }

    return color;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let width = u32(camera.viewport.z);
    let height = u32(camera.viewport.w);
    if (global_id.x >= width || global_id.y >= height) {
        return;
    }
    let pixel = vec2<i32>(global_id.xy) + vec2<i32>(camera.viewport.xy);
    var accumulated = vec3<f32>(0.0);
    for (var local_sample = 0u; local_sample < camera.sampler_info.w; local_sample++) {
        let sample_index = camera.sampler_info.z + local_sample;
        let camera_sample = independent_camera_sample(pixel, sample_index);
        let filter_offset = mix(-camera.filter_info.xy, camera.filter_info.xy, camera_sample.filter_sample);
        let film_position = vec2<f32>(pixel) + filter_offset + vec2<f32>(0.5);
        let sample_time = mix(camera.camera_info.z, camera.camera_info.w, camera_sample.time);
        _ = sample_time;
        accumulated += render_sample(film_position, camera_sample.lens, pixel, sample_index);
    }
    output[global_id.y * width + global_id.x] = vec4<f32>(
        accumulated / f32(camera.sampler_info.w),
        1.0,
    );
}
