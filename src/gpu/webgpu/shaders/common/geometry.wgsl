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
