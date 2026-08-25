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
