fn find_area_light(primitive: u32, triangle: u32) -> u32 {
    for (var light_index = 0u; light_index < light_source_count(); light_index += 1u) {
        let light = lights[light_index];
        if (light.kind == 2u) {
            for (var geometry_index = 0u; geometry_index < light.triangle; geometry_index += 1u) {
                let geometry = area_light_geometry(light, geometry_index);
                if (geometry.primitive == primitive && geometry.triangle == triangle) {
                    return light_index;
                }
            }
        }
    }
    return 0xffffffffu;
}

fn murmur_hash_float_position(position: vec3<f32>) -> f32 {
    let multiplier = vec2<u32>(0x5bd1e995u, 0xc6a4a793u);
    var hash = u64_mul(vec2<u32>(12u, 0u), multiplier);
    var key = vec2<u32>(bitcast<u32>(position.x), bitcast<u32>(position.y));
    key = u64_mul(key, multiplier);
    key = key ^ u64_shift_right_47(key);
    key = u64_mul(key, multiplier);
    hash = u64_mul(hash ^ key, multiplier);
    hash = hash ^ vec2<u32>(bitcast<u32>(position.z), 0u);
    hash = u64_mul(hash, multiplier);
    hash = hash ^ u64_shift_right_47(hash);
    hash = u64_mul(hash, multiplier);
    hash = hash ^ u64_shift_right_47(hash);
    return f32(hash.x) * 2.3283064365386963e-10;
}

fn area_light_alpha_accept(geometry: Light, uv: vec2<f32>, position: vec3<f32>) -> bool {
    if ((geometry.flags & 2u) != 0u) {
        return true;
    }
    let primitive = primitives[geometry.primitive];
    if (primitive.alpha == 0xffffffffu) {
        return true;
    }
    let alpha = sample_float_texture(primitive.alpha, uv, vec4<f32>(0.0));
    if (alpha >= 1.0) {
        return true;
    }
    if (alpha <= 0.0) {
        return false;
    }
    return murmur_hash_float_position(position) <= alpha;
}

fn infinite_emission() -> vec3<f32> {
    var emission = vec3<f32>(0.0);
    for (var light_index = 0u; light_index < light_source_count(); light_index += 1u) {
        let light = lights[light_index];
        if (light.kind == 1u) {
            emission += light.intensity.xyz;
        }
    }
    return emission;
}
