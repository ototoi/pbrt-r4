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
