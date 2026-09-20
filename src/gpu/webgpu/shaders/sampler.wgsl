const SAMPLER_KIND_INDEPENDENT: u32 = 0u;
const SAMPLER_KIND_HALTON: u32 = 1u;
const HALTON_RANDOMIZATION_NONE: u32 = 0u;
const HALTON_RANDOMIZATION_PERMUTE_DIGITS: u32 = 1u;
const HALTON_ONE_MINUS_EPSILON: f32 = 0.9999999403953552;

fn sampler_table_word(address: u32) -> u32 {
    let texel = address / 4u;
    let channel = address % 4u;
    let value = textureLoad(
        sampler_table,
        vec2<i32>(i32(texel % sampler_params.table_width), i32(texel / sampler_params.table_width)),
        0,
    );
    return value[channel];
}

fn inverse_radical_inverse(base: u32, inverse_in: u32, digit_count: u32) -> u32 {
    var inverse = inverse_in;
    var index = 0u;
    for (var digit = 0u; digit < digit_count; digit++) {
        let value = inverse % base;
        inverse /= base;
        index = index * base + value;
    }
    return index;
}

fn halton_index(pixel: vec2<u32>, sample_index: u32) -> u32 {
    let stride = sampler_params.base_scales.x * sampler_params.base_scales.y;
    var index = 0u;
    if (stride > 1u) {
        let pm = pixel % vec2<u32>(128u);
        let offsets = vec2<u32>(
            inverse_radical_inverse(2u, pm.x, sampler_params.base_exponents.x),
            inverse_radical_inverse(3u, pm.y, sampler_params.base_exponents.y),
        );
        index += offsets.x * (stride / sampler_params.base_scales.x)
            * sampler_params.mult_inverse.x;
        index += offsets.y * (stride / sampler_params.base_scales.y)
            * sampler_params.mult_inverse.y;
        index %= stride;
    }
    return index + sample_index * stride;
}

fn halton_radical_inverse_impl(dimension: u32, index_in: u32, randomize: bool) -> f32 {
    if (dimension >= sampler_params.dimension_count) {
        set_render_error();
        return 0.0;
    }
    let header = dimension * 4u;
    let base = sampler_table_word(header);
    if (dimension == 0u) {
        return min(f32(reverseBits(index_in)) * 2.3283064365386963e-10, HALTON_ONE_MINUS_EPSILON);
    }
    let inv_base = 1.0 / f32(base);
    var inverse = index_in;
    var inv_base_m = 1.0;
    var result = 0.0;
    for (var digit_index = 0u; digit_index < 32u; digit_index++) {
        if (!(1.0 - f32(base - 1u) * inv_base_m < 1.0)) { break; }
        let next = inverse / base;
        var digit_value = inverse - next * base;
        if (randomize && sampler_params.randomization == HALTON_RANDOMIZATION_PERMUTE_DIGITS) {
            let digit_count = sampler_table_word(header + 1u);
            if (digit_index >= digit_count) {
                set_render_error();
                return 0.0;
            }
            let permutation_offset = sampler_table_word(header + 2u);
            digit_value = sampler_table_word(permutation_offset + digit_index * base + digit_value);
        }
        inv_base_m *= inv_base;
        result += f32(digit_value) * inv_base_m;
        inverse = next;
    }
    return min(result, HALTON_ONE_MINUS_EPSILON);
}

fn halton_radical_inverse(dimension: u32, index: u32) -> f32 {
    return halton_radical_inverse_impl(dimension, index, true);
}

fn halton_pixel_radical_inverse(dimension: u32, index: u32) -> f32 {
    return halton_radical_inverse_impl(dimension, index, false);
}

fn sampler_get_1d(pixel_index: u32, dimension_in: u32) -> f32 {
    if (sampler_params.kind == SAMPLER_KIND_HALTON) {
        let pixel = vec2<u32>(pixel_index % viewport.width, pixel_index / viewport.width);
        let dimension = select(dimension_in, 2u, dimension_in == 0u);
        return halton_radical_inverse(dimension, halton_index(pixel, viewport.sample_index));
    }
    return random01(pixel_index, dimension_in, 0u);
}

fn sampler_get_2d(pixel_index: u32, dimension: u32) -> vec2<f32> {
    return vec2<f32>(
        sampler_get_1d(pixel_index, dimension),
        sampler_get_1d(pixel_index, dimension + 1u),
    );
}

fn sampler_get_pixel_2d(pixel_index: u32) -> vec2<f32> {
    if (sampler_params.kind == SAMPLER_KIND_HALTON) {
        let pixel = vec2<u32>(pixel_index % viewport.width, pixel_index / viewport.width);
        let index = halton_index(pixel, viewport.sample_index);
        return vec2<f32>(
            halton_pixel_radical_inverse(0u, index >> sampler_params.base_exponents.x),
            halton_pixel_radical_inverse(1u, index / sampler_params.base_scales.y),
        );
    }
    return sampler_get_2d(pixel_index, 1u);
}
