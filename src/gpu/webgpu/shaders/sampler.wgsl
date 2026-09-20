const SAMPLER_KIND_INDEPENDENT: u32 = 0u;
const SAMPLER_KIND_HALTON: u32 = 1u;
const SAMPLER_KIND_SOBOL: u32 = 2u;
const SAMPLER_KIND_PADDED_SOBOL: u32 = 3u;
const SAMPLER_KIND_Z_SOBOL: u32 = 4u;
const SAMPLER_KIND_PMJ02BN: u32 = 5u;
const SAMPLER_KIND_STRATIFIED: u32 = 6u;
const HALTON_RANDOMIZATION_NONE: u32 = 0u;
const HALTON_RANDOMIZATION_PERMUTE_DIGITS: u32 = 1u;
const SAMPLER_RANDOMIZATION_FAST_OWEN: u32 = 2u;
const SAMPLER_RANDOMIZATION_OWEN: u32 = 3u;
const HALTON_ONE_MINUS_EPSILON: f32 = 0.9999999403953552;

fn halton_base_scales() -> vec2<u32> { return sampler_params.payload[0].xy; }
fn halton_base_exponents() -> vec2<u32> { return sampler_params.payload[0].zw; }
fn halton_mult_inverse() -> vec2<u32> { return sampler_params.payload[1].xy; }
fn sobol_scale() -> u32 { return sampler_params.payload[0].x; }
fn sobol_log2_samples_per_pixel() -> u32 { return sampler_params.payload[0].y; }
fn zsobol_base4_digits() -> u32 { return sampler_params.payload[0].z; }
fn sobol_table_offset() -> u32 { return sampler_params.payload[0].w; }
fn sobol_vdc_offset() -> u32 { return sampler_params.payload[1].x; }
fn sobol_vdc_inverse_offset() -> u32 { return sampler_params.payload[1].y; }
fn pmj_table_offset() -> u32 { return sampler_params.payload[0].x; }
fn pmj_pixel_table_offset() -> u32 { return sampler_params.payload[0].y; }
fn pmj_blue_noise_offset() -> u32 { return sampler_params.payload[0].z; }
fn pmj_pixel_tile_size() -> u32 { return sampler_params.payload[0].w; }
fn stratified_x_samples() -> u32 { return sampler_params.payload[0].x; }
fn stratified_y_samples() -> u32 { return sampler_params.payload[0].y; }
fn stratified_jitter() -> bool { return sampler_params.payload[0].z != 0u; }

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low = a.x + b.x;
    return vec2<u32>(low, a.y + b.y + select(0u, 1u, low < a.x));
}

fn u32_mul_wide(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let w0 = a0 * b0;
    let t = a1 * b0 + (w0 >> 16u);
    var w1 = t & 0xffffu;
    let w2 = t >> 16u;
    w1 += a0 * b1;
    return vec2<u32>((w1 << 16u) | (w0 & 0xffffu), a1 * b1 + w2 + (w1 >> 16u));
}

fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low_product = u32_mul_wide(a.x, b.x);
    return vec2<u32>(low_product.x, low_product.y + a.x * b.y + a.y * b.x);
}

fn u64_shr(a: vec2<u32>, shift: u32) -> vec2<u32> {
    if (shift == 0u) { return a; }
    if (shift < 32u) {
        return vec2<u32>((a.x >> shift) | (a.y << (32u - shift)), a.y >> shift);
    }
    if (shift < 64u) { return vec2<u32>(a.y >> (shift - 32u), 0u); }
    return vec2<u32>(0u);
}

fn u64_shl(a: vec2<u32>, shift: u32) -> vec2<u32> {
    if (shift == 0u) { return a; }
    if (shift < 32u) {
        return vec2<u32>(a.x << shift, (a.y << shift) | (a.x >> (32u - shift)));
    }
    if (shift < 64u) { return vec2<u32>(0u, a.x << (shift - 32u)); }
    return vec2<u32>(0u);
}

fn u64_is_zero(a: vec2<u32>) -> bool {
    return (a.x | a.y) == 0u;
}

fn u64_mod_small(value: vec2<u32>, divisor: u32) -> u32 {
    let two32_mod = (0xffffffffu % divisor + 1u) % divisor;
    return ((value.x % divisor) + (value.y % divisor) * two32_mod) % divisor;
}

fn mix_bits_64(value_in: vec2<u32>) -> vec2<u32> {
    var value = value_in;
    value ^= u64_shr(value, 31u);
    value = u64_mul(value, vec2<u32>(0x728ea185u, 0x7fb5d329u));
    value ^= u64_shr(value, 27u);
    value = u64_mul(value, vec2<u32>(0xbc2dd44du, 0x81dadef4u));
    value ^= u64_shr(value, 33u);
    return value;
}

fn murmur_mix_block(hash_in: vec2<u32>, block_in: vec2<u32>) -> vec2<u32> {
    let multiplier = vec2<u32>(0x5bd1e995u, 0xc6a4a793u);
    var block = u64_mul(block_in, multiplier);
    block ^= u64_shr(block, 47u);
    block = u64_mul(block, multiplier);
    return u64_mul(hash_in ^ block, multiplier);
}

fn murmur_finish(hash_in: vec2<u32>) -> vec2<u32> {
    let multiplier = vec2<u32>(0x5bd1e995u, 0xc6a4a793u);
    var hash = hash_in ^ u64_shr(hash_in, 47u);
    hash = u64_mul(hash, multiplier);
    return hash ^ u64_shr(hash, 47u);
}

fn murmur_hash_8(block: vec2<u32>) -> vec2<u32> {
    let multiplier = vec2<u32>(0x5bd1e995u, 0xc6a4a793u);
    return murmur_finish(murmur_mix_block(u64_mul(vec2<u32>(8u, 0u), multiplier), block));
}

fn murmur_hash_12(block: vec2<u32>, tail: u32) -> vec2<u32> {
    let multiplier = vec2<u32>(0x5bd1e995u, 0xc6a4a793u);
    var hash = murmur_mix_block(u64_mul(vec2<u32>(12u, 0u), multiplier), block);
    hash = u64_mul(hash ^ vec2<u32>(tail, 0u), multiplier);
    return murmur_finish(hash);
}

fn murmur_hash_16(first: vec2<u32>, second: vec2<u32>) -> vec2<u32> {
    let multiplier = vec2<u32>(0x5bd1e995u, 0xc6a4a793u);
    var hash = murmur_mix_block(u64_mul(vec2<u32>(16u, 0u), multiplier), first);
    hash = murmur_mix_block(hash, second);
    return murmur_finish(hash);
}

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

fn sampler_table_u64(address: u32) -> vec2<u32> {
    return vec2<u32>(sampler_table_word(address), sampler_table_word(address + 1u));
}

fn u64_bit(value: vec2<u32>, bit: u32) -> u32 {
    return select((value.x >> bit) & 1u, (value.y >> (bit - 32u)) & 1u, bit >= 32u);
}

fn permutation_element(index_in: u32, length: u32, permutation: u32) -> u32 {
    var mask = length - 1u;
    mask |= mask >> 1u;
    mask |= mask >> 2u;
    mask |= mask >> 4u;
    mask |= mask >> 8u;
    mask |= mask >> 16u;
    var index = index_in;
    loop {
        index ^= permutation;
        index *= 0xe170893du;
        index ^= permutation >> 16u;
        index ^= (index & mask) >> 4u;
        index ^= permutation >> 8u;
        index *= 0x0929eb3fu;
        index ^= permutation >> 23u;
        index ^= (index & mask) >> 1u;
        index *= 1u | (permutation >> 27u);
        index *= 0x6935fa69u;
        index ^= (index & mask) >> 11u;
        index *= 0x74dcb303u;
        index ^= (index & mask) >> 2u;
        index *= 0x9e501cc3u;
        index ^= (index & mask) >> 2u;
        index *= 0xc860a3dfu;
        index &= mask;
        index ^= index >> 5u;
        if (index < length) { break; }
    }
    return (index + permutation) % length;
}

fn fast_owen_scramble(value_in: u32, seed: u32) -> u32 {
    var value = reverseBits(value_in);
    value ^= value * 0x3d20adeau;
    value += seed;
    value *= (seed >> 16u) | 1u;
    value ^= value * 0x05526c56u;
    value ^= value * 0x53a22864u;
    return reverseBits(value);
}

fn owen_scramble(value_in: u32, seed: u32) -> u32 {
    var value = value_in;
    if ((seed & 1u) != 0u) { value ^= 0x80000000u; }
    for (var bit = 1u; bit < 32u; bit++) {
        let mask = 0xffffffffu << (32u - bit);
        if ((mix_bits_64(vec2<u32>((value & mask) ^ seed, 0u)).x & (1u << bit)) != 0u) {
            value ^= 1u << (31u - bit);
        }
    }
    return value;
}

fn sobol_bits(index: vec2<u32>, dimension: u32, scramble: u32) -> u32 {
    var value = scramble;
    for (var bit = 0u; bit < 52u; bit++) {
        if (u64_bit(index, bit) != 0u) {
            value ^= sampler_table_word(sobol_table_offset() + dimension * 52u + bit);
        }
    }
    return value;
}

fn sobol_raw_sample(index: vec2<u32>, dimension: u32, scramble: u32) -> f32 {
    return min(f32(sobol_bits(index, dimension, scramble)) * 2.3283064365386963e-10, HALTON_ONE_MINUS_EPSILON);
}

fn sobol_sample(index: vec2<u32>, dimension: u32, hash: u32) -> f32 {
    var value = sobol_bits(index, dimension, select(0u, hash, sampler_params.randomization == HALTON_RANDOMIZATION_PERMUTE_DIGITS));
    if (sampler_params.randomization == SAMPLER_RANDOMIZATION_FAST_OWEN) {
        value = fast_owen_scramble(value, hash);
    } else if (sampler_params.randomization == SAMPLER_RANDOMIZATION_OWEN) {
        value = owen_scramble(value, hash);
    }
    return min(f32(value) * 2.3283064365386963e-10, HALTON_ONE_MINUS_EPSILON);
}

fn sobol_dimension_hash(dimension: u32) -> u32 {
    return mix_bits_64(vec2<u32>(sampler_params.seed, dimension)).x;
}

fn sobol_interval_index(pixel: vec2<u32>, sample_index: u32) -> vec2<u32> {
    let log2_scale = 31u - countLeadingZeros(sobol_scale());
    if (log2_scale == 0u) { return vec2<u32>(sample_index, 0u); }
    let shift = 2u * log2_scale;
    var frame = vec2<u32>(sample_index, 0u);
    var index = u64_shl(frame, shift);
    var delta = vec2<u32>(0u);
    for (var bit = 0u; bit < 52u; bit++) {
        if (u64_bit(frame, bit) != 0u) {
            delta ^= sampler_table_u64(sobol_vdc_offset() + bit * 2u);
        }
    }
    var packed_pixel = u64_shl(vec2<u32>(pixel.x, 0u), log2_scale);
    packed_pixel |= vec2<u32>(pixel.y, 0u);
    packed_pixel ^= delta;
    for (var bit = 0u; bit < 52u; bit++) {
        if (u64_bit(packed_pixel, bit) != 0u) {
            index ^= sampler_table_u64(sobol_vdc_inverse_offset() + bit * 2u);
        }
    }
    return index;
}

fn padded_sobol_index(pixel: vec2<u32>, dimension: u32) -> vec2<u32> {
    let hash = murmur_hash_16(pixel, vec2<u32>(dimension, sampler_params.seed));
    return vec2<u32>(permutation_element(viewport.sample_index, sampler_params.samples_per_pixel, hash.x), 0u);
}

fn padded_sobol_1d(pixel: vec2<u32>, dimension: u32) -> f32 {
    let hash = murmur_hash_16(pixel, vec2<u32>(dimension, sampler_params.seed));
    let index = vec2<u32>(permutation_element(viewport.sample_index, sampler_params.samples_per_pixel, hash.x), 0u);
    return sobol_sample(index, 0u, hash.y);
}

fn padded_sobol_2d(pixel: vec2<u32>, dimension: u32) -> vec2<f32> {
    let hash = murmur_hash_16(pixel, vec2<u32>(dimension, sampler_params.seed));
    let index = vec2<u32>(permutation_element(viewport.sample_index, sampler_params.samples_per_pixel, hash.x), 0u);
    return vec2<f32>(sobol_sample(index, 0u, hash.x), sobol_sample(index, 1u, hash.y));
}

fn morton_encode(pixel: vec2<u32>) -> vec2<u32> {
    var result = vec2<u32>(0u);
    for (var bit = 0u; bit < 32u; bit++) {
        let xb = (pixel.x >> bit) & 1u;
        let yb = (pixel.y >> bit) & 1u;
        if (2u * bit < 32u) {
            result.x |= xb << (2u * bit);
            result.x |= yb << (2u * bit + 1u);
        } else {
            result.y |= xb << (2u * bit - 32u);
            result.y |= yb << (2u * bit + 1u - 32u);
        }
    }
    return result;
}

fn zsobol_permutation(permutation: u32, digit: u32) -> u32 {
    let packed = array<u32, 24>(0xe4u,0xb4u,0xd8u,0x78u,0x6cu,0x9cu,0xe1u,0xb1u,0xc9u,0x39u,0x2du,0x8du,0xc6u,0x36u,0xd2u,0x72u,0x4eu,0x1eu,0x27u,0x87u,0x1bu,0x4bu,0x63u,0x93u);
    return (packed[permutation] >> (digit * 2u)) & 3u;
}

fn zsobol_index(pixel: vec2<u32>, dimension: u32) -> vec2<u32> {
    let morton = u64_add(u64_shl(morton_encode(pixel), sobol_log2_samples_per_pixel()), vec2<u32>(viewport.sample_index, 0u));
    var sample_index = vec2<u32>(0u);
    let odd = (sobol_log2_samples_per_pixel() & 1u) != 0u;
    let last_digit = select(0u, 1u, odd);
    var i = zsobol_base4_digits();
    loop {
        if (i == last_digit) { break; }
        i--;
        let shift = 2u * i - select(0u, 1u, odd);
        let digit = u64_shr(morton, shift).x & 3u;
        let higher = u64_shr(morton, shift + 2u);
        let mixed = mix_bits_64(higher ^ u64_mul(vec2<u32>(0x55555555u, 0u), vec2<u32>(dimension, 0u)));
        let permutation = u64_mod_small(u64_shr(mixed, 24u), 24u);
        sample_index |= u64_shl(vec2<u32>(zsobol_permutation(permutation, digit), 0u), shift);
    }
    if (odd) {
        let mixed = mix_bits_64(u64_shr(morton, 1u) ^ u64_mul(vec2<u32>(0x55555555u, 0u), vec2<u32>(dimension, 0u)));
        sample_index.x |= (morton.x & 1u) ^ (mixed.x & 1u);
    }
    return sample_index;
}

fn zsobol_1d(pixel: vec2<u32>, dimension: u32) -> f32 {
    return sobol_sample(zsobol_index(pixel, dimension), 0u, murmur_hash_8(vec2<u32>(dimension + 1u, sampler_params.seed)).x);
}

fn zsobol_2d(pixel: vec2<u32>, dimension: u32) -> vec2<f32> {
    let index = zsobol_index(pixel, dimension);
    let hash = murmur_hash_8(vec2<u32>(dimension + 2u, sampler_params.seed));
    return vec2<f32>(sobol_sample(index, 0u, hash.x), sobol_sample(index, 1u, hash.y));
}

fn pmj_blue_noise(dimension: u32, pixel: vec2<u32>) -> f32 {
    let tex = dimension % 48u;
    let x = pixel.x % 128u;
    let y = pixel.y % 128u;
    let element = (tex * 128u + x) * 128u + y;
    let packed = sampler_table_word(pmj_blue_noise_offset() + element / 2u);
    let value = select(packed & 0xffffu, packed >> 16u, (element & 1u) != 0u);
    return f32(value) / 65535.0;
}

fn pmj_1d(pixel: vec2<u32>, dimension: u32) -> f32 {
    let hash = murmur_hash_16(pixel, vec2<u32>(dimension, sampler_params.seed));
    let index = permutation_element(viewport.sample_index, sampler_params.samples_per_pixel, hash.x);
    return min((f32(index) + pmj_blue_noise(dimension, pixel)) / f32(sampler_params.samples_per_pixel), HALTON_ONE_MINUS_EPSILON);
}

fn pmj_2d(pixel: vec2<u32>, dimension: u32) -> vec2<f32> {
    let instance = dimension / 2u;
    var index = viewport.sample_index;
    if (instance >= 5u) {
        index = permutation_element(index, sampler_params.samples_per_pixel, murmur_hash_16(pixel, vec2<u32>(dimension, sampler_params.seed)).x);
    }
    let address = pmj_table_offset() + ((instance % 5u) * 65536u + (index % 65536u)) * 2u;
    var value = vec2<f32>(f32(sampler_table_word(address)), f32(sampler_table_word(address + 1u))) * 2.3283064365386963e-10;
    value = fract(value + vec2<f32>(pmj_blue_noise(dimension, pixel), pmj_blue_noise(dimension + 1u, pixel)));
    return min(value, vec2<f32>(HALTON_ONE_MINUS_EPSILON));
}

fn pcg_step(state_in: vec2<u32>, increment: vec2<u32>) -> vec3<u32> {
    let state = u64_add(u64_mul(state_in, vec2<u32>(0x4c957f2du, 0x5851f42du)), increment);
    let shifted = u64_shr(u64_shr(state_in, 18u) ^ state_in, 27u).x;
    let rotation = u64_shr(state_in, 59u).x;
    let rotated = (shifted >> rotation) | (shifted << ((0u - rotation) & 31u));
    return vec3<u32>(state, rotated);
}

fn pcg_advance(state_in: vec2<u32>, increment: vec2<u32>, delta_in: vec2<u32>) -> vec2<u32> {
    var current_multiplier = vec2<u32>(0x4c957f2du, 0x5851f42du);
    var current_increment = increment;
    var accumulator_multiplier = vec2<u32>(1u, 0u);
    var accumulator_increment = vec2<u32>(0u);
    var delta = delta_in;
    loop {
        if (u64_is_zero(delta)) { break; }
        if ((delta.x & 1u) != 0u) {
            accumulator_multiplier = u64_mul(accumulator_multiplier, current_multiplier);
            accumulator_increment = u64_add(u64_mul(accumulator_increment, current_multiplier), current_increment);
        }
        current_increment = u64_mul(u64_add(current_multiplier, vec2<u32>(1u, 0u)), current_increment);
        current_multiplier = u64_mul(current_multiplier, current_multiplier);
        delta = u64_shr(delta, 1u);
    }
    return u64_add(u64_mul(accumulator_multiplier, state_in), accumulator_increment);
}

fn independent_1d(pixel: vec2<u32>, dimension: u32) -> f32 {
    let sequence = murmur_hash_12(pixel, sampler_params.seed);
    let increment = u64_add(u64_shl(sequence, 1u), vec2<u32>(1u, 0u));
    var state = vec2<u32>(0u);
    state = pcg_step(state, increment).xy;
    state = u64_add(state, mix_bits_64(sequence));
    state = pcg_step(state, increment).xy;
    state = pcg_advance(state, increment, vec2<u32>(dimension + (viewport.sample_index << 16u), viewport.sample_index >> 16u));
    return min(f32(pcg_step(state, increment).z) * 2.3283064365386963e-10, HALTON_ONE_MINUS_EPSILON);
}

fn independent_2d(pixel: vec2<u32>, dimension: u32) -> vec2<f32> {
    let sequence = murmur_hash_12(pixel, sampler_params.seed);
    let increment = u64_add(u64_shl(sequence, 1u), vec2<u32>(1u, 0u));
    var state = vec2<u32>(0u);
    state = pcg_step(state, increment).xy;
    state = u64_add(state, mix_bits_64(sequence));
    state = pcg_step(state, increment).xy;
    state = pcg_advance(state, increment, vec2<u32>(dimension + (viewport.sample_index << 16u), viewport.sample_index >> 16u));
    let first = pcg_step(state, increment);
    let second = pcg_step(first.xy, increment);
    return min(vec2<f32>(f32(first.z), f32(second.z)) * 2.3283064365386963e-10, vec2<f32>(HALTON_ONE_MINUS_EPSILON));
}

fn stratified_1d(pixel: vec2<u32>, dimension: u32) -> f32 {
    let hash = murmur_hash_16(pixel, vec2<u32>(dimension, sampler_params.seed));
    let stratum = permutation_element(viewport.sample_index, sampler_params.samples_per_pixel, hash.x);
    let delta = select(0.5, independent_1d(pixel, dimension), stratified_jitter());
    return (f32(stratum) + delta) / f32(sampler_params.samples_per_pixel);
}

fn stratified_2d(pixel: vec2<u32>, dimension: u32) -> vec2<f32> {
    let hash = murmur_hash_16(pixel, vec2<u32>(dimension, sampler_params.seed));
    let stratum = permutation_element(viewport.sample_index, sampler_params.samples_per_pixel, hash.x);
    let delta = select(vec2<f32>(0.5), independent_2d(pixel, dimension), stratified_jitter());
    let x = stratum % stratified_x_samples();
    let y = stratum / stratified_x_samples();
    return (vec2<f32>(f32(x), f32(y)) + delta)
        / vec2<f32>(f32(stratified_x_samples()), f32(stratified_y_samples()));
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
    let base_scales = halton_base_scales();
    let base_exponents = halton_base_exponents();
    let mult_inverse = halton_mult_inverse();
    let stride = base_scales.x * base_scales.y;
    var index = 0u;
    if (stride > 1u) {
        let pm = pixel % vec2<u32>(128u);
        let offsets = vec2<u32>(
            inverse_radical_inverse(2u, pm.x, base_exponents.x),
            inverse_radical_inverse(3u, pm.y, base_exponents.y),
        );
        index += offsets.x * (stride / base_scales.x) * mult_inverse.x;
        index += offsets.y * (stride / base_scales.y) * mult_inverse.y;
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

fn halton_owen_radical_inverse(dimension: u32, index_in: u32) -> f32 {
    let base = sampler_table_word(dimension * 4u);
    let inverse_base = 1.0 / f32(base);
    let hash = mix_bits_64(vec2<u32>(1u + (dimension << 4u), 0u)).x;
    var index = index_in;
    var inverse_base_n = 1.0;
    var reversed_digits = vec2<u32>(0u);
    loop {
        if (!(1.0 - inverse_base_n < 1.0)) { break; }
        let next = index / base;
        let digit = index - next * base;
        let digit_hash = mix_bits_64(vec2<u32>(hash ^ reversed_digits.x, 0u)).x;
        let permuted = permutation_element(digit, base, digit_hash);
        reversed_digits = u64_add(u64_mul(reversed_digits, vec2<u32>(base, 0u)), vec2<u32>(permuted, 0u));
        inverse_base_n *= inverse_base;
        index = next;
    }
    return min(inverse_base_n * (f32(reversed_digits.y) * 4294967296.0 + f32(reversed_digits.x)), HALTON_ONE_MINUS_EPSILON);
}

fn halton_radical_inverse(dimension: u32, index: u32) -> f32 {
    if (sampler_params.randomization == SAMPLER_RANDOMIZATION_OWEN) {
        return halton_owen_radical_inverse(dimension, index);
    }
    return halton_radical_inverse_impl(dimension, index, true);
}

fn halton_pixel_radical_inverse(dimension: u32, index: u32) -> f32 {
    return halton_radical_inverse_impl(dimension, index, false);
}

fn sampler_get_1d(pixel_index: u32, dimension_in: u32) -> f32 {
    let pixel = vec2<u32>(pixel_index % viewport.width, pixel_index / viewport.width);
    if (sampler_params.kind == SAMPLER_KIND_HALTON) {
        let dimension = select(dimension_in, 2u, dimension_in == 0u);
        return halton_radical_inverse(dimension, halton_index(pixel, viewport.sample_index));
    }
    if (sampler_params.kind == SAMPLER_KIND_SOBOL) {
        let dimension = select(dimension_in, 2u, dimension_in == 0u);
        let index = sobol_interval_index(pixel, viewport.sample_index);
        return sobol_sample(index, dimension, sobol_dimension_hash(dimension));
    }
    if (sampler_params.kind == SAMPLER_KIND_PADDED_SOBOL) {
        return padded_sobol_1d(pixel, dimension_in);
    }
    if (sampler_params.kind == SAMPLER_KIND_Z_SOBOL) {
        return zsobol_1d(pixel, dimension_in);
    }
    if (sampler_params.kind == SAMPLER_KIND_PMJ02BN) {
        return pmj_1d(pixel, max(2u, dimension_in));
    }
    if (sampler_params.kind == SAMPLER_KIND_STRATIFIED) {
        return stratified_1d(pixel, dimension_in);
    }
    return independent_1d(pixel, dimension_in);
}

fn sampler_get_2d(pixel_index: u32, dimension: u32) -> vec2<f32> {
    let pixel = vec2<u32>(pixel_index % viewport.width, pixel_index / viewport.width);
    if (sampler_params.kind == SAMPLER_KIND_PADDED_SOBOL) {
        return padded_sobol_2d(pixel, dimension);
    }
    if (sampler_params.kind == SAMPLER_KIND_Z_SOBOL) {
        return zsobol_2d(pixel, dimension);
    }
    if (sampler_params.kind == SAMPLER_KIND_PMJ02BN) {
        return pmj_2d(pixel, max(2u, dimension));
    }
    if (sampler_params.kind == SAMPLER_KIND_STRATIFIED) {
        return stratified_2d(pixel, dimension);
    }
    if (sampler_params.kind == SAMPLER_KIND_INDEPENDENT) {
        return independent_2d(pixel, dimension);
    }
    return vec2<f32>(
        sampler_get_1d(pixel_index, dimension),
        sampler_get_1d(pixel_index, dimension + 1u),
    );
}

fn sampler_get_pixel_2d(pixel_index: u32) -> vec2<f32> {
    let pixel = vec2<u32>(pixel_index % viewport.width, pixel_index / viewport.width);
    if (sampler_params.kind == SAMPLER_KIND_HALTON) {
        let index = halton_index(pixel, viewport.sample_index);
        return vec2<f32>(
            halton_pixel_radical_inverse(0u, index >> halton_base_exponents().x),
            halton_pixel_radical_inverse(1u, index / halton_base_scales().y),
        );
    }
    if (sampler_params.kind == SAMPLER_KIND_SOBOL) {
        let index = sobol_interval_index(pixel, viewport.sample_index);
        let raw = vec2<f32>(sobol_raw_sample(index, 0u, 0u), sobol_raw_sample(index, 1u, 0u));
        return clamp(raw * f32(sobol_scale()) - vec2<f32>(pixel), vec2<f32>(0.0), vec2<f32>(HALTON_ONE_MINUS_EPSILON));
    }
    if (sampler_params.kind == SAMPLER_KIND_PMJ02BN) {
        let tile_size = pmj_pixel_tile_size();
        let x = pixel.x % tile_size;
        let y = pixel.y % tile_size;
        let address = pmj_pixel_table_offset() + ((x + y * tile_size) * sampler_params.samples_per_pixel + viewport.sample_index) * 2u;
        return vec2<f32>(bitcast<f32>(sampler_table_word(address)), bitcast<f32>(sampler_table_word(address + 1u)));
    }
    return sampler_get_2d(pixel_index, 1u);
}
