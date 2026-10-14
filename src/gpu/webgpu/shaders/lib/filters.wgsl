// pbrt-v4 filters.h: Filter::Sample and FilterSampler::Sample.
fn sample_tent(u: f32, radius: f32) -> f32 {
    if (u < 0.5) { return radius * (sqrt(2.0 * u) - 1.0); }
    return radius * (1.0 - sqrt(2.0 - 2.0 * u));
}

// FindInterval(cdf[i] <= u), including equal CDF entries.
fn filter_find_interval(offset: u32, count: u32, u: f32) -> u32 {
    var first = 0u;
    var size = count + 1u;
    while (size > 0u) {
        let half = size / 2u;
        let middle = first + half;
        if (filter_data.data[offset + middle] <= bitcast<u32>(u)) {
            first = middle + 1u;
            size -= half + 1u;
        } else { size = half; }
    }
    return min(max(first, 1u) - 1u, count - 1u);
}

// Scale nonnegative CDF values by a common power of two before subtraction.
// Integer mantissas also preserve subnormals that WGSL may otherwise flush.
fn filter_scaled_cdf(bits: u32, exponent: u32) -> f32 {
    let e = bits >> 23u;
    let mantissa = (bits & 0x7fffffu) | select(0u, 0x800000u, e != 0u);
    return f32(mantissa) * exp2(f32(i32(max(e, 1u)) - i32(exponent)));
}

fn filter_remap(offset: u32, cell: u32, u: f32) -> f32 {
    let lower_bits = filter_data.data[offset + cell];
    let upper_bits = filter_data.data[offset + cell + 1u];
    let exponent = max(upper_bits >> 23u, 1u);
    let lower = filter_scaled_cdf(lower_bits, exponent);
    let upper = filter_scaled_cdf(upper_bits, exponent);
    var du = filter_scaled_cdf(bitcast<u32>(u), exponent) - lower;
    if (upper > lower) { du /= upper - lower; }
    return du;
}

fn sample_filter(u: vec2<f32>) -> FilterSample {
    let radius = filter_data.radius;
    if (filter_data.kind == 0u) {
        return FilterSample((1.0 - u) * (-radius) + u * radius, 1.0);
    }
    if (filter_data.kind == 1u) {
        return FilterSample(vec2<f32>(sample_tent(u.x, radius.x), sample_tent(u.y, radius.y)), 1.0);
    }
    let y = filter_find_interval(filter_data.marginal_offset, filter_data.ny, u.y);
    let row = filter_data.conditional_offset + y * (filter_data.nx + 1u);
    let x = filter_find_interval(row, filter_data.nx, u.x);
    let unit = vec2<f32>(
        (f32(x) + filter_remap(row, x, u.x)) / f32(filter_data.nx),
        (f32(y) + filter_remap(filter_data.marginal_offset, y, u.y)) / f32(filter_data.ny),
    );
    let p = (1.0 - unit) * (-radius) + unit * radius;
    let value_bits = filter_data.data[filter_data.values_offset + y * filter_data.nx + x];
    // pdf = abs(f) / integral, hence f / pdf = sign(f) * integral.
    // Read the sign bit directly so negative subnormal cells keep their sign.
    let weight = select(filter_data.absolute_integral, -filter_data.absolute_integral,
        (value_bits & 0x80000000u) != 0u);
    return FilterSample(p, weight);
}
