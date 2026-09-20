fn measured_scalar(address: u32) -> f32 {
    let width = material_table._reserved3;
    let height = material_table._reserved4;
    let page_capacity = width * height * 4u;
    let page = address / page_capacity;
    if (page >= material_table._reserved5) {
        set_render_error();
        return 0.0;
    }
    let local = address - page * page_capacity;
    let texel = local / 4u;
    let value = textureLoad(
        texture_images[material_table._reserved2 + page],
        vec2<i32>(i32(texel % width), i32(texel / width)),
        0,
    );
    let lane = local & 3u;
    if (lane == 0u) { return value.x; }
    if (lane == 1u) { return value.y; }
    if (lane == 2u) { return value.z; }
    return value.w;
}

fn measured_parameter_lookup(table: MeasuredTableRecord, params: vec3<f32>) -> MeasuredLookup {
    var result: MeasuredLookup;
    result.offset = 0u;
    for (var i = 0u; i < 6u; i++) { result.weights[i] = 0.0; }
    for (var dim = 0u; dim < table.parameter_count; dim++) {
        let size = table.parameter_sizes[dim];
        if (size == 1u) {
            result.weights[2u * dim] = 1.0;
            result.weights[2u * dim + 1u] = 0.0;
            continue;
        }
        let values = table.parameter_value_offsets[dim];
        var first = 0u;
        var len = size;
        loop {
            if (len == 0u) { break; }
            let half = len / 2u;
            let middle = first + half;
            if (measured_scalar(values + middle) <= params[dim]) {
                first = middle + 1u;
                len = len - half - 1u;
            } else {
                len = half;
            }
        }
        let index = min(select(0u, first - 1u, first > 0u), size - 2u);
        let p0 = measured_scalar(values + index);
        let p1 = measured_scalar(values + index + 1u);
        let high = clamp((params[dim] - p0) / max(p1 - p0, 1.17549435e-38), 0.0, 1.0);
        result.weights[2u * dim] = 1.0 - high;
        result.weights[2u * dim + 1u] = high;
        result.offset += table.parameter_strides[dim] * index;
    }
    return result;
}

fn measured_lookup(table: MeasuredTableRecord, data: u32, index: u32,
                   slice_size: u32, lookup: MeasuredLookup) -> f32 {
    var sum = 0.0;
    let combinations = 1u << table.parameter_count;
    for (var combination = 0u; combination < combinations; combination++) {
        var offset = index;
        var weight = 1.0;
        for (var dim = 0u; dim < table.parameter_count; dim++) {
            if (((combination >> dim) & 1u) != 0u) {
                offset += table.parameter_strides[dim] * slice_size;
                weight *= lookup.weights[2u * dim + 1u];
            } else {
                weight *= lookup.weights[2u * dim];
            }
        }
        sum += weight * measured_scalar(data + offset);
    }
    return sum;
}

fn measured_evaluate(table_index: u32, position_in: vec2<f32>, params: vec3<f32>) -> f32 {
    if (table_index >= arrayLength(&measured_tables)) { set_render_error(); return 0.0; }
    let table = measured_tables[table_index];
    let lookup = measured_parameter_lookup(table, params);
    let inverse_patch = vec2<f32>(table.size - vec2<u32>(1u));
    let position = position_in * inverse_patch;
    let cell = min(vec2<u32>(max(position, vec2<f32>(0.0))), table.size - vec2<u32>(2u));
    let high = position - vec2<f32>(cell);
    let low = vec2<f32>(1.0) - high;
    let slice_size = table.size.x * table.size.y;
    let base = cell.x + cell.y * table.size.x + lookup.offset * slice_size;
    let v00 = measured_lookup(table, table.data_offset, base, slice_size, lookup);
    let v10 = measured_lookup(table, table.data_offset, base + 1u, slice_size, lookup);
    let v01 = measured_lookup(table, table.data_offset, base + table.size.x, slice_size, lookup);
    let v11 = measured_lookup(table, table.data_offset, base + table.size.x + 1u, slice_size, lookup);
    return (low.y * (low.x * v00 + high.x * v10)
        + high.y * (low.x * v01 + high.x * v11)) * inverse_patch.x * inverse_patch.y;
}

fn measured_sample(table_index: u32, sample_in: vec2<f32>, params: vec3<f32>) -> MeasuredPlSample {
    let table = measured_tables[table_index];
    let lookup = measured_parameter_lookup(table, params);
    let slice_size = table.size.x * table.size.y;
    var sample = clamp(sample_in, vec2<f32>(MACHINE_EPSILON), vec2<f32>(1.0 - MACHINE_EPSILON));
    var marginal_offset = lookup.offset * table.size.y;
    var first = 0u;
    var len = table.size.y;
    loop {
        if (len == 0u) { break; }
        let half = len / 2u;
        let middle = first + half;
        let value = measured_lookup(table, table.marginal_cdf_offset,
            marginal_offset + middle, table.size.y, lookup);
        if (value < sample.y) { first = middle + 1u; len = len - half - 1u; }
        else { len = half; }
    }
    let row = min(select(0u, first - 1u, first > 0u), table.size.y - 2u);
    sample.y -= measured_lookup(table, table.marginal_cdf_offset,
        marginal_offset + row, table.size.y, lookup);
    var offset = row * table.size.x + lookup.offset * slice_size;
    let r0 = measured_lookup(table, table.conditional_cdf_offset,
        offset + table.size.x - 1u, slice_size, lookup);
    let r1 = measured_lookup(table, table.conditional_cdf_offset,
        offset + 2u * table.size.x - 1u, slice_size, lookup);
    let row_constant = abs(r0 - r1) < 1e-4 * max(abs(r0 + r1), 1e-12);
    sample.y = select(r0 - sqrt(max(0.0, r0 * r0 - 2.0 * sample.y * (r0 - r1))),
                      2.0 * sample.y, row_constant);
    sample.y /= select(r0 - r1, r0 + r1, row_constant);
    sample.x *= mix(r0, r1, sample.y);

    first = 0u;
    len = table.size.x;
    loop {
        if (len == 0u) { break; }
        let half = len / 2u;
        let middle = first + half;
        let c0 = measured_lookup(table, table.conditional_cdf_offset,
            offset + middle, slice_size, lookup);
        let c1 = measured_lookup(table, table.conditional_cdf_offset,
            offset + middle + table.size.x, slice_size, lookup);
        if (mix(c0, c1, sample.y) < sample.x) { first = middle + 1u; len = len - half - 1u; }
        else { len = half; }
    }
    let column = min(select(0u, first - 1u, first > 0u), table.size.x - 2u);
    let conditional0 = measured_lookup(table, table.conditional_cdf_offset,
        offset + column, slice_size, lookup);
    let conditional1 = measured_lookup(table, table.conditional_cdf_offset,
        offset + column + table.size.x, slice_size, lookup);
    sample.x -= mix(conditional0, conditional1, sample.y);
    offset += column;
    let v00 = measured_lookup(table, table.data_offset, offset, slice_size, lookup);
    let v10 = measured_lookup(table, table.data_offset, offset + 1u, slice_size, lookup);
    let v01 = measured_lookup(table, table.data_offset, offset + table.size.x, slice_size, lookup);
    let v11 = measured_lookup(table, table.data_offset, offset + table.size.x + 1u, slice_size, lookup);
    let c0 = mix(v00, v01, sample.y);
    let c1 = mix(v10, v11, sample.y);
    let column_constant = abs(c0 - c1) < 1e-4 * max(abs(c0 + c1), 1e-12);
    sample.x = select(c0 - sqrt(max(0.0, c0 * c0 - 2.0 * sample.x * (c0 - c1))),
                      2.0 * sample.x, column_constant);
    sample.x /= select(c0 - c1, c0 + c1, column_constant);
    let inverse_patch = vec2<f32>(table.size - vec2<u32>(1u));
    return MeasuredPlSample(
        (vec2<f32>(f32(column), f32(row)) + sample) / inverse_patch,
        mix(c0, c1, sample.x) * inverse_patch.x * inverse_patch.y,
    );
}

fn measured_invert(table_index: u32, sample_in: vec2<f32>, params: vec3<f32>) -> MeasuredPlSample {
    let table = measured_tables[table_index];
    let lookup = measured_parameter_lookup(table, params);
    let slice_size = table.size.x * table.size.y;
    let inverse_patch = vec2<f32>(table.size - vec2<u32>(1u));
    var sample = sample_in * inverse_patch;
    let cell = min(vec2<u32>(max(sample, vec2<f32>(0.0))), table.size - vec2<u32>(2u));
    sample -= vec2<f32>(cell);
    var offset = cell.x + cell.y * table.size.x + lookup.offset * slice_size;
    let v00 = measured_lookup(table, table.data_offset, offset, slice_size, lookup);
    let v10 = measured_lookup(table, table.data_offset, offset + 1u, slice_size, lookup);
    let v01 = measured_lookup(table, table.data_offset, offset + table.size.x, slice_size, lookup);
    let v11 = measured_lookup(table, table.data_offset, offset + table.size.x + 1u, slice_size, lookup);
    let c0 = mix(v00, v01, sample.y);
    let c1 = mix(v10, v11, sample.y);
    let pdf = mix(c0, c1, sample.x);
    sample.x *= c0 + 0.5 * sample.x * (c1 - c0);
    let conditional0 = measured_lookup(table, table.conditional_cdf_offset,
        offset, slice_size, lookup);
    let conditional1 = measured_lookup(table, table.conditional_cdf_offset,
        offset + table.size.x, slice_size, lookup);
    sample.x += mix(conditional0, conditional1, sample.y);
    let row_offset = cell.y * table.size.x + lookup.offset * slice_size;
    let r0 = measured_lookup(table, table.conditional_cdf_offset,
        row_offset + table.size.x - 1u, slice_size, lookup);
    let r1 = measured_lookup(table, table.conditional_cdf_offset,
        row_offset + 2u * table.size.x - 1u, slice_size, lookup);
    sample.x /= mix(r0, r1, sample.y);
    sample.y *= r0 + 0.5 * sample.y * (r1 - r0);
    sample.y += measured_lookup(table, table.marginal_cdf_offset,
        lookup.offset * table.size.y + cell.y, table.size.y, lookup);
    return MeasuredPlSample(sample, pdf * inverse_patch.x * inverse_patch.y);
}

fn measured_id(material_node: u32) -> u32 {
    let reference = load_material_attribute(material_node, 0u);
    if (reference.kind != 3u || reference.index >= arrayLength(&measured_bsdfs)) {
        set_render_error();
        return 0xffffffffu;
    }
    return reference.index;
}

fn measured_theta_to_u(theta: f32) -> f32 { return sqrt(theta * (2.0 / PI)); }
fn measured_phi_to_u(phi: f32) -> f32 { return phi * (1.0 / (2.0 * PI)) + 0.5; }
fn measured_u_to_theta(u: f32) -> f32 { return u * u * (PI / 2.0); }
fn measured_u_to_phi(u: f32) -> f32 { return (2.0 * u - 1.0) * PI; }

fn measured_f(id: u32, wo_in: vec3<f32>, wi_in: vec3<f32>, lambda: vec4<f32>) -> vec4<f32> {
    if (id >= arrayLength(&measured_bsdfs) || wo_in.z * wi_in.z <= 0.0) { return vec4<f32>(0.0); }
    let brdf = measured_bsdfs[id];
    var wo = wo_in;
    var wi = wi_in;
    if (wo.z < 0.0) { wo = -wo; wi = -wi; }
    if (dot(wi + wo, wi + wo) == 0.0) { return vec4<f32>(0.0); }
    let wm = normalize(wi + wo);
    let theta_o = acos(clamp(wo.z, -1.0, 1.0));
    let phi_o = atan2(wo.y, wo.x);
    let theta_m = acos(clamp(wm.z, -1.0, 1.0));
    let phi_m = atan2(wm.y, wm.x);
    let u_wo = vec2<f32>(measured_theta_to_u(theta_o), measured_phi_to_u(phi_o));
    var u_wm = vec2<f32>(measured_theta_to_u(theta_m),
        measured_phi_to_u(select(phi_m, phi_m - phi_o, brdf.isotropic != 0u)));
    u_wm.y -= floor(u_wm.y);
    let inverted = measured_invert(brdf.vndf, u_wm, vec3<f32>(phi_o, theta_o, 0.0));
    var fr = vec4<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        fr[i] = max(0.0, measured_evaluate(brdf.spectra, inverted.p,
            vec3<f32>(phi_o, theta_o, lambda[i])));
    }
    return fr * measured_evaluate(brdf.ndf, u_wm, vec3<f32>(0.0))
        / (4.0 * measured_evaluate(brdf.sigma, u_wo, vec3<f32>(0.0)) * wi.z);
}

fn measured_pdf(id: u32, wo_in: vec3<f32>, wi_in: vec3<f32>) -> f32 {
    if (id >= arrayLength(&measured_bsdfs) || wo_in.z * wi_in.z <= 0.0) { return 0.0; }
    let brdf = measured_bsdfs[id];
    var wo = wo_in;
    var wi = wi_in;
    if (wo.z < 0.0) { wo = -wo; wi = -wi; }
    if (dot(wi + wo, wi + wo) == 0.0) { return 0.0; }
    let wm = normalize(wi + wo);
    let theta_o = acos(clamp(wo.z, -1.0, 1.0));
    let phi_o = atan2(wo.y, wo.x);
    let theta_m = acos(clamp(wm.z, -1.0, 1.0));
    let phi_m = atan2(wm.y, wm.x);
    var u_wm = vec2<f32>(measured_theta_to_u(theta_m),
        measured_phi_to_u(select(phi_m, phi_m - phi_o, brdf.isotropic != 0u)));
    u_wm.y -= floor(u_wm.y);
    let inverted = measured_invert(brdf.vndf, u_wm, vec3<f32>(phi_o, theta_o, 0.0));
    let luminance_pdf = measured_evaluate(brdf.luminance, inverted.p,
        vec3<f32>(phi_o, theta_o, 0.0));
    let sin_theta_m = length(wm.xy);
    let jacobian = 4.0 * dot(wo, wm)
        * max(2.0 * PI * PI * u_wm.x * sin_theta_m, 1e-6);
    return inverted.pdf * luminance_pdf / jacobian;
}

fn measured_sample_f(id: u32, wo_in: vec3<f32>, u: vec2<f32>, lambda: vec4<f32>) -> MeasuredBxdfSample {
    var invalid = MeasuredBxdfSample(vec4<f32>(0.0), vec3<f32>(0.0), 0.0, 0u);
    if (id >= arrayLength(&measured_bsdfs)) { return invalid; }
    let brdf = measured_bsdfs[id];
    var wo = wo_in;
    var flip_wi = false;
    if (wo.z <= 0.0) { wo = -wo; flip_wi = true; }
    let theta_o = acos(clamp(wo.z, -1.0, 1.0));
    let phi_o = atan2(wo.y, wo.x);
    let luminance = measured_sample(brdf.luminance, u, vec3<f32>(phi_o, theta_o, 0.0));
    let vndf = measured_sample(brdf.vndf, luminance.p, vec3<f32>(phi_o, theta_o, 0.0));
    var phi_m = measured_u_to_phi(vndf.p.y);
    let theta_m = measured_u_to_theta(vndf.p.x);
    if (brdf.isotropic != 0u) { phi_m += phi_o; }
    let wm = vec3<f32>(sin(theta_m) * cos(phi_m), sin(theta_m) * sin(phi_m), cos(theta_m));
    var wi = reflect(-wo, wm);
    if (wi.z <= 0.0) { return invalid; }
    var fr = vec4<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        fr[i] = max(0.0, measured_evaluate(brdf.spectra, luminance.p,
            vec3<f32>(phi_o, theta_o, lambda[i])));
    }
    let u_wo = vec2<f32>(measured_theta_to_u(theta_o), measured_phi_to_u(phi_o));
    fr *= measured_evaluate(brdf.ndf, vndf.p, vec3<f32>(0.0))
        / (4.0 * measured_evaluate(brdf.sigma, u_wo, vec3<f32>(0.0)) * abs(wi.z));
    let jacobian = 4.0 * dot(wo, wm)
        * max(2.0 * PI * PI * vndf.p.x * sin(theta_m), 1e-6);
    if (flip_wi) { wi = -wi; }
    return MeasuredBxdfSample(fr, wi, vndf.pdf * luminance.pdf / jacobian, 1u);
}
