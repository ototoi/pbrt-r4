struct BSSRDFSplineWeights { offset: i32, weights: vec4<f32>, valid: u32, }
fn find_interval(offset: u32, count: u32, x: f32) -> u32 {
    var low = 0u;
    var high = count - 1u;
    while (high - low > 1u) {
        let mid = (low + high) / 2u;
        if (bssrdf_values[offset + mid] <= x) { low = mid; } else { high = mid; }
    }
    return low;
}
// pbrt-v4 CatmullRomWeights.
fn catmull_rom_weights(offset: u32, count: u32, x: f32) -> BSSRDFSplineWeights {
    if (!(x >= bssrdf_values[offset] && x <= bssrdf_values[offset + count - 1u])) {
        return BSSRDFSplineWeights(0, vec4<f32>(0.0), 0u);
    }
    let idx = find_interval(offset, count, x);
    let x0 = bssrdf_values[offset + idx];
    let x1 = bssrdf_values[offset + idx + 1u];
    let t = (x - x0) / (x1 - x0);
    let t2 = t * t;
    let t3 = t2 * t;
    var w = vec4<f32>(0.0, 2.0 * t3 - 3.0 * t2 + 1.0, -2.0 * t3 + 3.0 * t2, 0.0);
    var w0 = t3 - 2.0 * t2 + t;
    if (idx > 0u) {
        w0 *= (x1 - x0) / (x1 - bssrdf_values[offset + idx - 1u]);
        w.x = -w0;
    } else { w.y -= w0; }
    w.z += w0;
    var w3 = t3 - t2;
    if (idx + 2u < count) {
        w3 *= (x1 - x0) / (bssrdf_values[offset + idx + 2u] - x0);
        w.w = w3;
    } else { w.z += w3; }
    w.y -= w3;
    return BSSRDFSplineWeights(i32(idx) - 1, w, 1u);
}
fn bssrdf_interpolate(offset: u32, width: u32, column: u32, w: BSSRDFSplineWeights) -> f32 {
    var value = 0.0;
    for (var i = 0u; i < 4u; i++) {
        if (w.weights[i] != 0.0) {
            value += w.weights[i] * bssrdf_values[offset + u32(w.offset + i32(i)) * width + column];
        }
    }
    return value;
}
fn bssrdf_integral(t: f32, f0: f32, f1: f32, d0: f32, d1: f32) -> vec2<f32> {
    let integral = t * (f0 + t * (0.5 * d0 + t * ((-2.0 * d0 - d1) / 3.0 + f1 - f0
        + t * (0.25 * (d0 + d1) + 0.5 * (f0 - f1)))));
    let density = f0 + t * (d0 + t * (-2.0 * d0 - d1 + 3.0 * (f1 - f0)
        + t * (d0 + d1 + 2.0 * (f0 - f1))));
    return vec2<f32>(integral, density);
}
// pbrt-v4 SampleCatmullRom2D and NewtonBisection, including endpoint roots.
fn sample_catmull_rom_2d(table: BSSRDFTableRecord, rho: f32, u: f32) -> f32 {
    let w = catmull_rom_weights(table.rho_offset, table.rho_count, rho);
    if (w.valid == 0u) { return 0.0; }
    let n = table.radius_count;
    let cdf_u = u * bssrdf_interpolate(table.cdf_offset, n, n - 1u, w);
    var low = 0u;
    var high = n - 1u;
    while (high - low > 1u) {
        let mid = (low + high) / 2u;
        if (bssrdf_interpolate(table.cdf_offset, n, mid, w) <= cdf_u) { low = mid; }
        else { high = mid; }
    }
    let idx = low;
    let f0 = bssrdf_interpolate(table.profile_offset, n, idx, w);
    let f1 = bssrdf_interpolate(table.profile_offset, n, idx + 1u, w);
    let x0 = bssrdf_values[table.radius_offset + idx];
    let x1 = bssrdf_values[table.radius_offset + idx + 1u];
    let width = x1 - x0;
    let local_u = (cdf_u - bssrdf_interpolate(table.cdf_offset, n, idx, w)) / width;
    var d0 = f1 - f0;
    var d1 = d0;
    if (idx > 0u) {
        d0 = width * (f1 - bssrdf_interpolate(table.profile_offset, n, idx - 1u, w))
            / (x1 - bssrdf_values[table.radius_offset + idx - 1u]);
    }
    if (idx + 2u < n) {
        d1 = width * (bssrdf_interpolate(table.profile_offset, n, idx + 2u, w) - f0)
            / (bssrdf_values[table.radius_offset + idx + 2u] - x0);
    }
    let fx0 = -local_u;
    let fx1 = bssrdf_integral(1.0, f0, f1, d0, d1).x - local_u;
    if (abs(fx0) < 1e-6) { return x0; }
    if (abs(fx1) < 1e-6) { return x1; }
    let start_negative = fx0 < 0.0;
    var a = 0.0;
    var b = 1.0;
    var t = -fx0 / (fx1 - fx0);
    loop {
        if (!(a < t && t < b)) { t = (a + b) * 0.5; }
        let evaluation = bssrdf_integral(t, f0, f1, d0, d1);
        let residual = evaluation.x - local_u;
        if (start_negative == (residual < 0.0)) { a = t; } else { b = t; }
        if (b - a < 1e-6 || abs(residual) < 1e-6) { break; }
        t -= residual / evaluation.y;
    }
    return x0 + width * t;
}
struct BSSRDFSegment { start: vec3<f32>, end: vec3<f32>, valid: u32, }
fn tabulated_bssrdf_sample_sp(work: BSSRDFProbeWorkItem) -> BSSRDFSegment {
    let invalid_segment = BSSRDFSegment(vec3<f32>(0.0), vec3<f32>(0.0), 0u);
    if (work.sigma_t.x == 0.0) { return invalid_segment; }
    if (work.table_index >= arrayLength(&bssrdf_tables)) { set_render_error(); return invalid_segment; }
    let table = bssrdf_tables[work.table_index];
    let r = sample_catmull_rom_2d(table, work.rho.x, work.sample.y) / work.sigma_t.x;
    let r_max = sample_catmull_rom_2d(table, work.rho.x, 0.999) / work.sigma_t.x;
    if (r >= r_max) { return invalid_segment; }
    let ns = work.normal.xyz;
    let c0 = coordinate_system_x(ns);
    let c1 = coordinate_system_y(ns);
    var frame = mat3x3<f32>(c0, c1, ns);
    if (work.sample.x < 0.25) { frame = mat3x3<f32>(ns, c0, c1); }
    else if (work.sample.x < 0.5) { frame = mat3x3<f32>(c1, ns, c0); }
    let phi = 2.0 * PI * work.sample.z;
    let half_length = sqrt(r_max * r_max - r * r);
    let start = work.position.xyz + r * (frame[0] * cos(phi) + frame[1] * sin(phi)) - half_length * frame[2];
    return BSSRDFSegment(start, start + 2.0 * half_length * frame[2], 1u);
}

// pbrt-v4 InvertCatmullRom(rhoSamples, rhoEff, reflectance).
fn invert_catmull_rom(table: BSSRDFTableRecord, u: f32) -> f32 {
    let nodes = table.rho_offset;
    let values = table.rho_eff_offset;
    let n = table.rho_count;
    if (!(u > bssrdf_values[values])) { return bssrdf_values[nodes]; }
    if (!(u < bssrdf_values[values + n - 1u])) { return bssrdf_values[nodes + n - 1u]; }
    let i = find_interval(values, n, u);
    let x0 = bssrdf_values[nodes + i];
    let x1 = bssrdf_values[nodes + i + 1u];
    let f0 = bssrdf_values[values + i];
    let f1 = bssrdf_values[values + i + 1u];
    var d0 = f1 - f0;
    var d1 = d0;
    if (i > 0u) { d0 = (x1 - x0) * (f1 - bssrdf_values[values + i - 1u]) / (x1 - bssrdf_values[nodes + i - 1u]); }
    if (i + 2u < n) { d1 = (x1 - x0) * (bssrdf_values[values + i + 2u] - f0) / (bssrdf_values[nodes + i + 2u] - x0); }
    if (abs(f0 - u) < 1e-6) { return x0; }
    if (abs(f1 - u) < 1e-6) { return x1; }
    var a = 0.0;
    var b = 1.0;
    var t = (u - f0) / (f1 - f0);
    loop {
        if (!(a < t && t < b)) { t = (a + b) * 0.5; }
        let t2 = t * t;
        let t3 = t2 * t;
        let residual = (2.0*t3 - 3.0*t2 + 1.0)*f0 + (-2.0*t3 + 3.0*t2)*f1
            + (t3 - 2.0*t2 + t)*d0 + (t3 - t2)*d1 - u;
        let derivative = (6.0*t2 - 6.0*t)*f0 + (-6.0*t2 + 6.0*t)*f1
            + (3.0*t2 - 4.0*t + 1.0)*d0 + (3.0*t2 - 2.0*t)*d1;
        if (residual < 0.0) { a = t; } else { b = t; }
        if (b - a < 1e-6 || abs(residual) < 1e-6) { break; }
        t -= residual / derivative;
    }
    return x0 + (x1 - x0) * t;
}

fn tabulated_bssrdf_sr(work: BSSRDFProbeWorkItem, radius: f32, normalized: bool) -> vec4<f32> {
    let table = bssrdf_tables[work.table_index];
    var sr = vec4<f32>(0.0);
    for (var channel = 0u; channel < 4u; channel++) {
        let r = radius * work.sigma_t[channel];
        let rw = catmull_rom_weights(table.rho_offset, table.rho_count, work.rho[channel]);
        let dw = catmull_rom_weights(table.radius_offset, table.radius_count, r);
        if (rw.valid == 0u || dw.valid == 0u) { continue; }
        var value = 0.0;
        for (var k = 0u; k < 4u; k++) {
            if (dw.weights[k] != 0.0) {
                value += dw.weights[k] * bssrdf_interpolate(table.profile_offset, table.radius_count,
                    u32(dw.offset + i32(k)), rw);
            }
        }
        if (r != 0.0) { value /= 2.0 * PI * r; }
        value *= work.sigma_t[channel] * work.sigma_t[channel];
        if (normalized) { value /= bssrdf_interpolate(table.rho_eff_offset, 1u, 0u, rw); }
        sr[channel] = max(0.0, value);
    }
    return sr;
}
fn tabulated_bssrdf_pdf_sp(work: BSSRDFProbeWorkItem, position: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    let ns = work.normal.xyz;
    let frame = mat3x3<f32>(coordinate_system_x(ns), coordinate_system_y(ns), ns);
    let d = transpose(frame) * (position - work.position.xyz);
    let n = abs(transpose(frame) * normal);
    return tabulated_bssrdf_sr(work, length(d.yz), true) * (n.x * 0.25)
        + tabulated_bssrdf_sr(work, length(d.zx), true) * (n.y * 0.25)
        + tabulated_bssrdf_sr(work, length(d.xy), true) * (n.z * 0.5);
}
