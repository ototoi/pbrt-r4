@group(0) @binding(0) var<storage, read> cases: array<LayeredBxDFData>;
@group(0) @binding(1) var<storage, read_write> results: array<vec4<f32>>;

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let data = cases[id.y];
    let seed = layered_hash(id.x + 1u);
    let eta = select(1.5, 1.0, id.y == 0u);
    let reflectance = vec3<f32>(0.5);
    let wo = vec3<f32>(0.0, 0.0, 1.0);
    let uc = layered_random(seed, 7u, 0u, 0u, 0u);
    let u = vec2<f32>(layered_random(seed, 7u, 0u, 0u, 1u), layered_random(seed, 7u, 0u, 0u, 2u));
    let bs = layered_sample(data, eta, reflectance, wo, uc, u, seed);
    let back = layered_sample(data, eta, reflectance, -wo, uc, u, seed);
    let z = layered_random(seed, 8u, 0u, 0u, 0u);
    let phi = 2.0 * LAYERED_PI * layered_random(seed, 8u, 0u, 0u, 1u);
    let wi = vec3<f32>(sqrt(1.0 - z * z) * cos(phi), sqrt(1.0 - z * z) * sin(phi), z);
    let f = layered_f(data, eta, reflectance, wo, wi, seed).x;
    let back_f = layered_f(data, eta, reflectance, -wo, -wi, seed).x;
    var weight = 0.0;
    var norm_error = 0.0;
    var analytic_weight = 0.0;
    if (bs.valid != 0u && bs.pdf > 0.0) {
        weight = bs.f.x * abs(bs.wi.z) / bs.pdf;
        norm_error = abs(length(bs.wi.xyz) - 1.0);
        analytic_weight = 0.5 * exp(-data.thickness * (1.0 + 1.0 / abs(bs.wi.z)));
    }
    let expected_pdf = 0.1 / (4.0 * LAYERED_PI)
        + select(0.0, 0.9 * sqrt(1.0 - (1.0 - wi.z * wi.z) / (eta * eta)) / LAYERED_PI, wi.z > 0.0);
    let spec = layered_sample(data, eta, reflectance, wo, 0.0, u, seed);
    var spec_weight = 0.0;
    if (spec.pdf > 0.0) { spec_weight = spec.f.x * abs(spec.wi.z) / spec.pdf; }
    let base = (id.y * 8192u + id.x) * 4u;
    results[base] = vec4<f32>(weight, f * wi.z * 2.0 * LAYERED_PI, f, back_f);
    results[base + 1u] = vec4<f32>(length(bs.wi.xyz + back.wi.xyz), norm_error,
        f32(bs.flags & SCATTER_TRANSMISSION), layered_pdf(data, eta, wo, wi));
    results[base + 2u] = vec4<f32>(expected_pdf, analytic_weight,
        0.5 / LAYERED_PI * exp(-data.thickness * (1.0 + 1.0 / wi.z)), spec_weight);
    results[base + 3u] = vec4<f32>(spec.pdf, f32(spec.flags), 0.0, 0.0);
}
