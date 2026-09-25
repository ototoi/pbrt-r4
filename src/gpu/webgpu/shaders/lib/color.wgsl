const RGB_TABLE_RESOLUTION: u32 = 64u;
const RGB_TABLE_SCALE_COUNT: u32 = RGB_TABLE_RESOLUTION;
const RGB_TABLE_COEFF_COUNT: u32 = 3u * RGB_TABLE_RESOLUTION * RGB_TABLE_RESOLUTION * RGB_TABLE_RESOLUTION * 3u;
const RGB_TABLE_STRIDE: u32 = RGB_TABLE_SCALE_COUNT + RGB_TABLE_COEFF_COUNT;

fn rgb_table_coeff(table_base: u32, maxc: u32, z: u32, y: u32, x: u32, component: u32) -> f32 {
    let index = (((((maxc * RGB_TABLE_RESOLUTION) + z) * RGB_TABLE_RESOLUTION + y)
        * RGB_TABLE_RESOLUTION + x) * 3u) + component;
    return rgb_spectrum_table[table_base + RGB_TABLE_SCALE_COUNT + index];
}

fn rgb_sigmoid(x: f32) -> f32 {
    return 0.5 + x / (2.0 * sqrt(1.0 + x * x));
}

fn rgb_to_spectrum_lane(rgb_input: vec3<f32>, lambda: f32, color_space: u32) -> f32 {
    let table_base = min(color_space, 3u) * RGB_TABLE_STRIDE;
    let rgb = clamp(rgb_input, vec3<f32>(0.0), vec3<f32>(1.0));
    let z = max(max(rgb.x, rgb.y), rgb.z);
    if (z == 0.0) { return 0.0; }
    if (rgb.x == rgb.y && rgb.y == rgb.z) {
        if (rgb.x <= 0.0) { return 0.0; }
        if (rgb.x >= 1.0) { return 1.0; }
        let c2 = (rgb.x - 0.5) / sqrt(max(rgb.x * (1.0 - rgb.x), 1e-8));
        return rgb_sigmoid(c2);
    }
    // Match pbrt-v4's strict tie-breaking: red, then green, then blue.
    var maxc = 2u;
    if (rgb.x > rgb.y) {
        if (rgb.x > rgb.z) { maxc = 0u; }
    } else if (rgb.y > rgb.z) {
        maxc = 1u;
    }
    var c1 = rgb.y;
    var c2 = rgb.z;
    if (maxc == 1u) { c1 = rgb.z; c2 = rgb.x; }
    if (maxc == 2u) { c1 = rgb.x; c2 = rgb.y; }
    let x = c1 * f32(RGB_TABLE_RESOLUTION - 1u) / z;
    let y = c2 * f32(RGB_TABLE_RESOLUTION - 1u) / z;
    let xi = min(u32(floor(x)), 62u);
    let yi = min(u32(floor(y)), 62u);
    var zi = 0u;
    for (var zindex = 0u; zindex < 63u; zindex++) {
        if (rgb_spectrum_table[table_base + zindex + 1u] < z) { zi = zindex + 1u; }
    }
    zi = min(zi, 62u);
    let dx = x - f32(xi);
    let dy = y - f32(yi);
    let z0 = rgb_spectrum_table[table_base + zi];
    let z1 = rgb_spectrum_table[table_base + zi + 1u];
    let dz = (z - z0) / max(z1 - z0, 1e-8);
    var polynomial_coeff = vec3<f32>(0.0);
    for (var component = 0u; component < 3u; component++) {
        let c000 = rgb_table_coeff(table_base, maxc, zi, yi, xi, component);
        let c100 = rgb_table_coeff(table_base, maxc, zi, yi, xi + 1u, component);
        let c010 = rgb_table_coeff(table_base, maxc, zi, yi + 1u, xi, component);
        let c110 = rgb_table_coeff(table_base, maxc, zi, yi + 1u, xi + 1u, component);
        let c001 = rgb_table_coeff(table_base, maxc, zi + 1u, yi, xi, component);
        let c101 = rgb_table_coeff(table_base, maxc, zi + 1u, yi, xi + 1u, component);
        let c011 = rgb_table_coeff(table_base, maxc, zi + 1u, yi + 1u, xi, component);
        let c111 = rgb_table_coeff(table_base, maxc, zi + 1u, yi + 1u, xi + 1u, component);
        let c0 = mix(mix(c000, c100, dx), mix(c010, c110, dx), dy);
        let c1 = mix(mix(c001, c101, dx), mix(c011, c111, dx), dy);
        polynomial_coeff[component] = mix(c0, c1, dz);
    }
    let polynomial = (polynomial_coeff.x * lambda + polynomial_coeff.y) * lambda + polynomial_coeff.z;
    return rgb_sigmoid(polynomial);
}

fn rgb_to_spectrum4(rgb: vec3<f32>, lambda: vec4<f32>, color_space: u32) -> vec4<f32> {
    return vec4<f32>(
        rgb_to_spectrum_lane(rgb, lambda.x, color_space),
        rgb_to_spectrum_lane(rgb, lambda.y, color_space),
        rgb_to_spectrum_lane(rgb, lambda.z, color_space),
        rgb_to_spectrum_lane(rgb, lambda.w, color_space),
    );
}

fn rgb_to_unbounded_spectrum4(rgb: vec3<f32>, lambda: vec4<f32>, color_space: u32) -> vec4<f32> {
    let max_value = max(max(rgb.x, rgb.y), rgb.z);
    if (max_value <= 0.0) { return vec4<f32>(0.0); }
    let scale = 2.0 * max_value;
    return scale * rgb_to_spectrum4(rgb / scale, lambda, color_space);
}
