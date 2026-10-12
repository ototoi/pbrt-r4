// pbrt-v4 util/math.h::FastExp (CPU approximation).
fn fast_exp(x: f32) -> f32 {
    let xp = x * 1.442695041;
    if (xp != xp) { set_render_error(); return 0.0; }
    if (xp < -126.0) { return 0.0; }
    if (xp >= 128.0) { return bitcast<f32>(0x7f800000u); }
    let fxp = floor(xp);
    let fraction = xp - fxp;
    let two_to_f = fma(fraction,
        fma(fraction, fma(fraction, 0.0781455737, 0.226173572), 0.695556856), 1.0);
    let bits = bitcast<u32>(two_to_f);
    let exponent = i32((bits >> 23u) & 255u) - 127 + i32(fxp);
    if (exponent < -126) { return 0.0; }
    if (exponent > 127) { return bitcast<f32>(0x7f800000u); }
    return bitcast<f32>((bits & 0x807fffffu) | (u32(exponent + 127) << 23u));
}

// pbrt-v4 util/spectrum.h::Blackbody.
fn blackbody(wavelength_nm: f32, temperature: f32) -> f32 {
    if (temperature <= 0.0) { return 0.0; }
    let c: f32 = 299792458.0;
    let h: f32 = 6.62606957e-34;
    let kb: f32 = 1.3806488e-23;
    let wavelength = wavelength_nm * 1e-9;
    let wavelength2 = wavelength * wavelength;
    let wavelength5 = wavelength2 * wavelength2 * wavelength;
    let exponent = (h * c) / (wavelength * kb * temperature);
    return (2.0 * h * c * c) / (wavelength5 * (fast_exp(exponent) - 1.0));
}

fn blackbody_spectrum(temperature: f32, lambda: vec4<f32>) -> vec4<f32> {
    if (temperature != temperature || abs(temperature) > RAY_T_MAX
        || any(lambda != lambda) || any(abs(lambda) > vec4<f32>(RAY_T_MAX))
        || any(lambda <= vec4<f32>(0.0))) {
        set_render_error();
        return vec4<f32>(0.0);
    }
    if (temperature <= 0.0) { return vec4<f32>(0.0); }
    let wavelength_max = 2.8977721e-3 / temperature;
    let normalization = 1.0 / blackbody(wavelength_max * 1e9, temperature);
    let value = normalization * vec4<f32>(
        blackbody(lambda.x, temperature), blackbody(lambda.y, temperature),
        blackbody(lambda.z, temperature), blackbody(lambda.w, temperature),
    );
    if (normalization != normalization || abs(normalization) > RAY_T_MAX
        || any(value != value) || any(abs(value) > vec4<f32>(RAY_T_MAX))) {
        set_render_error();
        return vec4<f32>(0.0);
    }
    return value;
}
