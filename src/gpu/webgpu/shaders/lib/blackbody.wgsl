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

// B(lambda,T)/B(lambda_max,T), with common radiance factors cancelled.
fn normalized_blackbody_ratio(wavelength_nm: f32, temperature: f32) -> f32 {
    let c: f32 = 299792458.0;
    let h: f32 = 6.62606957e-34;
    let kb: f32 = 1.3806488e-23;
    let wien: f32 = 2.8977721e-3;
    let wavelength_max_nm = 2.8977721e6 / temperature;
    let ratio = wavelength_max_nm / wavelength_nm;
    let peak_exponent = (h * c) / (wien * kb);
    let exponent = peak_exponent * ratio;
    if (exponent * 1.442695041 >= 128.0) { return 0.0; }
    var denominator_over_ratio: f32;
    if (exponent < 0.5) {
        // expm1(x)/x through x^6; the remainder is below 3.2e-7 for 0 <= x < 0.5.
        let expm1_over_x = fma(exponent, fma(exponent, fma(exponent,
            fma(exponent, fma(exponent, fma(exponent, 1.0 / 5040.0, 1.0 / 720.0),
                1.0 / 120.0), 1.0 / 24.0), 1.0 / 6.0), 0.5), 1.0);
        denominator_over_ratio = peak_exponent * expm1_over_x;
    } else {
        denominator_over_ratio = (exp(exponent) - 1.0) / ratio;
    }
    let ratio2 = ratio * ratio;
    return (ratio2 * ratio2) * (exp(peak_exponent) - 1.0) / denominator_over_ratio;
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
    let wavelength_max2 = wavelength_max * wavelength_max;
    let wavelength_max5 = wavelength_max2 * wavelength_max2 * wavelength_max;
    var value: vec4<f32>;
    if (wavelength_max5 < 1.175494351e-38) {
        // Avoid subnormal peak wavelength powers on devices that flush them to zero.
        value = vec4<f32>(
            normalized_blackbody_ratio(lambda.x, temperature), normalized_blackbody_ratio(lambda.y, temperature),
            normalized_blackbody_ratio(lambda.z, temperature), normalized_blackbody_ratio(lambda.w, temperature),
        );
    } else {
        let normalization = 1.0 / blackbody(wavelength_max * 1e9, temperature);
        value = normalization * vec4<f32>(
            blackbody(lambda.x, temperature), blackbody(lambda.y, temperature),
            blackbody(lambda.z, temperature), blackbody(lambda.w, temperature),
        );
        if (normalization != normalization || abs(normalization) > RAY_T_MAX) {
            set_render_error();
            return vec4<f32>(0.0);
        }
    }
    if (any(value != value) || any(abs(value) > vec4<f32>(RAY_T_MAX))) {
        set_render_error();
        return vec4<f32>(0.0);
    }
    return value;
}
