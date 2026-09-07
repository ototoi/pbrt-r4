const DENSE_LAMBDA_MIN: f32 = 360.0;
const DENSE_LAMBDA_MAX: f32 = 830.0;
const DENSE_SAMPLE_COUNT: u32 = 471u;
const SPECTRUM_FLAG_CONSTANT: u32 = 1u;

fn evaluate_spectrum_lane(id: u32, lambda: f32) -> f32 {
    if (id >= arrayLength(&spectrum_attributes)) {
        set_render_error();
        return 0.0;
    }
    if (lambda < DENSE_LAMBDA_MIN || lambda > DENSE_LAMBDA_MAX) {
        return 0.0;
    }
    let x = lambda - DENSE_LAMBDA_MIN;
    let i0 = u32(floor(x));
    let i1 = min(i0 + 1u, DENSE_SAMPLE_COUNT - 1u);
    return mix(spectrum_attributes[id].samples[i0], spectrum_attributes[id].samples[i1], fract(x));
}

fn evaluate_spectrum(id: u32, lambda: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        evaluate_spectrum_lane(id, lambda.x),
        evaluate_spectrum_lane(id, lambda.y),
        evaluate_spectrum_lane(id, lambda.z),
        evaluate_spectrum_lane(id, lambda.w),
    );
}

fn spectrum_is_constant(id: u32) -> bool {
    if (id >= arrayLength(&spectrum_attributes)) {
        set_render_error();
        return false;
    }
    return (spectrum_attributes[id].flags & SPECTRUM_FLAG_CONSTANT) != 0u;
}

fn safe_div_spectrum(value: vec4<f32>, pdf: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        select(0.0, value.x / pdf.x, pdf.x != 0.0),
        select(0.0, value.y / pdf.y, pdf.y != 0.0),
        select(0.0, value.z / pdf.z, pdf.z != 0.0),
        select(0.0, value.w / pdf.w, pdf.w != 0.0),
    );
}

fn average_spectrum(value: vec4<f32>) -> f32 {
    return dot(value, vec4<f32>(0.25));
}

fn max_spectrum(value: vec4<f32>) -> f32 {
    return max(max(value.x, value.y), max(value.z, value.w));
}
