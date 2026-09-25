fn set_render_error() {
    atomicStore(&render_error.value, 1u);
}

fn load_sample_radiance(pixel_index: u32) -> vec4<f32> {
    return pixel_sample_states[pixel_index].radiance;
}

fn store_sample_radiance(pixel_index: u32, radiance: vec4<f32>) {
    if (radiance.x != radiance.x || radiance.y != radiance.y || radiance.z != radiance.z
        || radiance.w != radiance.w || abs(radiance.x) > RAY_T_MAX || abs(radiance.y) > RAY_T_MAX
        || abs(radiance.z) > RAY_T_MAX || abs(radiance.w) > RAY_T_MAX) {
        set_render_error();
    }
    pixel_sample_states[pixel_index].radiance = radiance;
}

fn load_sample_lambda(pixel_index: u32) -> vec4<f32> {
    return pixel_sample_states[pixel_index].lambda;
}

fn load_sample_lambda_pdf(pixel_index: u32) -> vec4<f32> {
    return pixel_sample_states[pixel_index].lambda_pdf;
}

fn store_sample_wavelengths(pixel_index: u32, lambda: vec4<f32>, pdf: vec4<f32>) {
    pixel_sample_states[pixel_index].lambda = lambda;
    pixel_sample_states[pixel_index].lambda_pdf = pdf;
}

fn terminate_secondary_wavelengths(pixel_index: u32) {
    var pdf = load_sample_lambda_pdf(pixel_index);
    if (pdf.y == 0.0 && pdf.z == 0.0 && pdf.w == 0.0) { return; }
    pdf = vec4<f32>(pdf.x / 4.0, 0.0, 0.0, 0.0);
    pixel_sample_states[pixel_index].lambda_pdf = pdf;
}

fn load_ray_samples(pixel_index: u32) -> RaySamples {
    let state = pixel_sample_states[pixel_index];
    return RaySamples(state.direct, state.indirect);
}

fn store_ray_samples(pixel_index: u32, samples: RaySamples) {
    pixel_sample_states[pixel_index].direct = samples.direct;
    pixel_sample_states[pixel_index].indirect = samples.indirect;
}

fn generate_ray_samples(pixel_index: u32, depth: u32) -> RaySamples {
    let first_dimension = 6u + 8u * depth;
    return RaySamples(
        vec4<f32>(
            sampler_get_1d(pixel_index, first_dimension),
            sampler_get_2d(pixel_index, first_dimension + 1u),
            sampler_get_1d(pixel_index, first_dimension + 3u),
        ),
        vec4<f32>(
            sampler_get_1d(pixel_index, first_dimension + 4u),
            sampler_get_2d(pixel_index, first_dimension + 5u),
            sampler_get_1d(pixel_index, first_dimension + 7u),
        ),
    );
}
