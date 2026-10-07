struct ImageInfiniteSampleResult {
    uv: vec2<f32>,
    pdf: f32,
    valid: u32,
};

fn image_infinite_distribution_pdf(
    image: ImageInfiniteSamplingRecord,
    uv: vec2<f32>,
) -> f32 {
    if (image.width == 0u || image.height == 0u) {
        set_render_error();
        return 0.0;
    }
    let row_cdf_index = image.row_cdf_offset + image.height - 1u;
    if (row_cdf_index >= arrayLength(&image_infinite_row_cdf)) {
        set_render_error();
        return 0.0;
    }
    let integral = image_infinite_row_cdf[row_cdf_index];
    if (!(integral > 0.0)) { return 0.0; }
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let x = min(u32(p.x * f32(image.width)), image.width - 1u);
    let y = min(u32(p.y * f32(image.height)), image.height - 1u);
    let index = image.distribution_offset + y * image.width + x;
    if (index >= arrayLength(&image_infinite_distribution)) {
        set_render_error();
        return 0.0;
    }
    return image_infinite_distribution[index].weight / integral;
}

fn sample_image_infinite_distribution(
    image: ImageInfiniteSamplingRecord,
    u: vec2<f32>,
) -> ImageInfiniteSampleResult {
    if (image.width == 0u || image.height == 0u) {
        set_render_error();
        return ImageInfiniteSampleResult(vec2<f32>(0.0), 0.0, 0u);
    }
    let row_cdf_end = image.row_cdf_offset + image.height;
    let distribution_end = image.distribution_offset + image.width * image.height;
    if (row_cdf_end > arrayLength(&image_infinite_row_cdf)
        || distribution_end > arrayLength(&image_infinite_distribution)) {
        set_render_error();
        return ImageInfiniteSampleResult(vec2<f32>(0.0), 0.0, 0u);
    }
    let row_total = image_infinite_row_cdf[row_cdf_end - 1u];
    if (!(row_total > 0.0)) {
        return ImageInfiniteSampleResult(vec2<f32>(0.0), 0.0, 0u);
    }

    let sample_u = min(u, vec2<f32>(0.99999994));
    let row_target = sample_u.y * row_total;
    var row_first = 0u;
    var row_last = image.height;
    for (var iteration = 0u; iteration < 32u && row_first < row_last; iteration++) {
        let middle = row_first + (row_last - row_first) / 2u;
        let cdf = image_infinite_row_cdf[image.row_cdf_offset + middle];
        if (cdf <= row_target) {
            row_first = middle + 1u;
        } else {
            row_last = middle;
        }
    }
    let row = min(row_first, image.height - 1u);
    let row_cdf = image_infinite_row_cdf[image.row_cdf_offset + row];
    var previous_row_cdf = 0.0;
    if (row > 0u) {
        previous_row_cdf = image_infinite_row_cdf[image.row_cdf_offset + row - 1u];
    }
    let row_mass = row_cdf - previous_row_cdf;
    if (!(row_mass > 0.0)) {
        return ImageInfiniteSampleResult(vec2<f32>(0.0), 0.0, 0u);
    }
    let row_offset = image.distribution_offset + row * image.width;
    let conditional_total = image_infinite_distribution[row_offset + image.width - 1u].conditional_cdf;
    if (!(conditional_total > 0.0)) {
        return ImageInfiniteSampleResult(vec2<f32>(0.0), 0.0, 0u);
    }

    let column_target = sample_u.x * conditional_total;
    var column_first = 0u;
    var column_last = image.width;
    for (var iteration = 0u; iteration < 32u && column_first < column_last; iteration++) {
        let middle = column_first + (column_last - column_first) / 2u;
        let cdf = image_infinite_distribution[row_offset + middle].conditional_cdf;
        if (cdf <= column_target) {
            column_first = middle + 1u;
        } else {
            column_last = middle;
        }
    }
    let column = min(column_first, image.width - 1u);
    let selected = image_infinite_distribution[row_offset + column];
    var previous_column_cdf = 0.0;
    if (column > 0u) {
        previous_column_cdf = image_infinite_distribution[row_offset + column - 1u].conditional_cdf;
    }
    let column_mass = selected.conditional_cdf - previous_column_cdf;
    if (!(column_mass > 0.0)) {
        return ImageInfiniteSampleResult(vec2<f32>(0.0), 0.0, 0u);
    }
    let du = clamp((column_target - previous_column_cdf) / column_mass, 0.0, 1.0);
    let dv = clamp((row_target - previous_row_cdf) / row_mass, 0.0, 1.0);
    let uv = vec2<f32>(
        (f32(column) + du) / f32(image.width),
        (f32(row) + dv) / f32(image.height),
    );
    let pdf = selected.weight / row_total;
    if (!(pdf > 0.0) || pdf != pdf) {
        return ImageInfiniteSampleResult(vec2<f32>(0.0), 0.0, 0u);
    }
    return ImageInfiniteSampleResult(uv, pdf, 1u);
}

fn image_infinite_equal_area_square_to_sphere(uv: vec2<f32>) -> vec3<f32> {
    let u = 2.0 * uv.x - 1.0;
    let v = 2.0 * uv.y - 1.0;
    let up = abs(u);
    let vp = abs(v);
    let signed_distance = 1.0 - (up + vp);
    let d = abs(signed_distance);
    let r = 1.0 - d;
    var phi = 1.0;
    if (r != 0.0) {
        phi = (vp - up) / r + 1.0;
    }
    phi = phi * (PI / 4.0);
    let z_abs = 1.0 - r * r;
    let z = select(-z_abs, z_abs, signed_distance >= 0.0);
    let cos_phi = cos(phi);
    let sin_phi = sin(phi);
    let x = select(-abs(cos_phi), abs(cos_phi), u >= 0.0) * r * sqrt(max(0.0, 2.0 - r * r));
    let y = select(-abs(sin_phi), abs(sin_phi), v >= 0.0) * r * sqrt(max(0.0, 2.0 - r * r));
    return vec3<f32>(x, y, z);
}
