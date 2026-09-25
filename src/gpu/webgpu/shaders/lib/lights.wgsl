fn load_area_word(index: u32, word: u32) -> u32 {
    let area = light_sampling_models[index];
    if (word == 0u) { return area.geometry_index; }
    if (word == 1u) { return area.distribution_offset_words; }
    if (word == 2u) { return area.distribution_count; }
    if (word == 3u) { return bitcast<u32>(area.total_area); }
    if (word == 7u) { return area.flags; }
    return 0u;
}

fn load_area_instance(index: u32) -> u32 {
    return load_area_word(index, 0u);
}

fn load_area_total(index: u32) -> f32 {
    return bitcast<f32>(load_area_word(index, 3u));
}

fn load_area_distribution_count(index: u32) -> u32 {
    return load_area_word(index, 2u);
}

fn load_area_distribution_word(index: u32, distribution_index: u32, word: u32) -> u32 {
    let entry = triangle_distributions[load_area_word(index, 1u) + distribution_index];
    if (word == 0u) { return entry.primitive; }
    if (word == 1u) { return bitcast<u32>(entry.cdf); }
    if (word == 2u) { return bitcast<u32>(entry.area); }
    return 0u;
}

fn load_area_distribution_remapped(index: u32, distribution_index: u32, u_remapped: f32) -> AreaTriangleSelection {
    let total_area = load_area_total(index);
    let area = bitcast<f32>(load_area_distribution_word(index, distribution_index, 2u));
    return AreaTriangleSelection(
        load_area_distribution_word(index, distribution_index, 0u),
        area,
        area / total_area,
        u_remapped,
    );
}

fn load_area_distribution(index: u32, distribution_index: u32) -> AreaTriangleSelection {
    return load_area_distribution_remapped(index, distribution_index, 0.0);
}

fn select_area_triangle(index: u32, u: f32) -> AreaTriangleSelection {
    let count = load_area_distribution_count(index);
    var first = 0u;
    var last = count;
    let clamped_u = min(u, 0.99999994);
    for (var iteration = 0u; iteration < 32u && first < last; iteration++) {
        let middle = (first + last) / 2u;
        let cdf = bitcast<f32>(load_area_distribution_word(index, middle, 1u));
        if (clamped_u < cdf) {
            last = middle;
        } else {
            first = middle + 1u;
        }
    }
    let selected = min(first, count - 1u);
    var previous_cdf = 0.0;
    if (selected > 0u) {
        previous_cdf = bitcast<f32>(load_area_distribution_word(index, selected - 1u, 1u));
    }
    let selected_cdf = bitcast<f32>(load_area_distribution_word(index, selected, 1u));
    let u_remapped = (clamped_u - previous_cdf) / (selected_cdf - previous_cdf);
    return load_area_distribution_remapped(index, selected, u_remapped);
}

fn area_light_is_two_sided(index: u32) -> bool {
    return (load_area_word(index, 7u) & AREA_LIGHT_FLAG_TWO_SIDED) != 0u;
}

fn load_point_position(index: u32) -> vec3<f32> {
    if (index >= arrayLength(&light_records)) { set_render_error(); return vec3<f32>(0.0); }
    let model_index = light_records[index].sampling_model;
    if (model_index >= arrayLength(&light_sampling_models)) { set_render_error(); return vec3<f32>(0.0); }
    let model = light_sampling_models[model_index];
    if (model.geometry_index >= arrayLength(&light_positions)) { set_render_error(); return vec3<f32>(0.0); }
    return light_positions[model.geometry_index].xyz;
}

fn load_light_direction(index: u32) -> vec3<f32> {
    if (index >= arrayLength(&light_records)) { set_render_error(); return vec3<f32>(0.0); }
    let model_index = light_records[index].sampling_model;
    if (model_index >= arrayLength(&light_sampling_models)) { set_render_error(); return vec3<f32>(0.0); }
    let model = light_sampling_models[model_index];
    if (model.direction_index >= arrayLength(&light_positions)) { set_render_error(); return vec3<f32>(0.0); }
    return light_positions[model.direction_index].xyz;
}

fn area_light_is_zero_alpha_sample_only(index: u32) -> bool {
    return (load_area_word(index, 7u) & AREA_LIGHT_FLAG_ZERO_ALPHA_SAMPLE_ONLY) != 0u;
}

fn load_light_kind(index: u32) -> u32 {
    return light_records[index].kind;
}

fn load_light_payload(index: u32) -> u32 {
    return light_records[index].sampling_model;
}

fn load_light_attribute(index: u32, ordinal: u32) -> AttributeRef {
    if (index >= arrayLength(&light_records)) { set_render_error(); return AttributeRef(0u, 0u); }
    let record = light_records[index];
    if (ordinal >= record.attribute_count || record.attribute_offset + ordinal >= arrayLength(&attribute_refs)) { set_render_error(); return AttributeRef(0u, 0u); }
    return attribute_refs[record.attribute_offset + ordinal];
}

fn load_light_spectrum(index: u32, ordinal: u32, lambda: vec4<f32>) -> vec4<f32> {
    let attr = load_light_attribute(index, ordinal);
    if (attr.kind != 1u) { set_render_error(); return vec4<f32>(0.0); }
    return evaluate_spectrum(attr.index, lambda);
}

fn load_light_image_index(index: u32) -> u32 {
    let record = light_records[index];
    if (record.sampling_model >= arrayLength(&light_sampling_models)) {
        set_render_error();
        return 0xffffffffu;
    }
    return light_sampling_models[record.sampling_model].flags & 0x0fffffffu;
}

fn equal_area_sphere_to_square(direction: vec3<f32>) -> vec2<f32> {
    let d = normalize(direction);
    let x = abs(d.x);
    let y = abs(d.y);
    let z = abs(d.z);
    let r = sqrt(max(0.0, 1.0 - z));
    let a = max(x, y);
    let b = select(min(x, y) / a, 0.0, a == 0.0);
    let t1 = 0.406758566246788489601959989e-5;
    let t2 = 0.636226545274016134946890922156;
    let t3 = 0.61572017898280213493197203466e-2;
    let t4 = -0.247333733281268944196501420480;
    let t5 = 0.881770664775316294736387951347e-1;
    let t6 = 0.419038818029165735901852432784e-1;
    let t7 = -0.251390972343483509333252996350e-1;
    var phi = ((((((t7 * b + t6) * b + t5) * b + t4) * b + t3) * b + t2) * b + t1);
    if (x < y) { phi = 1.0 - phi; }
    var v = phi * r;
    var u = r - v;
    if (d.z < 0.0) {
        let old_u = u;
        u = 1.0 - v;
        v = 1.0 - old_u;
    }
    u = select(-u, u, d.x >= 0.0);
    v = select(-v, v, d.y >= 0.0);
    return vec2<f32>(0.5 * (u + 1.0), 0.5 * (v + 1.0));
}

fn load_light_image_spectrum(index: u32, direction: vec3<f32>, lambda: vec4<f32>) -> vec4<f32> {
    let image_index = load_light_image_index(index);
    if (image_index == 0xffffffffu) {
        return load_light_spectrum(index, 0u, lambda);
    }
    let record = light_records[index];
    let model = light_sampling_models[record.sampling_model];
    let d = normalize(vec3<f32>(
        dot(model.world_to_light0.xyz, direction),
        dot(model.world_to_light1.xyz, direction),
        dot(model.world_to_light2.xyz, direction),
    ));
    let uv = equal_area_sphere_to_square(d);
    let rgb = textureSampleLevel(texture_images[image_index], texture_samplers[0], uv, 0.0).rgb;
    let color_space = model.flags >> 28u;
    let illuminant = load_light_spectrum(index, 2u, lambda);
    return rgb_to_unbounded_spectrum4(max(rgb, vec3<f32>(0.0)), lambda, color_space) * illuminant;
}

fn load_light_scale(index: u32) -> f32 {
    let attr = load_light_attribute(index, 1u);
    if (attr.kind != 0u || attr.index >= arrayLength(&scalar_attributes)) { set_render_error(); return 0.0; }
    return scalar_attributes[attr.index];
}

fn load_light_scalar(index: u32, ordinal: u32) -> f32 {
    let attr = load_light_attribute(index, ordinal);
    if (attr.kind != 0u || attr.index >= arrayLength(&scalar_attributes)) { set_render_error(); return 0.0; }
    return scalar_attributes[attr.index];
}
