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

fn current_ray_count() -> u32 {
    return atomicLoad(&queue_counters.current.count);
}

fn next_ray_count() -> u32 {
    return atomicLoad(&queue_counters.next.count);
}

fn shadow_ray_count() -> u32 {
    return atomicLoad(&queue_counters.shadow.count);
}

fn append_shadow_ray(pixel_index: u32, origin: vec3<f32>, direction: vec3<f32>, t: f32, direct: vec4<f32>) {
    let index = atomicAdd(&queue_counters.shadow.count, 1u);
    if (index < queue_counters.shadow.capacity) {
        shadow_rays[index] = ShadowRayWorkItem(
            vec4<f32>(origin, 0.0),
            vec4<f32>(direction, 0.0),
            t,
            vec3<u32>(0u),
            direct,
            pixel_index,
            vec3<u32>(0u),
        );
    } else {
        atomicStore(&queue_counters.shadow.overflow, 1u);
    }
}

fn load_shadow_pixel(index: u32) -> u32 {
    return shadow_rays[index].pixel_index;
}

fn load_shadow_t(index: u32) -> f32 {
    return shadow_rays[index].max_t;
}

fn load_shadow_direct(index: u32) -> vec4<f32> {
    return shadow_rays[index].direct;
}

fn load_shadow_origin(index: u32) -> vec3<f32> {
    return shadow_rays[index].origin.xyz;
}

fn load_shadow_direction(index: u32) -> vec3<f32> {
    return shadow_rays[index].direction.xyz;
}

fn append_material_eval(ray_index: u32) {
    let index = atomicAdd(&queue_counters.material.count, 1u);
    if (index < queue_counters.material.capacity) {
        material_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.material.overflow, 1u);
    }
}

fn material_eval_count() -> u32 {
    return atomicLoad(&queue_counters.material.count);
}

fn load_material_eval_ray(index: u32) -> u32 {
    return material_ray_indices[index];
}

fn append_hit_area_light(ray_index: u32) {
    let index = atomicAdd(&queue_counters.hit_area.count, 1u);
    if (index < queue_counters.hit_area.capacity) {
        hit_area_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.hit_area.overflow, 1u);
    }
}

fn hit_area_light_count() -> u32 {
    return atomicLoad(&queue_counters.hit_area.count);
}

fn load_hit_area_ray(index: u32) -> u32 {
    return hit_area_ray_indices[index];
}

fn escaped_ray_count() -> u32 {
    return atomicLoad(&queue_counters.escaped.count);
}

fn append_escaped_ray(ray_index: u32) {
    let index = atomicAdd(&queue_counters.escaped.count, 1u);
    if (index < queue_counters.escaped.capacity) {
        escaped_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.escaped.overflow, 1u);
    }
}

fn load_current_ray(index: u32) -> RayWorkItem {
    return current_rays[index];
}

fn load_next_ray(index: u32) -> RayWorkItem {
    return next_rays[index];
}

fn store_current_ray(index: u32, ray: RayWorkItem) {
    current_rays[index] = ray;
}

fn store_next_ray(index: u32, ray: RayWorkItem) {
    next_rays[index] = ray;
}

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

fn load_area_distribution(index: u32, distribution_index: u32) -> AreaTriangleSelection {
    let total_area = load_area_total(index);
    let area = bitcast<f32>(load_area_distribution_word(index, distribution_index, 2u));
    return AreaTriangleSelection(
        load_area_distribution_word(index, distribution_index, 0u),
        area,
        area / total_area,
    );
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
    return load_area_distribution(index, min(first, count - 1u));
}

fn load_area_two_sided(index: u32) -> bool {
    return (load_area_word(index, 7u) & 1u) != 0u;
}

fn load_point_position(index: u32) -> vec3<f32> {
    let model = light_sampling_models[index];
    if (model.geometry_index >= arrayLength(&light_positions)) { set_render_error(); return vec3<f32>(0.0); }
    return light_positions[model.geometry_index].xyz;
}

fn pixel_count() -> u32 {
    return viewport.width * viewport.height;
}

fn load_material_kind(index: u32) -> u32 {
    if (material_table.debug_scattering_model != 0xffffffffu) {
        return material_table.debug_scattering_model;
    }
    let model_index = load_material_model(index);
    if (model_index >= material_table.scattering_model_count) {
        set_render_error();
        return MATERIAL_KIND_NORMAL;
    }
    if (model_index >= arrayLength(&scattering_models)) {
        set_render_error();
        return MATERIAL_KIND_NORMAL;
    }
    let node_index = scattering_models[model_index].surface_root;
    if (node_index >= arrayLength(&scattering_nodes)) {
        set_render_error();
        return MATERIAL_KIND_NORMAL;
    }
    let node_kind = scattering_nodes[node_index].kind_tag;
    if (node_kind == 0u) {
        return MATERIAL_KIND_DIFFUSE;
    }
    if (node_kind == 1u) {
        return MATERIAL_KIND_DIELECTRIC;
    }
    if (node_kind == 3u) {
        return MATERIAL_KIND_THIN_DIELECTRIC;
    }
    if (node_kind == 2u) {
        return MATERIAL_KIND_LAYERED;
    }
    if (node_kind == 4u) {
        return MATERIAL_KIND_CONDUCTOR;
    }
    set_render_error();
    return MATERIAL_KIND_NORMAL;
}

fn load_material_attribute(node_index: u32, ordinal: u32) -> AttributeRef {
    if (node_index >= arrayLength(&scattering_nodes)) { set_render_error(); return AttributeRef(0u, 0u); }
    let node = scattering_nodes[node_index];
    if (ordinal >= node.attribute_count || node.attribute_offset + ordinal >= arrayLength(&attribute_refs)) { set_render_error(); return AttributeRef(0u, 0u); }
    return attribute_refs[node.attribute_offset + ordinal];
}
fn load_node_scalar(node_index: u32, ordinal: u32) -> f32 {
    let attr_ref = load_material_attribute(node_index, ordinal);
    if (attr_ref.kind != 0u || attr_ref.index >= arrayLength(&scalar_attributes)) { set_render_error(); return 0.0; }
    return scalar_attributes[attr_ref.index];
}
fn load_node_spectrum(node_index: u32, ordinal: u32, lambda: vec4<f32>) -> vec4<f32> {
    let attr_ref = load_material_attribute(node_index, ordinal);
    if (attr_ref.kind != 1u) { set_render_error(); return vec4<f32>(0.0); }
    return evaluate_spectrum(attr_ref.index, lambda);
}
fn load_material_surface_node(index: u32) -> u32 {
    let model = load_material_model(index);
    if (model >= arrayLength(&scattering_models)) { set_render_error(); return 0u; }
    return scattering_models[model].surface_root;
}
fn load_material_model(index: u32) -> u32 {
    if (index >= arrayLength(&materials)) { set_render_error(); return 0u; }
    return materials[index].scattering_model;
}
fn load_diffuse_reflectance(material_index: u32, lambda: vec4<f32>) -> vec4<f32> {
    if (material_table.debug_scattering_model == MATERIAL_KIND_LAMBERT) { return vec4<f32>(0.5); }
    let model = load_material_model(material_index);
    return load_node_spectrum(scattering_models[model].surface_root, 0u, lambda);
}
fn load_dielectric_eta(node_index: u32, lambda: vec4<f32>) -> vec4<f32> { return load_node_spectrum(node_index, 0u, lambda); }
fn dielectric_eta_is_constant(node_index: u32) -> bool { return spectrum_is_constant(load_material_attribute(node_index, 0u).index); }
fn load_conductor_eta(node_index: u32, lambda: vec4<f32>) -> vec4<f32> { return load_node_spectrum(node_index, 0u, lambda); }
fn load_conductor_k(node_index: u32, lambda: vec4<f32>) -> vec4<f32> { return load_node_spectrum(node_index, 1u, lambda); }
fn load_conductor_roughness(node_index: u32) -> f32 { return load_node_scalar(node_index, 2u); }
fn conductor_fresnel(cosine_input: f32, eta: vec4<f32>, k: vec4<f32>) -> vec4<f32> {
    let c = clamp(abs(cosine_input), 0.0, 1.0);
    let c2 = c * c;
    let s2 = 1.0 - c2;
    let eta2 = eta * eta;
    let k2 = k * k;
    let t0 = eta2 - k2 - vec4<f32>(s2);
    let a2b2 = sqrt(t0 * t0 + 4.0 * eta2 * k2);
    let a = sqrt(max(vec4<f32>(0.0), 0.5 * (a2b2 + t0)));
    let t1 = a2b2 + vec4<f32>(c2);
    let t2 = 2.0 * c * a;
    let rs = (t1 - t2) / max(t1 + t2, vec4<f32>(1e-7));
    let t3 = c2 * a2b2 + vec4<f32>(s2 * s2);
    let t4 = t2 * s2;
    let rp = rs * (t3 - t4) / max(t3 + t4, vec4<f32>(1e-7));
    return 0.5 * (rs + rp);
}
fn load_scattering_node_word(index: u32, word: u32) -> u32 {
    if (index >= arrayLength(&scattering_nodes)) { set_render_error(); return 0u; }
    let node = scattering_nodes[index];
    if (word == 0u) { return node.kind_tag; } if (word == 1u) { return node.event_flags; }
    if (word == 2u) { return node.attribute_offset; } if (word == 3u) { return node.child_offset; }
    if (word == 4u) { return node.child_count; } if (word == 5u) { return node.attribute_count; } return 0u;
}
fn load_scattering_child(node_index: u32, child_index: u32) -> u32 {
    let child_count = load_scattering_node_word(node_index, 4u);
    if (child_index >= child_count) { set_render_error(); return 0u; }
    return scattering_children[load_scattering_node_word(node_index, 3u) + child_index];
}
fn load_layered_bxdf(material_index: u32, lambda: vec4<f32>) -> LayeredParams {
    let root = scattering_models[load_material_model(material_index)].surface_root;
    return LayeredParams(load_node_scalar(root, 0u), load_node_scalar(root, 1u), u32(max(load_node_scalar(root, 2u), 0.0)), u32(max(load_node_scalar(root, 3u), 0.0)), load_node_spectrum(root, 5u, lambda), u32(max(load_node_scalar(root, 4u), 0.0)), 0u, 0u, 0u);
}
fn load_layered_bottom_reflectance(material_index: u32, lambda: vec4<f32>) -> vec4<f32> {
    let root = scattering_models[load_material_model(material_index)].surface_root;
    let bottom = load_scattering_child(root, 1u);
    if (load_scattering_node_word(bottom, 0u) != 0u) { set_render_error(); return vec4<f32>(0.0); }
    return load_node_spectrum(bottom, 7u, lambda);
}
fn load_layered_eta_node(material_index: u32) -> u32 {
    let root = scattering_models[load_material_model(material_index)].surface_root;
    let top = load_scattering_child(root, 0u);
    if (load_scattering_node_word(top, 0u) != 1u) { set_render_error(); return 0u; }
    return top;
}
fn load_layered_eta(material_index: u32, lambda: vec4<f32>) -> vec4<f32> { return load_node_spectrum(load_layered_eta_node(material_index), 6u, lambda); }
fn layered_eta_is_constant(material_index: u32) -> bool { return spectrum_is_constant(load_material_attribute(load_layered_eta_node(material_index), 6u).index); }

fn layered_path_seed(pixel: u32, depth: u32) -> u32 {
    return path_hash(path_hash(path_hash(viewport.seed) ^ pixel)
        ^ path_hash(viewport.sample_index)) ^ path_hash(depth);
}

fn path_hash(value: u32) -> u32 {
    var x = value;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn scattering_local(w: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let t = make_tangent(n);
    return vec3<f32>(dot(w, t), dot(w, cross(n, t)), dot(w, n));
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
fn load_light_scale(index: u32) -> f32 {
    let attr = load_light_attribute(index, 1u);
    if (attr.kind != 0u || attr.index >= arrayLength(&scalar_attributes)) { set_render_error(); return 0.0; }
    return scalar_attributes[attr.index];
}

fn uniform_light_pmf_for_handle(light_handle: u32) -> f32 {
    if (light_handle >= light_table.light_count || light_table.light_leaf_offset == 0xffffffffu) {
        return 0.0;
    }
    if (light_bvh_leaves[light_handle] == 0xffffffffu) {
        return 0.0;
    }
    var count = 0u;
    for (var handle = 0u; handle < light_table.light_leaf_count; handle++) {
        if (light_bvh_leaves[handle] != 0xffffffffu) {
            count = count + 1u;
        }
    }
    if (count == 0u) {
        return 0.0;
    }
    return 1.0 / f32(count);
}

fn hash_u32(value: u32) -> u32 {
    var h = value;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

fn random01(pixel_index: u32, dimension: u32, depth: u32) -> f32 {
    let value = viewport.seed
        ^ (pixel_index * 0x9e3779b9u)
        ^ (viewport.sample_index * 0x85ebca6bu)
        ^ ((dimension + depth * 8u) * 0xc2b2ae35u);
    return f32(hash_u32(value) & 0x00ffffffu) / 16777216.0;
}

fn generate_ray_samples(pixel_index: u32, depth: u32) -> RaySamples {
    return RaySamples(
        vec4<f32>(
            random01(pixel_index, 2u, depth),
            random01(pixel_index, 3u, depth),
            random01(pixel_index, 4u, depth),
            0.0,
        ),
        vec4<f32>(
            random01(pixel_index, 5u, depth),
            random01(pixel_index, 6u, depth),
            random01(pixel_index, 7u, depth),
            random01(pixel_index, 8u, depth),
        ),
    );
}

fn sample_uniform_light(selector: f32) -> LightSelection {
    if (light_table.light_leaf_offset == 0xffffffffu || light_table.light_leaf_count == 0u) {
        return LightSelection(0xffffffffu, 0.0);
    }
    var count = 0u;
    for (var handle = 0u; handle < light_table.light_leaf_count; handle++) {
        if (light_bvh_leaves[handle] != 0xffffffffu) {
            count = count + 1u;
        }
    }
    if (count == 0u) {
        return LightSelection(0xffffffffu, 0.0);
    }
    let selected = min(u32(min(selector, 0.99999994) * f32(count)), count - 1u);
    var ordinal = 0u;
    for (var handle = 0u; handle < light_table.light_leaf_count; handle++) {
            if (light_bvh_leaves[handle] != 0xffffffffu) {
            if (ordinal == selected) {
                return LightSelection(handle, 1.0 / f32(count));
            }
            ordinal = ordinal + 1u;
        }
    }
    return LightSelection(0xffffffffu, 0.0);
}

fn light_bvh_word(node_index: u32, word: u32) -> u32 {
    return light_bvh_nodes[node_index * 8u + word];
}

fn decode_light_bvh_node(node_index: u32) -> DecodedLightBVHNode {
    let word0 = light_bvh_word(node_index, 0u);
    let word1 = light_bvh_word(node_index, 1u);
    let word2 = light_bvh_word(node_index, 2u);
    let q_min = vec3<u32>(word0 & 0xffffu, word0 >> 16u, word1 & 0xffffu);
    let q_max = vec3<u32>(word1 >> 16u, word2 & 0xffffu, word2 >> 16u);
    let all_min = vec3<f32>(
        bitcast<f32>(light_bvh_header[0u]),
        bitcast<f32>(light_bvh_header[1u]),
        bitcast<f32>(light_bvh_header[2u]),
    );
    let all_max = vec3<f32>(
        bitcast<f32>(light_bvh_header[4u]),
        bitcast<f32>(light_bvh_header[5u]),
        bitcast<f32>(light_bvh_header[6u]),
    );
    let extent = all_max - all_min;
    let bounds_min = all_min + vec3<f32>(q_min) / 65535.0 * extent;
    let bounds_max = all_min + vec3<f32>(q_max) / 65535.0 * extent;
    let direction_word = light_bvh_word(node_index, 3u);
    let encoded = vec2<f32>(
        f32(direction_word & 0xffffu) / 65535.0 * 2.0 - 1.0,
        f32(direction_word >> 16u) / 65535.0 * 2.0 - 1.0,
    );
    var direction = vec3<f32>(encoded.x, encoded.y, 1.0 - abs(encoded.x) - abs(encoded.y));
    if (direction.z < 0.0) {
        direction = vec3<f32>(
            (1.0 - abs(direction.y)) * select(-1.0, 1.0, direction.x >= 0.0),
            (1.0 - abs(direction.x)) * select(-1.0, 1.0, direction.y >= 0.0),
            direction.z,
        );
    }
    direction = normalize(direction);
    let cosine_word = light_bvh_word(node_index, 5u);
    let cosine_o = f32(cosine_word & 0x7fffu) / 32767.0 * 2.0 - 1.0;
    let cosine_e = f32((cosine_word >> 15u) & 0x7fffu) / 32767.0 * 2.0 - 1.0;
    let payload_word = light_bvh_word(node_index, 6u);
    return DecodedLightBVHNode(
        bounds_min,
        bounds_max,
        direction,
        bitcast<f32>(light_bvh_word(node_index, 4u)),
        cosine_o,
        cosine_e,
        (cosine_word & 0x40000000u) != 0u,
        payload_word & 0x7fffffffu,
        (payload_word & 0x80000000u) != 0u,
    );
}

fn light_bvh_importance(node: DecodedLightBVHNode, p: vec3<f32>, n: vec3<f32>) -> f32 {
    let center = (node.bounds_min + node.bounds_max) * 0.5;
    let diagonal = node.bounds_max - node.bounds_min;
    let delta = p - center;
    var d2 = max(dot(delta, delta), length(diagonal) * 0.5);
    if (d2 <= 0.0) {
        return 0.0;
    }
    let radius = 0.5 * length(diagonal);
    // Match pbrt-v4 LightBounds::importance: wi points from the light
    // bound's center toward the reference point.
    let center_to_point = p - center;
    let center_distance_squared = dot(center_to_point, center_to_point);
    var cos_theta_b = -1.0;
    if (center_distance_squared > radius * radius && center_distance_squared > 0.0) {
        cos_theta_b = sqrt(max(0.0, 1.0 - radius * radius / center_distance_squared));
    }
    let wi = normalize(center_to_point);
    var cos_theta_w = dot(node.direction, wi);
    if (node.two_sided) {
        cos_theta_w = abs(cos_theta_w);
    }
    let sin_theta_w = sqrt(max(0.0, 1.0 - cos_theta_w * cos_theta_w));
    let sin_theta_o = sqrt(max(0.0, 1.0 - node.cos_theta_o * node.cos_theta_o));
    let sin_theta_b = sqrt(max(0.0, 1.0 - cos_theta_b * cos_theta_b));
    let cos_theta_x = cos_sub_clamped(
        sin_theta_w,
        cos_theta_w,
        sin_theta_o,
        node.cos_theta_o,
    );
    let sin_theta_x = sin_sub_clamped(
        sin_theta_w,
        cos_theta_w,
        sin_theta_o,
        node.cos_theta_o,
    );
    let cos_theta_p = cos_sub_clamped(sin_theta_x, cos_theta_x, sin_theta_b, cos_theta_b);
    if (cos_theta_p <= node.cos_theta_e) {
        return 0.0;
    }
    var importance = node.phi * cos_theta_p / d2;
    if (dot(n, n) != 0.0) {
        let cos_theta_i = abs(dot(wi, normalize(n)));
        let sin_theta_i = sqrt(max(0.0, 1.0 - cos_theta_i * cos_theta_i));
        importance = importance
            * cos_sub_clamped(sin_theta_i, cos_theta_i, sin_theta_b, cos_theta_b);
    }
    return max(importance, 0.0);
}

fn cos_sub_clamped(sin_a: f32, cos_a: f32, sin_b: f32, cos_b: f32) -> f32 {
    if (cos_a > cos_b) {
        return 1.0;
    }
    return cos_a * cos_b + sin_a * sin_b;
}

fn sin_sub_clamped(sin_a: f32, cos_a: f32, sin_b: f32, cos_b: f32) -> f32 {
    if (cos_a > cos_b) {
        return 0.0;
    }
    return sin_a * cos_b - cos_a * sin_b;
}

fn sample_light_bvh(selector: f32, p: vec3<f32>, n: vec3<f32>) -> LightSelection {
    if (light_table.light_bvh_node_count == 0u || light_table.light_leaf_count == 0u) {
        return LightSelection(0xffffffffu, 0.0);
    }
    var node_index = 0u;
    var pmf = 1.0;
    var u = min(selector, 0.99999994);
    for (var iteration = 0u; iteration < light_table.light_bvh_node_count; iteration++) {
        let node = decode_light_bvh_node(node_index);
        if (node.is_leaf) {
            return LightSelection(node.payload, pmf);
        }
        let left = decode_light_bvh_node(node_index + 1u);
        let right = decode_light_bvh_node(node.payload);
        let left_weight = light_bvh_importance(left, p, n);
        let right_weight = light_bvh_importance(right, p, n);
        let total = left_weight + right_weight;
        if (total <= 0.0) {
            return LightSelection(0xffffffffu, 0.0);
        }
        let left_pmf = left_weight / total;
        if (u < left_pmf) {
            pmf = pmf * left_pmf;
            u = u / max(left_pmf, 1e-7);
            node_index = node_index + 1u;
        } else {
            pmf = pmf * (1.0 - left_pmf);
            u = (u - left_pmf) / max(1.0 - left_pmf, 1e-7);
            node_index = node.payload;
        }
        if (node_index >= light_table.light_bvh_node_count) {
            return LightSelection(0xffffffffu, 0.0);
        }
    }
    return LightSelection(0xffffffffu, 0.0);
}

fn light_bvh_pmf_for_handle(light_handle: u32, p: vec3<f32>, n: vec3<f32>) -> f32 {
    if (light_handle >= light_table.light_leaf_count) {
        return 0.0;
    }
    let leaf_index = light_bvh_leaves[light_handle];
    if (leaf_index >= light_table.light_bvh_node_count) {
        return 0.0;
    }
    var node_index = leaf_index;
    var pmf = 1.0;
    for (var iteration = 0u; iteration < light_table.light_bvh_node_count; iteration++) {
        let parent = light_bvh_word(node_index, 7u);
        if (parent == 0xffffffffu) {
            return pmf;
        }
        if (parent >= light_table.light_bvh_node_count) {
            return 0.0;
        }
        let parent_node = decode_light_bvh_node(parent);
        let left_child = parent + 1u;
        let left = decode_light_bvh_node(left_child);
        let right = decode_light_bvh_node(parent_node.payload);
        let left_weight = light_bvh_importance(left, p, n);
        let right_weight = light_bvh_importance(right, p, n);
        let total = left_weight + right_weight;
        if (total <= 0.0) {
            return 0.0;
        }
        if (node_index == left_child) {
            pmf = pmf * left_weight / total;
        } else if (node_index == parent_node.payload) {
            pmf = pmf * right_weight / total;
        } else {
            return 0.0;
        }
        node_index = parent;
    }
    return 0.0;
}

fn light_pmf_for_handle(light_handle: u32, p: vec3<f32>, n: vec3<f32>) -> f32 {
    if (light_table.light_sampler_kind == LIGHT_SAMPLER_KIND_BVH) {
        return light_bvh_pmf_for_handle(light_handle, p, n);
    }
    return uniform_light_pmf_for_handle(light_handle);
}

fn sample_scene_light(selector: f32, p: vec3<f32>, n: vec3<f32>) -> LightSelection {
    if (light_table.light_sampler_kind == LIGHT_SAMPLER_KIND_BVH) {
        return sample_light_bvh(selector, p, n);
    }
    return sample_uniform_light(selector);
}

fn gamma(n: f32) -> f32 {
    return (n * MACHINE_EPSILON) / (1.0 - n * MACHINE_EPSILON);
}

fn next_float_up(value: f32) -> f32 {
    if (value == 0.0) {
        return bitcast<f32>(1u);
    }
    let bits = bitcast<u32>(value);
    if (value < 0.0) {
        return bitcast<f32>(bits - 1u);
    }
    return bitcast<f32>(bits + 1u);
}

fn next_float_down(value: f32) -> f32 {
    if (value == 0.0) {
        return bitcast<f32>(0x80000001u);
    }
    let bits = bitcast<u32>(value);
    if (value > 0.0) {
        return bitcast<f32>(bits - 1u);
    }
    return bitcast<f32>(bits + 1u);
}

fn offset_ray_origin(position: vec3<f32>, error: vec3<f32>, normal: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    let offset = normal * dot(abs(normal), error);
    let signed_offset = select(-offset, offset, dot(direction, normal) >= 0.0);
    var result = position + signed_offset;
    if (signed_offset.x > 0.0) {
        result.x = next_float_up(result.x);
    } else if (signed_offset.x < 0.0) {
        result.x = next_float_down(result.x);
    }
    if (signed_offset.y > 0.0) {
        result.y = next_float_up(result.y);
    } else if (signed_offset.y < 0.0) {
        result.y = next_float_down(result.y);
    }
    if (signed_offset.z > 0.0) {
        result.z = next_float_up(result.z);
    } else if (signed_offset.z < 0.0) {
        result.z = next_float_down(result.z);
    }
    return result;
}

fn make_tangent(normal: vec3<f32>) -> vec3<f32> {
    if (abs(normal.x) > 0.1) {
        return normalize(cross(vec3<f32>(0.0, 1.0, 0.0), normal));
    }
    return normalize(cross(vec3<f32>(1.0, 0.0, 0.0), normal));
}
