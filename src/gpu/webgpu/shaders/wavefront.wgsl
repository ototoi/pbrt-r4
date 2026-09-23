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
            0u, 0u, 0u,
            direct,
            pixel_index,
            0u, 0u, 0u,
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

fn append_material_eval(ray_index: u32) -> u32 {
    let index = atomicAdd(&queue_counters.material.count, 1u);
    if (index < queue_counters.material.capacity) {
        material_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.material.overflow, 1u);
    }
    return index;
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

fn load_area_two_sided(index: u32) -> bool {
    return (load_area_word(index, 7u) & 1u) != 0u;
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

fn pixel_count() -> u32 {
    return viewport.width * viewport.height;
}

fn load_material_kind(index: u32) -> u32 {
    if (material_table.debug_material_kind != 0xffffffffu) {
        return material_table.debug_material_kind;
    }
    if (index >= arrayLength(&material_nodes) || index >= material_table.material_node_count) {
        set_render_error();
        return MATERIAL_KIND_NORMAL;
    }
    return material_nodes[index].kind;
}

fn load_material_attribute(material_node: u32, ordinal: u32) -> AttributeRef {
    if (material_node >= arrayLength(&material_nodes) || material_node >= material_table.material_node_count) { set_render_error(); return AttributeRef(0u, 0u); }
    let material = material_nodes[material_node];
    if (ordinal >= material.attribute_count || material.attribute_offset + ordinal >= arrayLength(&attribute_refs)) { set_render_error(); return AttributeRef(0u, 0u); }
    return attribute_refs[material.attribute_offset + ordinal];
}
fn load_texture_eval_result(material_node: u32, ordinal: u32, texture_root: u32) -> TextureEvalResult {
    for (var slot = 0u; slot < material_table.texture_eval_stride; slot++) {
        let result = texture_eval_results[material_texture_eval_base + slot];
        if (result.valid != 0u
            && result.material_node == material_node
            && result.attribute_ordinal == ordinal) {
            return result;
        }
        if (result.valid != 0u && result.texture_root == texture_root) {
            return result;
        }
    }
    set_render_error();
    return TextureEvalResult(0u, 0u, 0u, 0u, vec4<f32>(0.0));
}
fn load_material_scalar(material_node: u32, ordinal: u32) -> f32 {
    let attr_ref = load_material_attribute(material_node, ordinal);
    if (attr_ref.kind == 2u) {
        return load_texture_eval_result(material_node, ordinal, attr_ref.index).value.x;
    }
    if (attr_ref.kind != 0u || attr_ref.index >= arrayLength(&scalar_attributes)) { set_render_error(); return 0.0; }
    return scalar_attributes[attr_ref.index];
}

fn texture_noise_weight(t: f32) -> f32 {
    let t3 = t * t * t;
    let t4 = t3 * t;
    return 6.0 * t4 * t - 15.0 * t4 + 10.0 * t3;
}

fn texture_noise_perm(i: u32) -> u32 {
    return textureLoad(texture_noise_table, vec2<i32>(i32(i & 255u), 0), 0).x;
}

fn texture_noise_pair(a: u32, b: u32) -> u32 {
    return textureLoad(
        texture_noise_table,
        vec2<i32>(i32(a & 255u), i32((b & 255u) + 1u)),
        0,
    ).x;
}

fn texture_noise_grad(x: u32, y: u32, z: u32, dx: f32, dy: f32, dz: f32) -> f32 {
    let a = texture_noise_pair(x, y);
    let h = texture_noise_perm(a + z) & 15u;
    let u = select(dy, dx, h < 8u || h == 12u || h == 13u);
    let v = select(dz, dy, h < 4u || h == 12u || h == 13u);
    return select(u, -u, (h & 1u) != 0u) + select(v, -v, (h & 2u) != 0u);
}

fn texture_noise(p: vec3<f32>) -> f32 {
    let ix = i32(floor(p.x));
    let iy = i32(floor(p.y));
    let iz = i32(floor(p.z));
    let dx = p.x - f32(ix);
    let dy = p.y - f32(iy);
    let dz = p.z - f32(iz);
    let x = u32(ix) & 255u;
    let y = u32(iy) & 255u;
    let z = u32(iz) & 255u;
    let w000 = texture_noise_grad(x, y, z, dx, dy, dz);
    let w100 = texture_noise_grad((x + 1u) & 255u, y, z, dx - 1.0, dy, dz);
    let w010 = texture_noise_grad(x, (y + 1u) & 255u, z, dx, dy - 1.0, dz);
    let w110 = texture_noise_grad((x + 1u) & 255u, (y + 1u) & 255u, z, dx - 1.0, dy - 1.0, dz);
    let w001 = texture_noise_grad(x, y, (z + 1u) & 255u, dx, dy, dz - 1.0);
    let w101 = texture_noise_grad((x + 1u) & 255u, y, (z + 1u) & 255u, dx - 1.0, dy, dz - 1.0);
    let w011 = texture_noise_grad(x, (y + 1u) & 255u, (z + 1u) & 255u, dx, dy - 1.0, dz - 1.0);
    let w111 = texture_noise_grad((x + 1u) & 255u, (y + 1u) & 255u, (z + 1u) & 255u, dx - 1.0, dy - 1.0, dz - 1.0);
    let wx = texture_noise_weight(dx);
    let wy = texture_noise_weight(dy);
    let wz = texture_noise_weight(dz);
    let x0 = mix(w000, w100, wx);
    let x1 = mix(w010, w110, wx);
    let x2 = mix(w001, w101, wx);
    let x3 = mix(w011, w111, wx);
    return mix(mix(x0, x1, wy), mix(x2, x3, wy), wz);
}

fn texture_fbm(p: vec3<f32>, omega: f32, octaves: f32, absolute_value: bool) -> f32 {
    let count = u32(clamp(floor(octaves), 0.0, 8.0));
    var sum = 0.0;
    var frequency = 1.0;
    var weight = 1.0;
    for (var octave = 0u; octave < count; octave++) {
        let value = texture_noise(frequency * p);
        sum += weight * select(value, abs(value), absolute_value);
        frequency *= 1.99;
        weight *= omega;
    }
    if (absolute_value) {
        var tail_weight = weight;
        for (var tail = 0u; tail < count; tail++) {
            sum += tail_weight * 0.2;
            tail_weight *= omega;
        }
    }
    return sum;
}

fn texture_marble(p: vec3<f32>, omega: f32, octaves: f32, scale: f32, variation: f32) -> vec3<f32> {
    let q = scale * p;
    let marble = q.y + variation * texture_fbm(q, omega, octaves, false);
    var t = 0.5 + 0.5 * sin(marble);
    let colors = array<vec3<f32>, 9>(
        vec3<f32>(0.58, 0.58, 0.60), vec3<f32>(0.58, 0.58, 0.60),
        vec3<f32>(0.58, 0.58, 0.60), vec3<f32>(0.50, 0.50, 0.50),
        vec3<f32>(0.60, 0.59, 0.58), vec3<f32>(0.58, 0.58, 0.60),
        vec3<f32>(0.58, 0.58, 0.60), vec3<f32>(0.20, 0.20, 0.33),
        vec3<f32>(0.58, 0.58, 0.60),
    );
    let segments = 6;
    let scaled_t = t * f32(segments);
    let first = min(u32(floor(scaled_t)), 5u);
    t = scaled_t - f32(first);
    let c0 = colors[first];
    let c1 = colors[first + 1u];
    let c2 = colors[first + 2u];
    let c3 = colors[first + 3u];
    let a = mix(c0, c1, t);
    let b = mix(c1, c2, t);
    let c = mix(c2, c3, t);
    let d = mix(a, b, t);
    let e = mix(b, c, t);
    return 1.5 * mix(d, e, t);
}

fn mapped_texture_uv(node: TextureNodeRecord, uv: vec2<f32>) -> vec2<f32> {
    if (node.mapping_kind == 1u) {
        return (node.mapping * vec4<f32>(material_texture_position, 1.0)).xy;
    }
    let p = (node.mapping * vec4<f32>(material_texture_position, 1.0)).xyz;
    if (node.mapping_kind == 2u) {
        let q = normalize(p);
        var phi = atan2(q.y, q.x) / (2.0 * 3.14159265359);
        if (phi < 0.0) { phi = phi + 1.0; }
        return vec2<f32>(acos(clamp(q.z, -1.0, 1.0)) / 3.14159265359, phi);
    }
    if (node.mapping_kind == 3u) {
        let s = (3.14159265359 + atan2(p.y, p.x)) / (2.0 * 3.14159265359);
        return vec2<f32>(s, p.z);
    }
    return (node.mapping * vec4<f32>(uv, 0.0, 1.0)).xy;
}

fn mapped_texture_position(node: TextureNodeRecord) -> vec3<f32> {
    return (node.mapping * vec4<f32>(material_texture_position, 1.0)).xyz;
}

fn sample_texture_leaf(texture_index: u32, uv: vec2<f32>) -> vec3<f32> {
    if (texture_index >= arrayLength(&texture_nodes)) { set_render_error(); return vec3<f32>(0.0); }
    let node = texture_nodes[texture_index];
    if (node.operation == TEXTURE_OPERATION_CONSTANT) { return node.constant_value.rgb; }
    let mapped_uv = mapped_texture_uv(node, uv);
    if ((node.swrap_mode == 2u && (mapped_uv.x < 0.0 || mapped_uv.x > 1.0))
        || (node.twrap_mode == 2u && (mapped_uv.y < 0.0 || mapped_uv.y > 1.0))) {
        return vec3<f32>(0.0);
    }
    // pbrt-v4 image textures use image-space origin at the upper left.
    let image_uv = vec2<f32>(mapped_uv.x, 1.0 - mapped_uv.y);
    var value = textureSampleLevel(
        texture_images[node.texture_index],
        texture_samplers[node.sampler],
        image_uv,
        0.0,
    ).rgb * node.constant_value.x;
    if (node.constant_value.y > 0.5) {
        value = max(vec3<f32>(0.0), vec3<f32>(1.0) - value);
    }
    return value;
}

fn sample_texture_program(root: TextureRootRecord, uv: vec2<f32>) -> vec3<f32> {
    var values: array<vec3<f32>, TEXTURE_PROGRAM_CAPACITY>;
    if (root.instruction_count > TEXTURE_PROGRAM_CAPACITY
        || root.result >= root.instruction_count) {
        set_render_error();
        return vec3<f32>(0.0);
    }
    var local = 0u;
    loop {
        if (local >= root.instruction_count) { break; }
        let node_index = root.texture_node + local;
        if (node_index >= arrayLength(&texture_nodes)) {
            set_render_error();
            return vec3<f32>(0.0);
        }
        let node = texture_nodes[node_index];
        if (node.operation == TEXTURE_OPERATION_IMAGE) {
            values[local] = sample_texture_leaf(node_index, uv);
        } else if (node.operation == TEXTURE_OPERATION_CONSTANT) {
            values[local] = node.constant_value.rgb;
        } else if (node.operation == TEXTURE_OPERATION_SCALE) {
            if ((node.child_count != 1u && node.child_count != 2u)
                || node.first_child >= arrayLength(&texture_child_indices)
                || (node.child_count == 2u
                    && node.first_child + 1u >= arrayLength(&texture_child_indices))) {
                set_render_error();
                return vec3<f32>(0.0);
            }
            let child = texture_child_indices[node.first_child];
            if (child >= local) {
                set_render_error();
                return vec3<f32>(0.0);
            }
            var factor = node.constant_value.x;
            if (node.child_count == 2u) {
                let scale = texture_child_indices[node.first_child + 1u];
                if (scale >= local) {
                    set_render_error();
                    return vec3<f32>(0.0);
                }
                factor = values[scale].x;
            }
            if (factor == 0.0) {
                values[local] = vec3<f32>(0.0);
            } else {
                values[local] = values[child] * factor;
            }
        } else if (node.operation == TEXTURE_OPERATION_MIX) {
            if (node.child_count < 2u
                || node.first_child + 1u >= arrayLength(&texture_child_indices)) {
                set_render_error();
                return vec3<f32>(0.0);
            }
            let first = texture_child_indices[node.first_child];
            let second = texture_child_indices[node.first_child + 1u];
            if (first >= local || second >= local) {
                set_render_error();
                return vec3<f32>(0.0);
            }
            var amount = node.constant_value.x;
            if (node.child_count >= 3u) {
                if (node.first_child + 2u >= arrayLength(&texture_child_indices)) {
                    set_render_error();
                    return vec3<f32>(0.0);
                }
                let amount_slot = texture_child_indices[node.first_child + 2u];
                if (amount_slot >= local) {
                    set_render_error();
                    return vec3<f32>(0.0);
                }
                amount = clamp(values[amount_slot].x, 0.0, 1.0);
            }
            values[local] = mix(values[first], values[second], amount);
        } else if (node.operation == TEXTURE_OPERATION_CHECKERBOARD
            || node.operation == TEXTURE_OPERATION_DIRECTION_MIX
            || node.operation == TEXTURE_OPERATION_BILERP) {
            if (node.first_child + node.child_count > arrayLength(&texture_child_indices)) {
                set_render_error();
                return vec3<f32>(0.0);
            }
            if (node.operation == TEXTURE_OPERATION_BILERP) {
                if (node.child_count != 4u) { set_render_error(); return vec3<f32>(0.0); }
                let v00 = texture_child_indices[node.first_child];
                let v01 = texture_child_indices[node.first_child + 1u];
                let v10 = texture_child_indices[node.first_child + 2u];
                let v11 = texture_child_indices[node.first_child + 3u];
                if (v00 >= local || v01 >= local || v10 >= local || v11 >= local) {
                    set_render_error(); return vec3<f32>(0.0);
                }
                let st = mapped_texture_uv(node, uv);
                values[local] = mix(mix(values[v00], values[v10], st.x),
                                    mix(values[v01], values[v11], st.x), st.y);
            } else {
                if (node.child_count != 2u) { set_render_error(); return vec3<f32>(0.0); }
                let first = texture_child_indices[node.first_child];
                let second = texture_child_indices[node.first_child + 1u];
                if (first >= local || second >= local) {
                    set_render_error(); return vec3<f32>(0.0);
                }
                if (node.operation == TEXTURE_OPERATION_DIRECTION_MIX) {
                    let amount = abs((node.mapping * vec4<f32>(material_texture_normal, 0.0)).x);
                    values[local] = mix(values[second], values[first], amount);
                } else {
                    var odd = false;
                    if (node.mapping_kind == 4u) {
                        let p = (node.mapping * vec4<f32>(material_texture_position, 1.0)).xyz;
                        odd = (i32(floor(p.x)) + i32(floor(p.y)) + i32(floor(p.z))) % 2 != 0;
                    } else {
                        let st = mapped_texture_uv(node, uv);
                        odd = (i32(floor(st.x)) + i32(floor(st.y))) % 2 != 0;
                    }
                    values[local] = select(values[first], values[second], odd);
                }
            }
        // TEXTURE_NOISE_BRANCH_BEGIN
        } else if (node.operation == TEXTURE_OPERATION_DOTS
            || node.operation == TEXTURE_OPERATION_FBM
            || node.operation == TEXTURE_OPERATION_WRINKLED
            || node.operation == TEXTURE_OPERATION_WINDY
            || node.operation == TEXTURE_OPERATION_MARBLE) {
            if (node.operation == TEXTURE_OPERATION_DOTS) {
                if (node.child_count != 2u
                    || node.first_child + 1u >= arrayLength(&texture_child_indices)) {
                    set_render_error(); return vec3<f32>(0.0);
                }
                let outside = texture_child_indices[node.first_child];
                let inside = texture_child_indices[node.first_child + 1u];
                if (outside >= local || inside >= local) {
                    set_render_error(); return vec3<f32>(0.0);
                }
                let st = mapped_texture_uv(node, uv);
                let cell = floor(st + vec2<f32>(0.5));
                var is_inside = false;
                if (texture_noise(vec3<f32>(cell + vec2<f32>(0.5), 0.0)) > 0.0) {
                    let center = cell + 0.15 * vec2<f32>(
                        texture_noise(vec3<f32>(cell.x + 1.5, cell.y + 2.8, 0.0)),
                        texture_noise(vec3<f32>(cell.x + 4.5, cell.y + 9.8, 0.0)),
                    );
                    let delta = st - center;
                    is_inside = dot(delta, delta) < 0.35 * 0.35;
                }
                values[local] = select(values[outside], values[inside], is_inside);
            } else {
                let p = mapped_texture_position(node);
                if (node.operation == TEXTURE_OPERATION_FBM) {
                    values[local] = vec3<f32>(texture_fbm(
                        p, node.constant_value.x, node.constant_value.y, false));
                } else if (node.operation == TEXTURE_OPERATION_WRINKLED) {
                    values[local] = vec3<f32>(texture_fbm(
                        p, node.constant_value.x, node.constant_value.y, true));
                } else if (node.operation == TEXTURE_OPERATION_WINDY) {
                    let wind = texture_fbm(0.1 * p, 0.5, 3.0, false);
                    let wave = texture_fbm(p, 0.5, 6.0, false);
                    values[local] = vec3<f32>(abs(wind) * wave);
                } else {
                    values[local] = texture_marble(
                        p,
                        node.constant_value.x,
                        node.constant_value.y,
                        node.constant_value.z,
                        node.constant_value.w,
                    );
                }
            }
        // TEXTURE_NOISE_BRANCH_END
        } else {
            set_render_error();
            return vec3<f32>(0.0);
        }
        local += 1u;
    }
    return values[root.result];
}

/* Legacy graph evaluator retained in source history; the post-order VM above
   is the only texture evaluation path. */
/*
fn texture_blend_amount(node: TextureNodeRecord, uv: vec2<f32>) -> f32 {
    if (node.operation == 3u && node.child_count >= 3u
        && node.first_child + 2u < arrayLength(&texture_child_indices)) {
        let amount_index = texture_child_indices[node.first_child + 2u];
        if (amount_index >= arrayLength(&texture_nodes)) {
            set_render_error();
            return 0.0;
        }
        let amount_node = texture_nodes[amount_index];
        if (amount_node.kind != 0u || amount_node.child_count != 0u
            || (amount_node.operation != 0u && amount_node.operation != 1u
                && amount_node.operation != 7u && amount_node.operation != 8u
                && amount_node.operation != 9u)) {
            set_render_error();
            return 0.0;
        }
        return clamp(sample_texture_leaf(amount_index, uv).x, 0.0, 1.0);
    }
    if (node.operation == 3u) { return clamp(node.constant_value.x, 0.0, 1.0); }
    return 1.0;
}

fn deferred_sample_texture_graph(texture_index: u32, uv: vec2<f32>) -> vec3<f32> {
    var stack_index: array<u32, 32>;
    var stack_state: array<u32, 32>;
    var stack_value: array<vec3<f32>, 32>;
    var stack_aux0: array<vec3<f32>, 32>;
    var depth = 0u;
    stack_index[0] = texture_index;
    stack_state[0] = 0u;
    for (var step = 0u; step < 128u; step++) {
        if (depth >= 32u || stack_index[depth] >= arrayLength(&texture_nodes)) {
            set_render_error(); return vec3<f32>(0.0);
        }
        let index = stack_index[depth];
        let node = texture_nodes[index];
        if (stack_state[depth] == 0u) {
            if (node.operation == 1u
                || node.operation >= 4u
                || (node.operation == 0u && node.child_count == 0u)) {
                stack_value[depth] = sample_texture_leaf(index, uv);
                stack_state[depth] = 3u;
            } else if (node.operation == 2u
                && (node.child_count == 1u || node.child_count == 2u)) {
                if (node.first_child >= arrayLength(&texture_child_indices)
                    || (node.child_count == 2u
                        && node.first_child + 1u >= arrayLength(&texture_child_indices))
                    || depth + 1u >= 32u) {
                    set_render_error(); return vec3<f32>(0.0);
                }
                stack_state[depth] = 1u;
                depth += 1u;
                stack_index[depth] = texture_child_indices[node.first_child];
                stack_state[depth] = 0u;
            } else if (node.operation == 3u && node.child_count >= 2u) {
                if (node.first_child + 1u >= arrayLength(&texture_child_indices) || depth + 1u >= 32u) {
                    set_render_error(); return vec3<f32>(0.0);
                }
                stack_state[depth] = 1u;
                depth += 1u;
                stack_index[depth] = texture_child_indices[node.first_child];
                stack_state[depth] = 0u;
            } else {
                set_render_error(); return vec3<f32>(0.0);
            }
        } else {
            if (depth == 0u) { return stack_value[0]; }
            let parent_depth = depth - 1u;
            let parent = texture_nodes[stack_index[parent_depth]];
            if (parent.operation == 2u && stack_state[parent_depth] == 1u) {
                if (parent.child_count >= 2u) {
                    stack_value[parent_depth] = stack_value[depth];
                    stack_state[parent_depth] = 2u;
                    depth += 1u;
                    stack_index[depth] = texture_child_indices[parent.first_child + 1u];
                    stack_state[depth] = 0u;
                } else {
                    if (parent.constant_value.x == 0.0) {
                        stack_value[parent_depth] = vec3<f32>(0.0);
                    } else {
                        stack_value[parent_depth] =
                            stack_value[depth] * parent.constant_value.x;
                    }
                    stack_state[parent_depth] = 3u;
                    depth = parent_depth;
                }
            } else if (parent.operation == 2u && stack_state[parent_depth] == 2u) {
                let factor = stack_value[depth].x;
                if (factor == 0.0) {
                    stack_value[parent_depth] = vec3<f32>(0.0);
                } else {
                    stack_value[parent_depth] = stack_value[parent_depth] * factor;
                }
                stack_state[parent_depth] = 3u;
                depth = parent_depth;
            } else if (parent.operation == 3u && stack_state[parent_depth] == 1u) {
                stack_value[parent_depth] = stack_value[depth];
                stack_state[parent_depth] = 2u;
                depth += 1u;
                stack_index[depth] = texture_child_indices[parent.first_child + 1u];
                stack_state[depth] = 0u;
            } else if (parent.operation == 3u && stack_state[parent_depth] == 2u) {
                if (parent.operation == 3u && parent.child_count >= 3u) {
                    if (parent.first_child + 2u >= arrayLength(&texture_child_indices)
                        || depth + 1u >= 32u) {
                        set_render_error(); return vec3<f32>(0.0);
                    }
                    stack_aux0[parent_depth] = stack_value[depth];
                    stack_state[parent_depth] = 3u;
                    depth += 1u;
                    stack_index[depth] = texture_child_indices[parent.first_child + 2u];
                    stack_state[depth] = 0u;
                } else {
                    stack_value[parent_depth] = mix(stack_value[parent_depth], stack_value[depth], texture_blend_amount(parent, uv));
                    stack_state[parent_depth] = 4u;
                    depth = parent_depth;
                }
            } else if (parent.operation == 3u && stack_state[parent_depth] == 3u) {
                stack_value[parent_depth] = mix(
                    stack_value[parent_depth],
                    stack_aux0[parent_depth],
                    clamp(stack_value[depth].x, 0.0, 1.0),
                );
                stack_state[parent_depth] = 4u;
                depth = parent_depth;
            } else {
                set_render_error(); return vec3<f32>(0.0);
            }
        }
    }
    set_render_error();
    return vec3<f32>(0.0);
}
fn sample_texture_rgb(texture_index: u32, uv: vec2<f32>) -> vec3<f32> {
    return deferred_sample_texture_graph(texture_index, uv);
}
*/
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
fn load_material_spectrum(material_node: u32, ordinal: u32, lambda: vec4<f32>) -> vec4<f32> {
    let attr_ref = load_material_attribute(material_node, ordinal);
    if (attr_ref.kind == 2u) {
        let result = load_texture_eval_result(material_node, ordinal, attr_ref.index);
        if (result.valid == 0u || result.texture_root >= arrayLength(&texture_roots)) {
            return vec4<f32>(0.0);
        }
        let root = texture_roots[result.texture_root];
        let rgb = result.value.rgb;
        if (texture_nodes[root.texture_node + root.result].kind == 0u) {
            return vec4<f32>(rgb.x);
        }
        if (root.spectrum_type == 1u) {
            return rgb_to_unbounded_spectrum4(rgb, lambda, texture_nodes[root.texture_node + root.result].color_space);
        }
        return rgb_to_spectrum4(rgb, lambda, texture_nodes[root.texture_node + root.result].color_space);
    }
    if (attr_ref.kind != 1u) { set_render_error(); return vec4<f32>(0.0); }
    return evaluate_spectrum(attr_ref.index, lambda);
}
fn load_diffuse_reflectance(material_node: u32, lambda: vec4<f32>) -> vec4<f32> {
    if (material_table.debug_material_kind == MATERIAL_KIND_LAMBERT) { return vec4<f32>(0.5); }
    return load_material_spectrum(material_node, 0u, lambda);
}
fn load_diffuse_transmission_reflectance(material_node: u32, lambda: vec4<f32>) -> vec4<f32> {
    return load_material_spectrum(material_node, 0u, lambda);
}
fn load_diffuse_transmission_transmittance(material_node: u32, lambda: vec4<f32>) -> vec4<f32> {
    return load_material_spectrum(material_node, 1u, lambda);
}
fn load_diffuse_transmission_scale(material_node: u32) -> f32 {
    return load_material_scalar(material_node, 2u);
}
fn load_attributes_eval_work_item(root: u32) -> AttributesEvalWorkItem {
    if (root >= arrayLength(&attributes_eval_work_items)) {
        set_render_error();
        return attributes_eval_work_items[0u];
    }
    return attributes_eval_work_items[root];
}
fn resolve_attributes_eval_work_item(root: u32) -> AttributesEvalWorkItem {
    var current = root;
    for (var depth = 0u; depth < material_table.attributes_eval_stride; depth++) {
        let item = load_attributes_eval_work_item(current);
        if (item.bxdf_kind != MATERIAL_KIND_MIX) { return item; }
        if (item.selected_child_work_item == 0xffffffffu) { set_render_error(); return item; }
        current = item.selected_child_work_item;
    }
    set_render_error();
    return load_attributes_eval_work_item(root);
}
fn next_material_node(material_root: MaterialRoot, current: u32) -> u32 {
    if (current < material_root.node_offset || current >= material_root.node_offset + material_root.node_count) {
        set_render_error();
        return 0xffffffffu;
    }
    var node = material_nodes[current];
    if (node.child0 != 0xffffffffu) { return node.child0; }
    if (node.child1 != 0xffffffffu) { return node.child1; }
    var child = current;
    for (var climbed = 0u; climbed < material_root.node_count; climbed++) {
        if (node.parent == 0xffffffffu) { return 0xffffffffu; }
        let parent = material_nodes[node.parent];
        if (parent.child0 == child && parent.child1 != 0xffffffffu) {
            return parent.child1;
        }
        child = node.parent;
        node = parent;
    }
    set_render_error();
    return 0xffffffffu;
}
fn load_layered_params(root: AttributesEvalWorkItem, kind: u32) -> LayeredParams {
    var params: LayeredParams;
    params.thickness = root.values[0].x;
    params.g = root.values[2].x;
    params.max_depth = root.values[3].x;
    params.n_samples = root.values[4].x;
    params.albedo = root.values[1];
    if (kind == MATERIAL_KIND_COATED_DIFFUSE) { params.albedo = root.values[5]; }
    return params;
}
fn dielectric_interface_alpha(item: AttributesEvalWorkItem) -> vec2<f32> {
    var roughness = max(vec2<f32>(item.values[1].x, item.values[2].x), vec2<f32>(0.0));
    if (item.values[3].x != 0.0) {
        roughness = sqrt(roughness);
    }
    return roughness;
}
fn invalid_dielectric_interface_sample() -> DielectricInterfaceSample {
    return DielectricInterfaceSample(vec4<f32>(0.0), vec3<f32>(0.0), 0.0, 1.0, 0u, 0u, 0u);
}
fn refract_interface(wo: vec3<f32>, normal_input: vec3<f32>, eta_input: f32) -> DielectricInterfaceSample {
    var cosine_i = dot(normal_input, wo);
    var eta = eta_input;
    var normal = normal_input;
    if (cosine_i < 0.0) {
        eta = 1.0 / eta;
        cosine_i = -cosine_i;
        normal = -normal;
    }
    let sin2_t = max(0.0, 1.0 - cosine_i * cosine_i) / (eta * eta);
    if (sin2_t >= 1.0) { return invalid_dielectric_interface_sample(); }
    let cosine_t = sqrt(max(0.0, 1.0 - sin2_t));
    var result = invalid_dielectric_interface_sample();
    result.wi = normalize(-wo / eta + (cosine_i / eta - cosine_t) * normal);
    result.etap = eta;
    result.valid = 1u;
    result.transmission = 1u;
    return result;
}
fn sample_smooth_dielectric_interface(
    wo: vec3<f32>, eta: f32, uc: f32, allow_reflection: bool, allow_transmission: bool,
) -> DielectricInterfaceSample {
    let fresnel = dielectric_fresnel(wo.z, eta);
    let pr = select(0.0, fresnel, allow_reflection);
    let pt = select(0.0, 1.0 - fresnel, allow_transmission);
    if (pr + pt == 0.0) { return invalid_dielectric_interface_sample(); }
    if (uc < pr / (pr + pt)) {
        let wi = vec3<f32>(-wo.x, -wo.y, wo.z);
        return DielectricInterfaceSample(
            vec4<f32>(fresnel / abs(wi.z)), wi, pr / (pr + pt), 1.0, 1u, 0u, 1u,
        );
    }
    var result = refract_interface(wo, vec3<f32>(0.0, 0.0, 1.0), eta);
    if (result.valid == 0u) { return result; }
    var ft = (1.0 - fresnel) / abs(result.wi.z);
    // WavefrontPathIntegrator transports radiance.
    ft = ft / (result.etap * result.etap);
    result.f = vec4<f32>(ft);
    result.pdf = pt / (pr + pt);
    result.specular = 1u;
    return result;
}
fn tr_distribution_d(wm: vec3<f32>, alpha: vec2<f32>) -> f32 {
    let cos2 = wm.z * wm.z;
    if (cos2 < 1e-16) { return 0.0; }
    let e = (wm.x * wm.x / (alpha.x * alpha.x)
        + wm.y * wm.y / (alpha.y * alpha.y)) / cos2;
    return 1.0 / (PI * alpha.x * alpha.y * cos2 * cos2 * (1.0 + e) * (1.0 + e));
}
fn tr_distribution_lambda(w: vec3<f32>, alpha: vec2<f32>) -> f32 {
    let wz2 = w.z * w.z;
    if (wz2 == 0.0) { return 0.0; }
    let alpha2_tan2 = (alpha.x * w.x) * (alpha.x * w.x)
        + (alpha.y * w.y) * (alpha.y * w.y);
    return 0.5 * (sqrt(1.0 + alpha2_tan2 / wz2) - 1.0);
}
fn tr_distribution_g1(w: vec3<f32>, alpha: vec2<f32>) -> f32 {
    return 1.0 / (1.0 + tr_distribution_lambda(w, alpha));
}
fn tr_distribution_g(wo: vec3<f32>, wi: vec3<f32>, alpha: vec2<f32>) -> f32 {
    return 1.0 / (1.0 + tr_distribution_lambda(wo, alpha) + tr_distribution_lambda(wi, alpha));
}
fn sample_visible_tr_wm(wo_input: vec3<f32>, alpha: vec2<f32>, u: vec2<f32>) -> vec3<f32> {
    var wh = normalize(vec3<f32>(alpha.x * wo_input.x, alpha.y * wo_input.y, wo_input.z));
    if (wh.z < 0.0) { wh = -wh; }
    var t1 = vec3<f32>(1.0, 0.0, 0.0);
    if (wh.z < 0.99999) { t1 = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), wh)); }
    let t2 = cross(wh, t1);
    let radius = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    var p = vec2<f32>(radius * cos(phi), radius * sin(phi));
    let h = sqrt(max(0.0, 1.0 - p.x * p.x));
    p.y = mix(h, p.y, (1.0 + wh.z) * 0.5);
    let pz = sqrt(max(0.0, 1.0 - dot(p, p)));
    let nh = p.x * t1 + p.y * t2 + pz * wh;
    return normalize(vec3<f32>(alpha.x * nh.x, alpha.y * nh.y, max(1e-6, nh.z)));
}
fn tr_visible_wm_pdf(wo: vec3<f32>, wm: vec3<f32>, alpha: vec2<f32>) -> f32 {
    if (abs(wo.z) == 0.0) { return 0.0; }
    return tr_distribution_d(wm, alpha) * tr_distribution_g1(wo, alpha)
        * abs(dot(wo, wm)) / abs(wo.z);
}
fn sample_rough_dielectric_interface(
    wo: vec3<f32>, eta: f32, alpha_input: vec2<f32>, uc: f32, u: vec2<f32>,
    allow_reflection: bool, allow_transmission: bool,
) -> DielectricInterfaceSample {
    let alpha = max(alpha_input, vec2<f32>(1e-4));
    let wm = sample_visible_tr_wm(wo, alpha, u);
    let fresnel = dielectric_fresnel(dot(wo, wm), eta);
    let pr = select(0.0, fresnel, allow_reflection);
    let pt = select(0.0, 1.0 - fresnel, allow_transmission);
    if (pr + pt == 0.0) { return invalid_dielectric_interface_sample(); }
    let wm_pdf = tr_visible_wm_pdf(wo, wm, alpha);
    if (uc < pr / (pr + pt)) {
        let wi = normalize(-wo + 2.0 * dot(wo, wm) * wm);
        if (wo.z * wi.z <= 0.0) { return invalid_dielectric_interface_sample(); }
        let pdf = wm_pdf / max(4.0 * abs(dot(wo, wm)), 1e-7) * pr / (pr + pt);
        let value = tr_distribution_d(wm, alpha) * tr_distribution_g(wo, wi, alpha)
            * fresnel / max(abs(4.0 * wi.z * wo.z), 1e-7);
        return DielectricInterfaceSample(vec4<f32>(value), wi, pdf, 1.0, 1u, 0u, 0u);
    }
    var result = refract_interface(wo, wm, eta);
    if (result.valid == 0u || wo.z * result.wi.z >= 0.0 || result.wi.z == 0.0) { return invalid_dielectric_interface_sample(); }
    let denominator = dot(result.wi, wm) + dot(wo, wm) / result.etap;
    let denominator2 = denominator * denominator;
    if (denominator2 == 0.0) { return invalid_dielectric_interface_sample(); }
    let dwm_dwi = abs(dot(result.wi, wm)) / denominator2;
    result.pdf = wm_pdf * dwm_dwi * pt / (pr + pt);
    var ft = (1.0 - fresnel) * tr_distribution_d(wm, alpha)
        * tr_distribution_g(wo, result.wi, alpha)
        * abs(dot(result.wi, wm) * dot(wo, wm)
            / max(abs(result.wi.z * wo.z) * denominator2, 1e-7));
    ft = ft / (result.etap * result.etap);
    result.f = vec4<f32>(ft);
    result.specular = 0u;
    return result;
}
fn sample_dielectric_interface(
    item: AttributesEvalWorkItem, wo: vec3<f32>, uc: f32, u: vec2<f32>,
    allow_reflection: bool, allow_transmission: bool,
) -> DielectricInterfaceSample {
    let eta = select(item.values[0].x, 1.0, item.values[0].x == 0.0);
    let alpha = dielectric_interface_alpha(item);
    if (eta == 1.0 || max(alpha.x, alpha.y) < 1e-3) {
        return sample_smooth_dielectric_interface(
            wo, eta, uc, allow_reflection, allow_transmission,
        );
    }
    return sample_rough_dielectric_interface(
        wo, eta, alpha, uc, u, allow_reflection, allow_transmission,
    );
}
fn conductor_interface_alpha(item: AttributesEvalWorkItem) -> vec2<f32> {
    var roughness = max(vec2<f32>(item.values[2].x, item.values[3].x), vec2<f32>(0.0));
    if (item.values[4].x != 0.0) { roughness = sqrt(roughness); }
    return roughness;
}
fn sample_conductor_interface(
    item: AttributesEvalWorkItem, wo: vec3<f32>, u: vec2<f32>,
) -> DielectricInterfaceSample {
    let alpha = conductor_interface_alpha(item);
    if (max(alpha.x, alpha.y) < 1e-3) {
        let wi = vec3<f32>(-wo.x, -wo.y, wo.z);
        let value = conductor_fresnel(abs(wo.z), item.values[0], item.values[1]) / abs(wi.z);
        return DielectricInterfaceSample(value, wi, 1.0, 1.0, 1u, 0u, 1u);
    }
    let bounded_alpha = max(alpha, vec2<f32>(1e-4));
    let wm = sample_visible_tr_wm(wo, bounded_alpha, u);
    let wi = normalize(-wo + 2.0 * dot(wo, wm) * wm);
    if (wo.z * wi.z <= 0.0) { return invalid_dielectric_interface_sample(); }
    let pdf = tr_visible_wm_pdf(wo, wm, bounded_alpha)
        / max(4.0 * abs(dot(wo, wm)), 1e-7);
    let value = conductor_fresnel(abs(dot(wo, wm)), item.values[0], item.values[1])
        * tr_distribution_d(wm, bounded_alpha)
        * tr_distribution_g(wo, wi, bounded_alpha)
        / max(abs(4.0 * wo.z * wi.z), 1e-7);
    return DielectricInterfaceSample(value, wi, pdf, 1.0, 1u, 0u, 0u);
}
fn dielectric_interface_f(
    item: AttributesEvalWorkItem, wo: vec3<f32>, wi: vec3<f32>,
) -> vec4<f32> {
    let eta = select(item.values[0].x, 1.0, item.values[0].x == 0.0);
    let alpha = dielectric_interface_alpha(item);
    if (eta == 1.0 || max(alpha.x, alpha.y) < 1e-3 || wo.z == 0.0 || wi.z == 0.0) {
        return vec4<f32>(0.0);
    }
    let reflection = wo.z * wi.z > 0.0;
    var etap = 1.0;
    if (!reflection) { etap = select(1.0 / eta, eta, wo.z > 0.0); }
    var wm = wi * etap + wo;
    if (dot(wm, wm) == 0.0) { return vec4<f32>(0.0); }
    wm = normalize(wm);
    if (wm.z < 0.0) { wm = -wm; }
    if (dot(wm, wi) * wi.z < 0.0 || dot(wm, wo) * wo.z < 0.0) {
        return vec4<f32>(0.0);
    }
    let bounded_alpha = max(alpha, vec2<f32>(1e-4));
    let fresnel = dielectric_fresnel(dot(wo, wm), eta);
    if (reflection) {
        let value = tr_distribution_d(wm, bounded_alpha)
            * tr_distribution_g(wo, wi, bounded_alpha) * fresnel
            / max(abs(4.0 * wi.z * wo.z), 1e-7);
        return vec4<f32>(value);
    }
    let denominator = dot(wi, wm) + dot(wo, wm) / etap;
    var value = tr_distribution_d(wm, bounded_alpha) * (1.0 - fresnel)
        * tr_distribution_g(wo, wi, bounded_alpha)
        * abs(dot(wi, wm) * dot(wo, wm)
            / max(abs(wi.z * wo.z) * denominator * denominator, 1e-7));
    value /= etap * etap;
    return vec4<f32>(value);
}
fn dielectric_interface_pdf(
    item: AttributesEvalWorkItem, wo: vec3<f32>, wi: vec3<f32>,
    allow_reflection: bool, allow_transmission: bool,
) -> f32 {
    let eta = select(item.values[0].x, 1.0, item.values[0].x == 0.0);
    let alpha = dielectric_interface_alpha(item);
    if (eta == 1.0 || max(alpha.x, alpha.y) < 1e-3 || wo.z == 0.0 || wi.z == 0.0) {
        return 0.0;
    }
    let reflection = wo.z * wi.z > 0.0;
    var etap = 1.0;
    if (!reflection) { etap = select(1.0 / eta, eta, wo.z > 0.0); }
    var wm = wi * etap + wo;
    if (dot(wm, wm) == 0.0) { return 0.0; }
    wm = normalize(wm);
    if (wm.z < 0.0) { wm = -wm; }
    let fresnel = dielectric_fresnel(dot(wo, wm), eta);
    let pr = select(0.0, fresnel, allow_reflection);
    let pt = select(0.0, 1.0 - fresnel, allow_transmission);
    if (pr + pt == 0.0) { return 0.0; }
    let wm_pdf = tr_visible_wm_pdf(wo, wm, max(alpha, vec2<f32>(1e-4)));
    if (reflection) {
        return wm_pdf / max(4.0 * abs(dot(wo, wm)), 1e-7) * pr / (pr + pt);
    }
    let denominator = dot(wi, wm) + dot(wo, wm) / etap;
    return wm_pdf * abs(dot(wi, wm)) / max(denominator * denominator, 1e-7)
        * pt / (pr + pt);
}
fn conductor_interface_f(
    item: AttributesEvalWorkItem, wo: vec3<f32>, wi: vec3<f32>,
) -> vec4<f32> {
    let alpha = conductor_interface_alpha(item);
    if (max(alpha.x, alpha.y) < 1e-3 || wo.z * wi.z <= 0.0) { return vec4<f32>(0.0); }
    var wm = normalize(wo + wi);
    if (wm.z < 0.0) { wm = -wm; }
    let bounded_alpha = max(alpha, vec2<f32>(1e-4));
    return conductor_fresnel(abs(dot(wo, wm)), item.values[0], item.values[1])
        * tr_distribution_d(wm, bounded_alpha) * tr_distribution_g(wo, wi, bounded_alpha)
        / max(abs(4.0 * wo.z * wi.z), 1e-7);
}
fn conductor_interface_pdf(
    item: AttributesEvalWorkItem, wo: vec3<f32>, wi: vec3<f32>,
) -> f32 {
    let alpha = conductor_interface_alpha(item);
    if (max(alpha.x, alpha.y) < 1e-3 || wo.z * wi.z <= 0.0) { return 0.0; }
    var wm = normalize(wo + wi);
    if (wm.z < 0.0) { wm = -wm; }
    return tr_visible_wm_pdf(wo, wm, max(alpha, vec2<f32>(1e-4)))
        / max(4.0 * abs(dot(wo, wm)), 1e-7);
}
fn sample_dielectric_interface_importance(
    item: AttributesEvalWorkItem, wo: vec3<f32>, uc: f32, u: vec2<f32>,
    allow_reflection: bool, allow_transmission: bool,
) -> DielectricInterfaceSample {
    var sample = sample_dielectric_interface(
        item, wo, uc, u, allow_reflection, allow_transmission,
    );
    if (sample.valid != 0u && sample.transmission != 0u) {
        sample.f *= sample.etap * sample.etap;
    }
    return sample;
}
fn layered_bottom_f(
    kind: u32, bottom: AttributesEvalWorkItem, wo: vec3<f32>, wi: vec3<f32>,
) -> vec4<f32> {
    if (kind == MATERIAL_KIND_COATED_DIFFUSE) {
        return select(vec4<f32>(0.0), bottom.values[0] / PI, wo.z * wi.z > 0.0);
    }
    return conductor_interface_f(bottom, wo, wi);
}
fn layered_bottom_pdf(
    kind: u32, bottom: AttributesEvalWorkItem, wo: vec3<f32>, wi: vec3<f32>,
) -> f32 {
    if (kind == MATERIAL_KIND_COATED_DIFFUSE) {
        return select(0.0, abs(wi.z) / PI, wo.z * wi.z > 0.0);
    }
    return conductor_interface_pdf(bottom, wo, wi);
}
fn sample_layered_bottom(
    kind: u32, bottom: AttributesEvalWorkItem, wo: vec3<f32>, u: vec2<f32>,
) -> DielectricInterfaceSample {
    if (kind == MATERIAL_KIND_COATED_CONDUCTOR) {
        return sample_conductor_interface(bottom, wo, u);
    }
    let radius = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    var wi = vec3<f32>(radius * cos(phi), radius * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
    if (wo.z < 0.0) { wi.z = -wi.z; }
    let pdf = abs(wi.z) / PI;
    if (pdf == 0.0) { return invalid_dielectric_interface_sample(); }
    return DielectricInterfaceSample(bottom.values[0] / PI, wi, pdf, 1.0, 1u, 0u, 0u);
}
fn power_heuristic_one(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    return a2 / max(a2 + pdf_b * pdf_b, 1e-30);
}
fn layered_tr(distance: f32, w: vec3<f32>) -> f32 {
    if (abs(distance) <= 1.17549435e-38) { return 1.0; }
    return exp(-abs(distance / w.z));
}
fn evaluate_layered_f(
    root: AttributesEvalWorkItem, kind: u32, wo_input: vec3<f32>, wi_input: vec3<f32>,
    pixel_index: u32, path_depth: u32,
) -> vec4<f32> {
    if (wo_input.z == 0.0 || wi_input.z == 0.0 || wo_input.z * wi_input.z <= 0.0) {
        return vec4<f32>(0.0);
    }
    var wo = wo_input;
    var wi = wi_input;
    if (wo.z < 0.0) { wo = -wo; wi = -wi; }
    let top = load_attributes_eval_work_item(root.child_work_item0);
    let bottom = load_attributes_eval_work_item(root.child_work_item1);
    let params = load_layered_params(root, kind);
    let n_samples = max(1u, u32(params.n_samples));
    let max_depth = max(1u, u32(params.max_depth));
    var result = f32(n_samples) * dielectric_interface_f(top, wo, wi);
    let top_rough = max(dielectric_interface_alpha(top).x, dielectric_interface_alpha(top).y) >= 1e-3;
    let bottom_rough = kind == MATERIAL_KIND_COATED_DIFFUSE
        || max(conductor_interface_alpha(bottom).x, conductor_interface_alpha(bottom).y) >= 1e-3;
    for (var sample_index = 0u; sample_index < n_samples; sample_index++) {
        let sample_base = 64u + sample_index * 256u;
        let wos = sample_dielectric_interface(
            top, wo,
            random01(pixel_index, sample_base, path_depth),
            vec2<f32>(random01(pixel_index, sample_base + 1u, path_depth),
                random01(pixel_index, sample_base + 2u, path_depth)),
            false, true,
        );
        if (wos.valid == 0u || wos.pdf == 0.0 || wos.wi.z == 0.0) { continue; }
        let wis = sample_dielectric_interface_importance(
            top, wi,
            random01(pixel_index, sample_base + 3u, path_depth),
            vec2<f32>(random01(pixel_index, sample_base + 4u, path_depth),
                random01(pixel_index, sample_base + 5u, path_depth)),
            false, true,
        );
        if (wis.valid == 0u || wis.pdf == 0.0 || wis.wi.z == 0.0) { continue; }
        var beta = wos.f * abs(wos.wi.z) / wos.pdf;
        var z = max(params.thickness, 1.17549435e-38);
        var w = wos.wi;
        for (var layer_depth = 0u; layer_depth < max_depth; layer_depth++) {
            let random_base = sample_base + 8u + layer_depth * 6u;
            if (layer_depth > 3u && max_spectrum(beta) < 0.25) {
                let q = max(0.0, 1.0 - max_spectrum(beta));
                if (random01(pixel_index, random_base, path_depth) < q) { break; }
                beta /= max(1.0 - q, 1e-7);
            }
            if (w.z == 0.0) { break; }
            if (max_spectrum(params.albedo) > 0.0) {
                let dz = sample_layered_exponential(
                    random01(pixel_index, random_base + 1u, path_depth), 1.0 / abs(w.z),
                );
                let zp = select(z - dz, z + dz, w.z > 0.0);
                if (zp == z) { continue; }
                if (zp > 0.0 && zp < params.thickness) {
                    let phase_to_wi = hg_phase(dot(normalize(-w), normalize(-wis.wi)), params.g);
                    var wt = 1.0;
                    if (top_rough) { wt = power_heuristic_one(wis.pdf, phase_to_wi); }
                    result += beta * params.albedo * phase_to_wi * wt
                        * layered_tr(zp - params.thickness, wis.wi) * wis.f / wis.pdf;
                    let phase_wi = sample_hg_direction(
                        normalize(-w),
                        vec2<f32>(random01(pixel_index, random_base + 2u, path_depth),
                            random01(pixel_index, random_base + 3u, path_depth)), params.g,
                    );
                    let phase_pdf = hg_phase(dot(normalize(-w), phase_wi), params.g);
                    if (phase_pdf == 0.0 || phase_wi.z == 0.0) { continue; }
                    beta *= params.albedo;
                    w = phase_wi;
                    z = zp;
                    if (w.z > 0.0 && top_rough) {
                        let f_exit = dielectric_interface_f(top, -w, wi);
                        let exit_pdf = dielectric_interface_pdf(top, -w, wi, false, true);
                        result += beta * layered_tr(zp - params.thickness, w) * f_exit
                            * power_heuristic_one(phase_pdf, exit_pdf);
                    }
                    continue;
                }
                z = clamp(zp, 0.0, params.thickness);
            } else {
                z = select(params.thickness, 0.0, z == params.thickness);
                beta *= layered_tr(params.thickness, w);
            }
            if (z == params.thickness) {
                let reflected = sample_dielectric_interface(
                    top, -w, random01(pixel_index, random_base + 1u, path_depth),
                    vec2<f32>(random01(pixel_index, random_base + 2u, path_depth),
                        random01(pixel_index, random_base + 3u, path_depth)), true, false,
                );
                if (reflected.valid == 0u || reflected.pdf == 0.0 || reflected.wi.z == 0.0) { break; }
                beta *= reflected.f * abs(reflected.wi.z) / reflected.pdf;
                w = reflected.wi;
            } else {
                if (bottom_rough) {
                    var wt = 1.0;
                    if (top_rough) {
                        wt = power_heuristic_one(wis.pdf,
                            layered_bottom_pdf(kind, bottom, -w, -wis.wi));
                    }
                    result += beta * layered_bottom_f(kind, bottom, -w, -wis.wi)
                        * abs(wis.wi.z) * wt * layered_tr(params.thickness, wis.wi)
                        * wis.f / wis.pdf;
                }
                let reflected = sample_layered_bottom(
                    kind, bottom, -w,
                    vec2<f32>(random01(pixel_index, random_base + 2u, path_depth),
                        random01(pixel_index, random_base + 3u, path_depth)),
                );
                if (reflected.valid == 0u || reflected.pdf == 0.0 || reflected.wi.z == 0.0) { break; }
                beta *= reflected.f * abs(reflected.wi.z) / reflected.pdf;
                w = reflected.wi;
                if (top_rough) {
                    let f_exit = dielectric_interface_f(top, -w, wi);
                    let exit_pdf = dielectric_interface_pdf(top, -w, wi, false, true);
                    result += beta * layered_tr(params.thickness, w) * f_exit
                        * power_heuristic_one(reflected.pdf, exit_pdf);
                }
            }
        }
    }
    return result / f32(n_samples);
}
fn evaluate_layered_pdf(
    root: AttributesEvalWorkItem, kind: u32, wo_input: vec3<f32>, wi_input: vec3<f32>,
    pixel_index: u32, path_depth: u32,
) -> f32 {
    if (wo_input.z == 0.0 || wi_input.z == 0.0 || wo_input.z * wi_input.z <= 0.0) {
        return 0.0;
    }
    var wo = wo_input;
    var wi = wi_input;
    if (wo.z < 0.0) { wo = -wo; wi = -wi; }
    let top = load_attributes_eval_work_item(root.child_work_item0);
    let bottom = load_attributes_eval_work_item(root.child_work_item1);
    let n_samples = max(1u, u32(load_layered_params(root, kind).n_samples));
    let top_rough = max(dielectric_interface_alpha(top).x, dielectric_interface_alpha(top).y) >= 1e-3;
    let bottom_rough = kind == MATERIAL_KIND_COATED_DIFFUSE
        || max(conductor_interface_alpha(bottom).x, conductor_interface_alpha(bottom).y) >= 1e-3;
    var pdf_sum = f32(n_samples) * dielectric_interface_pdf(top, wo, wi, true, false);
    for (var sample_index = 0u; sample_index < n_samples; sample_index++) {
        let random_base = 20000u + sample_index * 8u;
        let wos = sample_dielectric_interface(
            top, wo,
            random01(pixel_index, random_base, path_depth),
            vec2<f32>(random01(pixel_index, random_base + 1u, path_depth),
                random01(pixel_index, random_base + 2u, path_depth)),
            false, true,
        );
        if (wos.valid == 0u || wos.pdf == 0.0 || wos.wi.z == 0.0) { continue; }
        let wis = sample_dielectric_interface_importance(
            top, wi,
            random01(pixel_index, random_base + 3u, path_depth),
            vec2<f32>(random01(pixel_index, random_base + 4u, path_depth),
                random01(pixel_index, random_base + 5u, path_depth)),
            false, true,
        );
        if (wis.valid == 0u || wis.pdf == 0.0 || wis.wi.z == 0.0) { continue; }
        let reflection_pdf = layered_bottom_pdf(kind, bottom, -wos.wi, -wis.wi);
        if (!top_rough) {
            pdf_sum += reflection_pdf;
            continue;
        }
        let reflected = sample_layered_bottom(
            kind, bottom, -wos.wi,
            vec2<f32>(random01(pixel_index, random_base + 6u, path_depth),
                random01(pixel_index, random_base + 7u, path_depth)),
        );
        if (reflected.valid == 0u || reflected.pdf == 0.0 || reflected.wi.z == 0.0) {
            continue;
        }
        let transmission_pdf = dielectric_interface_pdf(
            top, -reflected.wi, wi, false, true,
        );
        if (!bottom_rough) {
            pdf_sum += transmission_pdf;
        } else {
            pdf_sum += power_heuristic_one(wis.pdf, reflection_pdf) * reflection_pdf;
            pdf_sum += power_heuristic_one(reflected.pdf, transmission_pdf) * transmission_pdf;
        }
    }
    return mix(1.0 / (4.0 * PI), pdf_sum / f32(n_samples), 0.9);
}
fn load_dielectric_eta(material_node: u32, lambda: vec4<f32>) -> vec4<f32> { return load_material_spectrum(material_node, 0u, lambda); }
fn dielectric_eta_is_constant(material_node: u32) -> bool { return spectrum_is_constant(load_material_attribute(material_node, 0u).index); }
fn load_conductor_eta(material_node: u32, lambda: vec4<f32>) -> vec4<f32> { return load_material_spectrum(material_node, 0u, lambda); }
fn load_conductor_k(material_node: u32, lambda: vec4<f32>) -> vec4<f32> { return load_material_spectrum(material_node, 1u, lambda); }
fn load_conductor_roughness(material_node: u32) -> f32 { return load_material_scalar(material_node, 2u); }
fn load_conductor_reflectance_roughness(material_node: u32) -> f32 { return load_material_scalar(material_node, 1u); }
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
fn dielectric_fresnel(cosine: f32, eta: f32) -> f32 {
    let c = clamp(abs(cosine), 0.0, 1.0);
    let e = select(eta, 1.0 / eta, cosine < 0.0);
    let sin2_t = max(0.0, 1.0 - c * c) / (e * e);
    if (sin2_t >= 1.0) { return 1.0; }
    let ct = sqrt(1.0 - sin2_t);
    let rp = (e * c - ct) / (e * c + ct);
    let rs = (c - e * ct) / (c + e * ct);
    return (rp * rp + rs * rs) * 0.5;
}

fn scattering_local(w: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let t = make_tangent(n);
    return vec3<f32>(dot(w, t), dot(w, cross(n, t)), dot(w, n));
}

fn scattering_local_frame(w: vec3<f32>, tangent: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(w, tangent), dot(w, cross(normal, tangent)), dot(w, normal));
}

fn scattering_world_frame(w: vec3<f32>, tangent: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return tangent * w.x + cross(normal, tangent) * w.y + normal * w.z;
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

fn uniform_light_pmf_for_handle(light_handle: u32) -> f32 {
    if (light_handle >= light_table.light_count) {
        return 0.0;
    }
    return 1.0 / f32(light_table.light_count);
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

// pbrt-v4 HGPhaseFunction helpers used by the layered medium random walk.
fn hg_phase(cosine: f32, g: f32) -> f32 {
    let bounded_g = clamp(g, -0.99, 0.99);
    let gg = bounded_g * bounded_g;
    let denominator = max(1.0 + gg + 2.0 * bounded_g * cosine, 1e-7);
    return (1.0 - gg) / (4.0 * PI * denominator * sqrt(denominator));
}
fn sample_hg_cosine(u: f32, g: f32) -> f32 {
    let bounded_g = clamp(g, -0.99, 0.99);
    if (abs(bounded_g) < 1e-3) {
        return 1.0 - 2.0 * u;
    }
    let t = (1.0 - bounded_g * bounded_g)
        / max(1.0 + bounded_g - 2.0 * bounded_g * u, 1e-7);
    return clamp(
        -(1.0 + bounded_g * bounded_g - t * t) / (2.0 * bounded_g),
        -1.0,
        1.0,
    );
}
fn sample_hg_direction(reference: vec3<f32>, u: vec2<f32>, g: f32) -> vec3<f32> {
    let cosine = sample_hg_cosine(u.x, g);
    let sine = sqrt(max(0.0, 1.0 - cosine * cosine));
    let phi = 2.0 * PI * u.y;
    let tangent = make_tangent(reference);
    let bitangent = cross(reference, tangent);
    return normalize(
        tangent * (sine * cos(phi))
        + bitangent * (sine * sin(phi))
        + reference * cosine,
    );
}
fn sample_layered_exponential(u: f32, rate: f32) -> f32 {
    return -log(max(1.0 - min(u, 0.99999994), 1e-7)) / max(rate, 1e-7);
}

fn generate_ray_samples(pixel_index: u32, depth: u32) -> RaySamples {
    let first_dimension = 6u + 7u * depth;
    return RaySamples(
        vec4<f32>(
            sampler_get_1d(pixel_index, first_dimension),
            sampler_get_2d(pixel_index, first_dimension + 1u),
            0.0,
        ),
        vec4<f32>(
            sampler_get_1d(pixel_index, first_dimension + 3u),
            sampler_get_2d(pixel_index, first_dimension + 4u),
            sampler_get_1d(pixel_index, first_dimension + 6u),
        ),
    );
}

fn sample_uniform_light(selector: f32) -> LightSelection {
    if (light_table.light_count == 0u) {
        return LightSelection(0xffffffffu, 0.0);
    }
    let selected = min(u32(min(selector, 0.99999994) * f32(light_table.light_count)), light_table.light_count - 1u);
    return LightSelection(selected, 1.0 / f32(light_table.light_count));
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
        let finite_group_count = select(0u, 1u, light_table.light_bvh_node_count > 0u);
        let group_count = light_table.infinite_light_count + finite_group_count;
        if (group_count == 0u) {
            return 0.0;
        }
        if (light_handle >= light_table.finite_light_count) {
            if (light_handle >= light_table.light_count) {
                return 0.0;
            }
            return 1.0 / f32(group_count);
        }
        return light_bvh_pmf_for_handle(light_handle, p, n)
            * f32(finite_group_count) / f32(group_count);
    }
    return uniform_light_pmf_for_handle(light_handle);
}

fn sample_scene_light(selector: f32, p: vec3<f32>, n: vec3<f32>) -> LightSelection {
    if (light_table.light_sampler_kind == LIGHT_SAMPLER_KIND_BVH) {
        let finite_group_count = select(0u, 1u, light_table.light_bvh_node_count > 0u);
        let group_count = light_table.infinite_light_count + finite_group_count;
        if (group_count == 0u) {
            return LightSelection(0xffffffffu, 0.0);
        }
        let infinite_probability = f32(light_table.infinite_light_count) / f32(group_count);
        let u = min(selector, 0.99999994);
        if (u < infinite_probability) {
            let infinite_selector = u / infinite_probability;
            let offset = min(
                u32(infinite_selector * f32(light_table.infinite_light_count)),
                light_table.infinite_light_count - 1u,
            );
            return LightSelection(
                light_table.finite_light_count + offset,
                1.0 / f32(group_count),
            );
        }
        if (finite_group_count == 0u) {
            return LightSelection(0xffffffffu, 0.0);
        }
        let finite_selection = sample_light_bvh(
            (u - infinite_probability) / (1.0 - infinite_probability),
            p,
            n,
        );
        return LightSelection(
            finite_selection.index,
            finite_selection.pmf / f32(group_count),
        );
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

// pbrt-v4 CoordinateSystem(): return the x axis paired with a unit z axis.
fn coordinate_system_x(z: vec3<f32>) -> vec3<f32> {
    let sign = select(1.0, -1.0, (bitcast<u32>(z.z) & 0x80000000u) != 0u);
    let a = -1.0 / (sign + z.z);
    let b = z.x * z.y * a;
    return vec3<f32>(1.0 + sign * z.x * z.x * a, sign * b, -sign * z.x);
}
var<private> material_texture_uv: vec2<f32>;
var<private> material_texture_normal: vec3<f32>;
var<private> material_texture_position: vec3<f32>;
var<private> material_texture_eval_base: u32;
