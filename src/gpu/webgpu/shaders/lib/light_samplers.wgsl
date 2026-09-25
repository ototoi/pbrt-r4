fn uniform_light_pmf_for_handle(light_handle: u32) -> f32 {
    if (light_handle >= light_table.light_count) {
        return 0.0;
    }
    return 1.0 / f32(light_table.light_count);
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
