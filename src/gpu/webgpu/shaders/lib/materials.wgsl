fn load_material_kind_raw(index: u32) -> u32 {
    if (index >= arrayLength(&material_nodes) || index >= material_table.material_node_count) {
        set_render_error();
        return MATERIAL_KIND_NORMAL;
    }
    return material_nodes[index].kind;
}

fn load_material_kind(index: u32) -> u32 {
    if (material_table.debug_material_kind != 0xffffffffu) {
        let actual_kind = load_material_kind_raw(index);
        if (actual_kind != MATERIAL_KIND_ALPHA_MASK) {
            return material_table.debug_material_kind;
        }
        return actual_kind;
    }
    return load_material_kind_raw(index);
}

fn load_surface_material_kind(material_root: MaterialRoot) -> u32 {
    var material_node = material_root.node_offset;
    for (var depth = 0u; depth < material_root.node_count; depth++) {
        if (load_material_kind_raw(material_node) != MATERIAL_KIND_ALPHA_MASK) {
            return load_material_kind(material_node);
        }
        if (material_node >= arrayLength(&material_nodes)) {
            set_render_error();
            return MATERIAL_KIND_NORMAL;
        }
        let child = material_nodes[material_node].child0;
        if (child == 0xffffffffu || child < material_root.node_offset
            || child >= material_root.node_offset + material_root.node_count) {
            set_render_error();
            return MATERIAL_KIND_NORMAL;
        }
        material_node = child;
    }
    set_render_error();
    return MATERIAL_KIND_NORMAL;
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
        if (item.bxdf_kind == MATERIAL_KIND_ALPHA_MASK) {
            if (item.child_work_item0 == 0xffffffffu) { set_render_error(); return item; }
            current = item.child_work_item0;
            continue;
        }
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
