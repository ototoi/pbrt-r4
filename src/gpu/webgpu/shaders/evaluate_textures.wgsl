@compute @workgroup_size(64, 1, 1)
fn evaluate_textures(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= material_eval_count()) {
        return;
    }
    let ray_index = load_material_eval_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let surface = surfaces[ray.pixel_index];
    material_texture_uv = surface.uv;
    material_texture_normal = surface.normal.xyz;
    material_texture_position = surface.position.xyz;
    let result_base = queue_index * material_table.texture_eval_stride;
    for (var slot = 0u; slot < material_table.texture_eval_stride; slot++) {
        texture_eval_results[result_base + slot] =
            TextureEvalResult(0u, 0u, 0u, 0u, vec4<f32>(0.0));
    }

    var result_slot = 0u;
    if (surface.material_root >= arrayLength(&material_roots)) {
        set_render_error();
        return;
    }
    let material_root = material_roots[surface.material_root];
    if (material_root.node_count > material_table.attributes_eval_stride
        || material_root.node_offset + material_root.node_count > arrayLength(&material_nodes)) {
        set_render_error();
        return;
    }
    var material_node_index = material_root.node_offset;
    for (var visited = 0u; visited < material_root.node_count; visited++) {
        let material_node = material_nodes[material_node_index];
        let material_node_index_value = material_node_index;
        let material = material_node;
        for (var ordinal = 0u; ordinal < material.attribute_count; ordinal++) {
            let attribute_ref = load_material_attribute(material_node_index_value, ordinal);
            if (attribute_ref.kind != 2u) {
                continue;
            }
            if (result_slot >= material_table.texture_eval_stride
                || attribute_ref.index >= arrayLength(&texture_roots)) {
                set_render_error();
                return;
            }
            let rgb = sample_texture_program(texture_roots[attribute_ref.index], surface.uv);
            texture_eval_results[result_base + result_slot] = TextureEvalResult(
                material_node_index_value,
                ordinal,
                attribute_ref.index,
                1u,
                vec4<f32>(rgb, 0.0),
            );
            result_slot += 1u;
        }
        material_node_index = next_material_node(material_root, material_node_index);
    }
}
