@compute @workgroup_size(8, 8, 1)
fn evaluate_textures(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let queue_index = global_id.y * viewport.width + global_id.x;
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
    if (surface.material_tree_layout >= arrayLength(&material_tree_layouts)) {
        set_render_error();
        return;
    }
    let tree_layout = material_tree_layouts[surface.material_tree_layout];
    if (tree_layout.node_count > material_table.attributes_eval_stride
        || tree_layout.node_offset + tree_layout.node_count > arrayLength(&material_tree_nodes)) {
        set_render_error();
        return;
    }
    var tree_node_index = tree_layout.node_offset;
    for (var visited = 0u; visited < tree_layout.node_count; visited++) {
        let tree_node = material_tree_nodes[tree_node_index];
        let material_node = tree_node_index;
        let material = tree_node;
        for (var ordinal = 0u; ordinal < material.attribute_count; ordinal++) {
            let attribute_ref = load_material_attribute(material_node, ordinal);
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
                material_node,
                ordinal,
                attribute_ref.index,
                1u,
                vec4<f32>(rgb, 0.0),
            );
            result_slot += 1u;
        }
        tree_node_index = next_material_tree_node(tree_layout, tree_node_index);
    }
}
