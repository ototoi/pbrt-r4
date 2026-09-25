fn alpha_mask_hash(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    let hash = murmur_hash_24(
        vec2<u32>(bitcast<u32>(origin.x), bitcast<u32>(origin.y)),
        vec2<u32>(bitcast<u32>(origin.z), bitcast<u32>(direction.x)),
        vec2<u32>(bitcast<u32>(direction.y), bitcast<u32>(direction.z)),
    );
    return f32(hash.x) * (1.0 / 4294967296.0);
}

fn alpha_mask_point_hash(position: vec3<f32>) -> f32 {
    let hash = murmur_hash_12(
        vec2<u32>(bitcast<u32>(position.x), bitcast<u32>(position.y)),
        bitcast<u32>(position.z),
    );
    return f32(hash.x) * (1.0 / 4294967296.0);
}

fn alpha_mask_value(material_root_index: u32, uv: vec2<f32>, position: vec3<f32>, normal: vec3<f32>) -> f32 {
    if (material_root_index >= arrayLength(&material_roots)) {
        set_render_error();
        return 0.0;
    }
    let material_root = material_roots[material_root_index];
    if (material_root.node_count == 0u
        || material_root.node_offset >= arrayLength(&material_nodes)) {
        set_render_error();
        return 0.0;
    }
    let material_node = material_root.node_offset;
    if (load_material_kind_raw(material_node) != MATERIAL_KIND_ALPHA_MASK) {
        return 1.0;
    }
    let alpha_ref = load_material_attribute(material_node, 0u);
    if (alpha_ref.kind == 0u) {
        if (alpha_ref.index >= arrayLength(&scalar_attributes)) {
            set_render_error();
            return 0.0;
        }
        return scalar_attributes[alpha_ref.index];
    }
    if (alpha_ref.kind == 2u) {
        if (alpha_ref.index >= arrayLength(&texture_roots)) {
            set_render_error();
            return 0.0;
        }
        material_texture_uv = uv;
        material_texture_position = position;
        material_texture_normal = normal;
        return sample_texture_program(texture_roots[alpha_ref.index], uv).x;
    }
    set_render_error();
    return 0.0;
}

fn alpha_mask_point_accept(alpha: f32, position: vec3<f32>) -> bool {
    if (alpha >= 1.0) { return true; }
    if (alpha <= 0.0) { return false; }
    return alpha_mask_point_hash(position) <= alpha;
}

fn alpha_area_sample_accept(
    area_light: u32,
    primitive: u32,
    barycentrics: vec3<f32>,
    position: vec3<f32>,
) -> bool {
    if (area_light_is_zero_alpha_sample_only(area_light)) { return true; }
    let instance_index = load_area_instance(area_light);
    let surface = reconstruct_triangle_surface(instance_index, primitive, barycentrics);
    if (surface.valid == 0u) { return false; }
    let instance = instances[instance_index];
    let alpha = alpha_mask_value(
        instance.material_root, surface.uv, position, surface.geometric_normal,
    );
    return alpha_mask_point_accept(alpha, position);
}

fn alpha_mask_candidate_accept(origin: vec3<f32>, direction: vec3<f32>, hit: RayIntersection) -> bool {
    if (hit.instance_custom_data >= arrayLength(&instances)) {
        set_render_error();
        return false;
    }
    let instance = instances[hit.instance_custom_data];
    if (instance.material_root >= arrayLength(&material_roots)) {
        set_render_error();
        return false;
    }
    let material_root = material_roots[instance.material_root];
    if (material_root.node_count == 0u
        || material_root.node_offset >= arrayLength(&material_nodes)) {
        set_render_error();
        return false;
    }
    let root_node_index = material_root.node_offset;
    if (load_material_kind_raw(root_node_index) != MATERIAL_KIND_ALPHA_MASK) {
        return true;
    }
    let b1 = hit.barycentrics.x;
    let b2 = hit.barycentrics.y;
    let surface = reconstruct_triangle_surface(
        hit.instance_custom_data, hit.primitive_index, vec3<f32>(1.0 - b1 - b2, b1, b2),
    );
    if (surface.valid == 0u) { return false; }
    let alpha = alpha_mask_value(
        instance.material_root, surface.uv, surface.position, surface.geometric_normal,
    );
    if (alpha >= 1.0) { return true; }
    if (alpha <= 0.0) { return false; }
    return alpha_mask_hash(origin, direction) <= alpha;
}
