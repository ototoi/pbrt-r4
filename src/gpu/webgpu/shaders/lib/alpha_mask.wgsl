fn alpha_mask_hash(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    let hash = murmur_hash_64a_24(
        vec2<u32>(bitcast<u32>(origin.x), bitcast<u32>(origin.y)),
        vec2<u32>(bitcast<u32>(origin.z), bitcast<u32>(direction.x)),
        vec2<u32>(bitcast<u32>(direction.y), bitcast<u32>(direction.z)),
    );
    return f32(hash.x) * (1.0 / 4294967296.0);
}

fn alpha_mask_point_hash(position: vec3<f32>) -> f32 {
    let hash = murmur_hash_64a_12(
        vec2<u32>(bitcast<u32>(position.x), bitcast<u32>(position.y)),
        bitcast<u32>(position.z),
    );
    return f32(hash.x) * (1.0 / 4294967296.0);
}

fn alpha_mask_value(instance: Instance, uv: vec2<f32>, position: vec3<f32>, normal: vec3<f32>) -> f32 {
    if (instance.alpha_kind == ALPHA_MASK_OPAQUE) { return 1.0; }
    if (instance.alpha_kind == ALPHA_MASK_CONSTANT) { return bitcast<f32>(instance.alpha_value); }
    if (instance.alpha_kind == ALPHA_MASK_TEXTURE) {
        if (instance.alpha_value >= arrayLength(&texture_roots)) {
            set_render_error();
            return 0.0;
        }
        material_texture_uv = uv;
        material_texture_position = position;
        material_texture_normal = normal;
        return sample_texture_program(texture_roots[instance.alpha_value], uv).x;
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
    sampling_normal: vec3<f32>,
) -> bool {
    if (area_light_is_zero_alpha_sample_only(area_light)) { return true; }
    let instance_index = load_area_instance(area_light);
    let surface = reconstruct_triangle_surface(instance_index, primitive, barycentrics);
    if (surface.valid == 0u) { return false; }
    let instance = instances[instance_index];
    let alpha = alpha_mask_value(
        instance, surface.uv, position, sampling_normal,
    );
    return alpha_mask_point_accept(alpha, position);
}

fn alpha_mask_candidate_accept(origin: vec3<f32>, direction: vec3<f32>, hit: RayIntersection) -> bool {
    if (hit.instance_custom_data >= arrayLength(&instances)) {
        set_render_error();
        return false;
    }
    let instance = instances[hit.instance_custom_data];
    if (instance.alpha_kind == ALPHA_MASK_OPAQUE) { return true; }
    let b1 = hit.barycentrics.x;
    let b2 = hit.barycentrics.y;
    let surface = reconstruct_triangle_surface(
        hit.instance_custom_data, hit.primitive_index, vec3<f32>(1.0 - b1 - b2, b1, b2),
    );
    if (surface.valid == 0u) { return false; }
    let alpha = alpha_mask_value(
        instance, surface.uv, surface.position, surface.geometric_normal,
    );
    if (alpha >= 1.0) { return true; }
    if (alpha <= 0.0) { return false; }
    return alpha_mask_hash(origin, direction) <= alpha;
}
