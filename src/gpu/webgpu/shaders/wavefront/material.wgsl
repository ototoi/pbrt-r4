@compute @workgroup_size(64)
fn evaluate_material(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slot_index = global_id.x;
    if (slot_index >= pixel_count() || slot_index >= arena.capacity) {
        return;
    }

    let ray = arena.rays[slot_index];
    if (ray.indices.y != RAY_STATE_SURFACE) {
        return;
    }

    let primitive = primitives[ray.indices.z];
    let material = materials[primitive.material];
    var reflectance = material.reflectance.xyz;
    if ((material.flags & 1u) != 0u) {
        reflectance = sample_texture(material.texture, ray.surface_uv.xy, vec4<f32>(0.0));
    }
    arena.rays[slot_index].material_reflectance = vec4<f32>(reflectance, 1.0);
    arena.rays[slot_index].indices.y = RAY_STATE_MATERIAL;
}
