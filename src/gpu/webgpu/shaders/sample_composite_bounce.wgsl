@compute @workgroup_size(8, 8, 1)
fn sample_composite_bounce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) { return; }
    let ray_index = global_id.y * viewport.width + global_id.x;
    if (ray_index >= current_ray_count()) { return; }
    let ray = load_current_ray(ray_index);
    let surface = surfaces[ray.pixel_index];
    if (surface.hit == 0u || surface.flags != 0u) { return; }
    let kind = load_material_kind(surface.material);
    if (kind != MATERIAL_KIND_COATED_DIFFUSE && kind != MATERIAL_KIND_COATED_CONDUCTOR) { return; }
    // LayeredBxDF sampling is implemented in this stage boundary. Until the
    // full random-walk sampler is installed, leave the ray unmodified.
}
