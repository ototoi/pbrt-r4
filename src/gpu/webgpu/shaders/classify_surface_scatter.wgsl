@compute @workgroup_size(8, 8, 1)
fn classify_surface_scatter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let queue_index = global_id.y * viewport.width + global_id.x;
    if (queue_index >= material_eval_count()) {
        return;
    }
    let ray_index = load_material_eval_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    if (surface.hit == 0u || surface.flags != 0u) {
        return;
    }
    let root_kind = resolve_attributes_eval_work_item(surface.attributes_eval_work_item).bxdf_kind;
    var supports_direct = false;
    if (root_kind == MATERIAL_KIND_DIFFUSE) {
        append_scatter_diffuse(ray_index);
        supports_direct = true;
    } else if (root_kind == MATERIAL_KIND_DIFFUSE_TRANSMISSION) {
        append_scatter_diffuse_transmission(ray_index);
        supports_direct = true;
    } else if (root_kind == MATERIAL_KIND_CONDUCTOR_ETA_K || root_kind == MATERIAL_KIND_CONDUCTOR_REFLECTANCE) {
        append_scatter_conductor(ray_index);
        supports_direct = true;
    } else if (root_kind == MATERIAL_KIND_DIELECTRIC) {
        append_scatter_dielectric(ray_index);
    } else if (root_kind == MATERIAL_KIND_THIN_DIELECTRIC) {
        append_scatter_thin_dielectric(ray_index);
    } else if (root_kind == MATERIAL_KIND_MEASURED) {
        append_scatter_measured(ray_index);
        supports_direct = true;
    } else if (root_kind == MATERIAL_KIND_COATED_DIFFUSE || root_kind == MATERIAL_KIND_COATED_CONDUCTOR) {
        append_scatter_coated(ray_index);
        supports_direct = true;
    } else {
        return;
    }
    if (supports_direct && ray.depth < viewport.max_depth && light_table.light_count != 0u) {
        append_direct_eval(ray_index);
    }
}
