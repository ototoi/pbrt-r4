@compute @workgroup_size(64, 1, 1)
fn classify_surface_scatter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
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
    let evaluated = resolve_attributes_eval_work_item(surface.attributes_eval_work_item);
    let root_kind = evaluated.bxdf_kind;
    var supports_direct = false;
    if (root_kind == MATERIAL_KIND_DIFFUSE) {
        append_scatter_diffuse(ray_index);
        supports_direct = true;
    } else if (root_kind == MATERIAL_KIND_DIFFUSE_TRANSMISSION) {
        append_scatter_diffuse_transmission(ray_index);
        supports_direct = true;
    } else if (root_kind == MATERIAL_KIND_CONDUCTOR_ETA_K || root_kind == MATERIAL_KIND_CONDUCTOR_REFLECTANCE) {
        append_scatter_conductor(ray_index);
        // pbrt-v4 ConductorBxDF::Flags(): EffectivelySmooth (alpha <= 1e-3,
        // matching scatter_conductor's own indirect-bounce threshold) is
        // Specular, so SampleLd is skipped for a mirror-smooth conductor.
        supports_direct = evaluated.values[2].x > 1e-3;
    } else if (root_kind == MATERIAL_KIND_DIELECTRIC) {
        append_scatter_dielectric(ray_index);
        // pbrt-v4 DielectricBxDF::Flags(): a rough (non-EffectivelySmooth)
        // interface is Glossy, so SampleLd applies; eta == 1 or a smooth
        // interface stays Specular with no direct lighting.
        let eta = select(evaluated.values[0].x, 1.0, evaluated.values[0].x == 0.0);
        let alpha = dielectric_interface_alpha(evaluated);
        supports_direct = eta != 1.0 && max(alpha.x, alpha.y) >= 1e-3;
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
