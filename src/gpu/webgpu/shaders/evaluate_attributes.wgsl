fn evaluate_material_attributes(material_node: u32, lambda: vec4<f32>) -> AttributesEvalWorkItem {
    var e: AttributesEvalWorkItem;
    e.material_node = material_node;
    e.child_work_item0 = 0xffffffffu; e.child_work_item1 = 0xffffffffu;
    e.bxdf_kind = load_material_kind(material_node); e.selected_child_work_item = 0xffffffffu;
    e._padding0 = 0u; e._padding1 = 0u; e._padding2 = 0u;
    for (var i = 0u; i < 10u; i++) { e.values[i] = vec4<f32>(0.0); }
    if (e.bxdf_kind == MATERIAL_KIND_DIFFUSE) {
        e.values[0] = load_diffuse_reflectance(material_node, lambda);
    } else if (e.bxdf_kind == MATERIAL_KIND_DIFFUSE_TRANSMISSION) {
        let scale = load_diffuse_transmission_scale(material_node);
        e.values[0] = clamp(load_diffuse_transmission_reflectance(material_node, lambda) * scale, vec4<f32>(0.0), vec4<f32>(1.0));
        e.values[1] = clamp(load_diffuse_transmission_transmittance(material_node, lambda) * scale, vec4<f32>(0.0), vec4<f32>(1.0));
        e.values[2].x = scale;
    } else if (e.bxdf_kind == MATERIAL_KIND_MIX) {
        e.values[0].x = load_material_scalar(material_node, 0u);
    } else if (e.bxdf_kind == MATERIAL_KIND_CONDUCTOR_ETA_K) {
        e.values[0] = load_conductor_eta(material_node, lambda); e.values[1] = load_conductor_k(material_node, lambda); e.values[2].x = load_conductor_roughness(material_node);
    } else if (e.bxdf_kind == MATERIAL_KIND_CONDUCTOR_REFLECTANCE) {
        let r = clamp(load_material_spectrum(material_node, 0u, lambda), vec4<f32>(0.0), vec4<f32>(0.9999));
        e.values[0] = vec4<f32>(1.0);
        e.values[1] = 2.0 * sqrt(r) / sqrt(max(vec4<f32>(1.0) - r, vec4<f32>(1e-7)));
        e.values[2].x = load_conductor_reflectance_roughness(material_node);
    } else if (e.bxdf_kind == MATERIAL_KIND_DIELECTRIC) {
        e.values[0] = load_dielectric_eta(material_node, lambda); e.values[1].x = load_material_scalar(material_node, 1u); e.values[2].x = load_material_scalar(material_node, 2u); e.values[3].x = load_material_scalar(material_node, 3u);
    } else if (e.bxdf_kind == MATERIAL_KIND_THIN_DIELECTRIC) {
        e.values[0] = load_dielectric_eta(material_node, lambda);
    } else if (e.bxdf_kind == MATERIAL_KIND_COATED_DIFFUSE || e.bxdf_kind == MATERIAL_KIND_COATED_CONDUCTOR) {
        e.values[0].x = load_material_scalar(material_node, 0u); e.values[1] = clamp(load_material_spectrum(material_node, 1u, lambda), vec4<f32>(0.0), vec4<f32>(1.0)); e.values[2].x = clamp(load_material_scalar(material_node, 2u), -1.0, 1.0); e.values[3].x = load_material_scalar(material_node, 3u); e.values[4].x = load_material_scalar(material_node, 4u);
        if (e.bxdf_kind == MATERIAL_KIND_COATED_DIFFUSE) { e.values[5] = clamp(load_material_spectrum(material_node, 5u, lambda), vec4<f32>(0.0), vec4<f32>(1.0)); }
        else { e.values[5].x = load_material_scalar(material_node, 12u); }
    }
    return e;
}

fn evaluate_coated_child(parent: u32, parent_kind: u32, slot: u32, input: AttributesEvalWorkItem, lambda: vec4<f32>) -> AttributesEvalWorkItem {
    var e = input;
    if (slot == 0u && e.bxdf_kind == MATERIAL_KIND_DIELECTRIC) {
        let eta = select(5u, 6u, parent_kind == MATERIAL_KIND_COATED_DIFFUSE); let ur = select(6u, 7u, parent_kind == MATERIAL_KIND_COATED_DIFFUSE); let vr = select(7u, 8u, parent_kind == MATERIAL_KIND_COATED_DIFFUSE); let remap = select(12u, 9u, parent_kind == MATERIAL_KIND_COATED_DIFFUSE);
        e.values[0] = load_material_spectrum(parent, eta, lambda); e.values[1].x = load_material_scalar(parent, ur); e.values[2].x = load_material_scalar(parent, vr); e.values[3].x = load_material_scalar(parent, remap);
    } else if (slot == 1u && parent_kind == MATERIAL_KIND_COATED_DIFFUSE && e.bxdf_kind == MATERIAL_KIND_DIFFUSE) {
        e.values[0] = clamp(load_material_spectrum(parent, 1u, lambda), vec4<f32>(0.0), vec4<f32>(1.0));
    } else if (slot == 1u && parent_kind == MATERIAL_KIND_COATED_CONDUCTOR
        && (e.bxdf_kind == MATERIAL_KIND_CONDUCTOR_ETA_K
            || e.bxdf_kind == MATERIAL_KIND_CONDUCTOR_REFLECTANCE)) {
        var eta_scalar = load_material_spectrum(parent, 5u, lambda).x; if (eta_scalar == 0.0) { eta_scalar = 1.0; } let eta = vec4<f32>(eta_scalar);
        if (load_material_scalar(parent, 14u) != 0.0) { let r = clamp(load_material_spectrum(parent, 13u, lambda), vec4<f32>(0.0), vec4<f32>(0.9999)); e.values[0] = vec4<f32>(1.0) / eta; e.values[1] = 2.0 * sqrt(r) / sqrt(max(vec4<f32>(1.0) - r, vec4<f32>(1e-7))) / eta; }
        else { e.values[0] = max(load_material_spectrum(parent, 8u, lambda), vec4<f32>(0.0)) / eta; e.values[1] = max(load_material_spectrum(parent, 9u, lambda), vec4<f32>(0.0)) / eta; }
        e.values[2].x = load_material_scalar(parent, 10u); e.values[3].x = load_material_scalar(parent, 11u); e.values[4].x = load_material_scalar(parent, 12u);
    }
    return e;
}

fn resolved_bump_material_node(material_root: MaterialRoot, attributes_base: u32) -> u32 {
    var node_index = material_root.node_offset;
    for (var depth = 0u; depth < material_root.node_count; depth++) {
        let kind = load_material_kind_raw(node_index);
        if (kind == MATERIAL_KIND_MIX) {
            let work_index = attributes_base + node_index - material_root.node_offset;
            let evaluation = attributes_eval_work_items[work_index];
            if (evaluation.selected_child_work_item == 0xffffffffu
                || evaluation.selected_child_work_item < attributes_base) {
                set_render_error();
                return node_index;
            }
            let child_offset = evaluation.selected_child_work_item - attributes_base;
            if (child_offset >= material_root.node_count) {
                set_render_error();
                return node_index;
            }
            node_index = material_root.node_offset + child_offset;
        } else if (kind == MATERIAL_KIND_ALPHA_MASK) {
            let child = material_nodes[node_index].child0;
            if (child == 0xffffffffu) {
                set_render_error();
                return node_index;
            }
            node_index = child;
        } else {
            return node_index;
        }
    }
    set_render_error();
    return node_index;
}

fn apply_surface_bump(pixel: u32, surface: SurfaceWorkItem, material_root: MaterialRoot, attributes_base: u32) {
    let material_node = resolved_bump_material_node(material_root, attributes_base);
    if (material_node >= arrayLength(&material_nodes)) { set_render_error(); return; }
    let texture_root = material_nodes[material_node].displacement_texture_root;
    if (texture_root == 0xffffffffu) { return; }
    if (texture_root >= arrayLength(&texture_roots)) { set_render_error(); return; }

    let uv_differentials = surface.uv_differentials;
    var du = 0.5 * (abs(uv_differentials.x) + abs(uv_differentials.z));
    var dv = 0.5 * (abs(uv_differentials.y) + abs(uv_differentials.w));
    if (du == 0.0) { du = 0.0005; }
    if (dv == 0.0) { dv = 0.0005; }

    material_texture_uv_dx = vec2<f32>(uv_differentials.x, uv_differentials.y);
    material_texture_uv_dy = vec2<f32>(uv_differentials.z, uv_differentials.w);
    material_texture_dpdx = surface.dpdx.xyz;
    material_texture_dpdy = surface.dpdy.xyz;
    material_texture_normal = surface.normal.xyz;

    material_texture_position = surface.position.xyz + du * surface.dpdu.xyz;
    material_texture_uv = surface.uv + vec2<f32>(du, 0.0);
    let u_displacement = sample_texture_program(texture_roots[texture_root], material_texture_uv).x;

    material_texture_position = surface.position.xyz + dv * surface.dpdv.xyz;
    material_texture_uv = surface.uv + vec2<f32>(0.0, dv);
    let v_displacement = sample_texture_program(texture_roots[texture_root], material_texture_uv).x;

    material_texture_position = surface.position.xyz;
    material_texture_uv = surface.uv;
    let displacement = sample_texture_program(texture_roots[texture_root], surface.uv).x;

    // Mesh dndu/dndv come from per-triangle vertex-normal differences and
    // carry a component along the shading normal. With a nonzero constant
    // displacement that tilts the bumped normal by a different amount on
    // each triangle, leaving visible seams on tessellated shapes. Keep only
    // the tangent-plane part; analytic shapes already have dndu/dndv there.
    let shading_normal = surface.normal.xyz;
    let dndu = surface.dndu.xyz - dot(surface.dndu.xyz, shading_normal) * shading_normal;
    let dndv = surface.dndv.xyz - dot(surface.dndv.xyz, shading_normal) * shading_normal;
    let bumped_dpdu = surface.dpdu.xyz
        + ((u_displacement - displacement) / du) * shading_normal
        + displacement * dndu;
    let bumped_dpdv = surface.dpdv.xyz
        + ((v_displacement - displacement) / dv) * shading_normal
        + displacement * dndv;
    var bumped_normal = normalize(cross(bumped_dpdu, bumped_dpdv));
    if (dot(bumped_normal, surface.geometric_normal.xyz) < 0.0) {
        bumped_normal = -bumped_normal;
    }
    let bumped_tangent = normalize(bumped_dpdu);
    surfaces[pixel].normal = vec4<f32>(bumped_normal, 0.0);
    surfaces[pixel].tangent = vec4<f32>(bumped_tangent, 0.0);
}

@compute @workgroup_size(64, 1, 1)
fn evaluate_attributes(@builtin(global_invocation_id) id: vec3<u32>) {
    let queue_index = id.y * INDIRECT_ROW_ITEMS + id.x; if (queue_index >= material_eval_count()) { return; }
    let ray = load_current_ray(load_material_eval_ray(queue_index)); let pixel = ray.pixel_index; let surface = surfaces[pixel];
    material_texture_uv = surface.uv; material_texture_normal = surface.normal.xyz; material_texture_position = surface.position.xyz; material_texture_eval_base = queue_index * material_table.texture_eval_stride;
    if (surface.material_root >= arrayLength(&material_roots)) { set_render_error(); return; }
    let material_root = material_roots[surface.material_root];
    if (material_root.node_count == 0u || material_root.node_count > material_table.attributes_eval_stride || material_root.node_offset + material_root.node_count > arrayLength(&material_nodes)) { set_render_error(); return; }
    let base = queue_index * material_table.attributes_eval_stride; let lambda = load_sample_lambda(pixel); let samples = load_ray_samples(pixel); var node_index = material_root.node_offset;
    for (var visited = 0u; visited < material_root.node_count; visited++) {
        let node = material_nodes[node_index]; var e = evaluate_material_attributes(node_index, lambda);
        if (node.parent != 0xffffffffu) { let pk = load_material_kind(node.parent); if (pk == MATERIAL_KIND_COATED_DIFFUSE || pk == MATERIAL_KIND_COATED_CONDUCTOR) { e = evaluate_coated_child(node.parent, pk, node.parent_slot, e, lambda); } }
        if (node.child0 != 0xffffffffu) { e.child_work_item0 = base + node.child0 - material_root.node_offset; }
        if (node.child1 != 0xffffffffu) { e.child_work_item1 = base + node.child1 - material_root.node_offset; }
        if (e.bxdf_kind == MATERIAL_KIND_MIX) { if (node.child0 == 0xffffffffu || node.child1 == 0xffffffffu) { set_render_error(); return; } let amount = clamp(e.values[0].x, 0.0, 1.0); e.selected_child_work_item = select(e.child_work_item1, e.child_work_item0, samples.indirect.x < amount); }
        attributes_eval_work_items[base + node_index - material_root.node_offset] = e; node_index = next_material_node(material_root, node_index);
    }
    if (surface.flags == 0u) {
        apply_surface_bump(pixel, surface, material_root, base);
    }
}
