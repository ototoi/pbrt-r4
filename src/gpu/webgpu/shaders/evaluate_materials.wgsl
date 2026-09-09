@compute @workgroup_size(8, 8, 1)
fn evaluate_materials(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    material_texture_uv = surface.uv;
    material_texture_normal = surface.normal.xyz;
    material_texture_position = surface.position.xyz;
    var material_index = resolve_material_leaf(surface.material);
    var material_kind = load_material_kind(material_index);
    let lambda = load_sample_lambda(pixel_index);
    let samples = load_ray_samples(pixel_index);
    var current_index = surface.material;
    var parent_index = 0xffffffffu;
    var parent_slot = 0xffffffffu;
    let attributes_eval_base = queue_index * material_table.attributes_eval_stride;
    // Clear the complete per-item slice before filling the nodes that are
    // currently materialized. This keeps future recursive expansion from
    // observing stale records in a reused queue buffer.
    for (var clear_slot = 0u; clear_slot < material_table.attributes_eval_stride; clear_slot++) {
        var empty: AttributesEvalWorkItem;
        empty.surface_index = pixel_index;
        empty.material_index = 0xffffffffu;
        empty.parent_work_item = 0xffffffffu;
        empty.parent_slot = 0xffffffffu;
        empty.child_work_item0 = 0xffffffffu;
        empty.child_work_item1 = 0xffffffffu;
        empty.bxdf_kind = 0u;
        empty.selected_child_work_item = 0xffffffffu;
        for (var clear_value = 0u; clear_value < 10u; clear_value++) {
            empty.values[clear_value] = vec4<f32>(0.0);
        }
        attributes_eval_work_items[attributes_eval_base + clear_slot] = empty;
    }
    for (var tree_depth = 0u; tree_depth < 1u; tree_depth++) {
        let work_index = attributes_eval_base + tree_depth;
        var evaluated: AttributesEvalWorkItem;
        evaluated.surface_index = pixel_index;
        evaluated.material_index = current_index;
        evaluated.parent_work_item = parent_index;
        evaluated.parent_slot = parent_slot;
        evaluated.child_work_item0 = 0xffffffffu;
        evaluated.child_work_item1 = 0xffffffffu;
        evaluated.bxdf_kind = load_material_kind(current_index);
        evaluated.selected_child_work_item = 0xffffffffu;
        for (var value_index = 0u; value_index < 10u; value_index++) {
            evaluated.values[value_index] = vec4<f32>(0.0);
        }
        // Reserved throughput slots for the layered-BxDF walk.
        evaluated.values[8] = vec4<f32>(1.0);
        evaluated.values[9] = vec4<f32>(1.0);
        if (evaluated.bxdf_kind == MATERIAL_KIND_DIFFUSE) {
            evaluated.values[0] = load_diffuse_reflectance(current_index, lambda);
        } else if (evaluated.bxdf_kind == MATERIAL_KIND_MIX) {
            evaluated.values[0].x = load_material_scalar(current_index, 2u);
        } else if (evaluated.bxdf_kind == MATERIAL_KIND_CONDUCTOR) {
            evaluated.values[0] = load_conductor_eta(current_index, lambda);
            evaluated.values[1] = load_conductor_k(current_index, lambda);
            evaluated.values[2].x = load_conductor_roughness(current_index);
        } else if (evaluated.bxdf_kind == MATERIAL_KIND_DIELECTRIC || evaluated.bxdf_kind == MATERIAL_KIND_THIN_DIELECTRIC) {
            evaluated.values[0] = load_dielectric_eta(current_index, lambda);
        } else if (evaluated.bxdf_kind == MATERIAL_KIND_COATED_DIFFUSE) {
            evaluated.values[0].x = load_material_scalar(current_index, 2u);
            evaluated.values[1] = load_material_spectrum(current_index, 3u, lambda);
            evaluated.values[2].x = load_material_scalar(current_index, 4u);
            evaluated.values[3].x = load_material_scalar(current_index, 5u);
            evaluated.values[4].x = load_material_scalar(current_index, 6u);
            evaluated.values[5] = load_material_spectrum(current_index, 7u, lambda);
        } else if (evaluated.bxdf_kind == MATERIAL_KIND_COATED_CONDUCTOR) {
            evaluated.values[0].x = load_material_scalar(current_index, 2u);
            evaluated.values[1].x = load_material_scalar(current_index, 3u);
            evaluated.values[2].x = load_material_scalar(current_index, 4u);
            evaluated.values[3].x = load_material_scalar(current_index, 5u);
        }
        let current_kind = load_material_kind(current_index);
        if (current_kind == MATERIAL_KIND_MIX || current_kind == MATERIAL_KIND_COATED_DIFFUSE || current_kind == MATERIAL_KIND_COATED_CONDUCTOR) {
            let child0 = load_material_attribute(current_index, 0u);
            let child1 = load_material_attribute(current_index, 1u);
            if (child0.kind != 3u || child1.kind != 3u) { set_render_error(); break; }
            evaluated.child_work_item0 = work_index + 1u;
            evaluated.child_work_item1 = work_index + 2u;
            if (current_kind == MATERIAL_KIND_MIX) {
                let amount = clamp(load_material_scalar(current_index, 2u), 0.0, 1.0);
                let choice = select(work_index + 2u, work_index + 1u, samples.indirect.x < amount);
                evaluated.selected_child_work_item = choice;
            }
            attributes_eval_work_items[work_index] = evaluated;
            var child_eval: AttributesEvalWorkItem;
            child_eval.surface_index = pixel_index;
            child_eval.material_index = child0.index;
            child_eval.parent_work_item = work_index;
            child_eval.parent_slot = 0u;
            child_eval.child_work_item0 = 0xffffffffu;
            child_eval.child_work_item1 = 0xffffffffu;
            child_eval.bxdf_kind = load_material_kind(child0.index);
            child_eval.selected_child_work_item = 0xffffffffu;
            for (var child_value_index = 0u; child_value_index < 10u; child_value_index++) {
                child_eval.values[child_value_index] = vec4<f32>(0.0);
            }
            if (child_eval.bxdf_kind == MATERIAL_KIND_DIELECTRIC || child_eval.bxdf_kind == MATERIAL_KIND_THIN_DIELECTRIC) {
                child_eval.values[0] = load_dielectric_eta(child0.index, lambda);
            } else if (child_eval.bxdf_kind == MATERIAL_KIND_DIFFUSE) {
                child_eval.values[0] = load_diffuse_reflectance(child0.index, lambda);
            }
            attributes_eval_work_items[work_index + 1u] = child_eval;
            child_eval.material_index = child1.index;
            child_eval.parent_slot = 1u;
            child_eval.bxdf_kind = load_material_kind(child1.index);
            for (var child1_value_index = 0u; child1_value_index < 10u; child1_value_index++) {
                child_eval.values[child1_value_index] = vec4<f32>(0.0);
            }
            if (child_eval.bxdf_kind == MATERIAL_KIND_CONDUCTOR) {
                child_eval.values[0] = load_conductor_eta(child1.index, lambda);
                child_eval.values[1] = load_conductor_k(child1.index, lambda);
                child_eval.values[2].x = load_conductor_roughness(child1.index);
            } else if (child_eval.bxdf_kind == MATERIAL_KIND_DIFFUSE) {
                child_eval.values[0] = load_diffuse_reflectance(child1.index, lambda);
            }
            attributes_eval_work_items[work_index + 2u] = child_eval;
            if (child_eval.bxdf_kind == MATERIAL_KIND_MIX
                || child_eval.bxdf_kind == MATERIAL_KIND_COATED_DIFFUSE
                || child_eval.bxdf_kind == MATERIAL_KIND_COATED_CONDUCTOR) {
                set_render_error();
            }
            parent_index = work_index;
            parent_slot = 0u;
            current_index = child0.index;
        } else {
            attributes_eval_work_items[work_index] = evaluated;
            break;
        }
    }
    let root_evaluated = load_attributes_eval_work_item(surface.attributes_eval_work_item);
    if (load_material_kind(surface.material) == MATERIAL_KIND_MIX
        && root_evaluated.selected_child_work_item != 0xffffffffu) {
        let selected = load_attributes_eval_work_item(root_evaluated.selected_child_work_item);
        material_index = selected.material_index;
        material_kind = selected.bxdf_kind;
    }
    let root_surface_kind = load_material_kind(surface.material);
    var coated_diffuse_root = false;
    if (root_surface_kind == MATERIAL_KIND_COATED_DIFFUSE) {
        // Phase 1 of layered evaluation: expose the evaluated bottom
        // reflectance to the direct-light path. The full v4 LayeredBxDF random
        // walk is implemented behind the same parameter seam in a later phase.
        material_kind = MATERIAL_KIND_DIFFUSE;
        coated_diffuse_root = true;
    }
    if (surface.hit == 0u
        || (material_kind != MATERIAL_KIND_DIFFUSE
            && material_kind != MATERIAL_KIND_CONDUCTOR)) {
        return;
    }
    var reflectance = vec4<f32>(0.0);
    if (material_kind == MATERIAL_KIND_DIFFUSE && coated_diffuse_root) {
        reflectance = load_coated_diffuse_params(root_evaluated).reflectance;
    } else if (material_kind == MATERIAL_KIND_DIFFUSE) {
        reflectance = load_diffuse_reflectance(material_index, lambda);
    }
    if (ray.depth >= viewport.max_depth || light_table.light_count == 0u) {
        return;
    }
    let wo = -ray.direction.xyz;
    let light_sample_origin = offset_ray_origin(
        surface.position.xyz,
        surface.position_error.xyz,
        surface.geometric_normal.xyz,
        wo,
    );
    let light_selection = sample_scene_light(samples.direct.x, surface.position.xyz, surface.normal.xyz);
    if (light_selection.pmf <= 0.0 || light_selection.index == 0xffffffffu) {
        return;
    }
    let light_index = light_selection.index;
    let light_kind = load_light_kind(light_index);
    let light_payload = load_light_payload(light_index);
    var light_position = vec3<f32>(0.0);
    var light_error = vec3<f32>(0.0);
    var light_normal = vec3<f32>(0.0);
    var light_radiance = vec4<f32>(0.0);
    var sampled_light_pdf = light_selection.pmf;
    if (light_kind == LIGHT_KIND_POINT) {
        light_position = load_point_position(light_index);
        light_radiance = load_light_spectrum(light_index, 0u, lambda) * load_light_scale(light_index);
    } else if (light_kind == LIGHT_KIND_AREA) {
        let total_area = load_area_total(light_payload);
        let distribution_count = load_area_distribution_count(light_payload);
        if (total_area <= 0.0 || distribution_count == 0u) {
            return;
        }
        let triangle_selection = select_area_triangle(light_payload, samples.direct.y);
        let triangle = load_area_triangle(light_payload, triangle_selection.primitive);
        let triangle_sample = sample_uniform_triangle_for_context(
            triangle,
            light_sample_origin,
            vec2<f32>(samples.direct.z, samples.direct.w),
            triangle_selection.area,
        );
        if (triangle_sample.w <= 0.0) {
            return;
        }
        let b = triangle_sample.xyz;
        light_normal = triangle_geometric_normal(triangle);
        light_position = triangle.p0.xyz * b.x + triangle.p1.xyz * b.y + triangle.p2.xyz * b.z;
        light_radiance = load_light_spectrum(light_index, 0u, lambda) * load_light_scale(light_index);
        light_error = (abs(triangle.p0.xyz * b.x) + abs(triangle.p1.xyz * b.y)
            + abs(triangle.p2.xyz * b.z)) * gamma(6.0);
        let area_wi = normalize(light_position - light_sample_origin);
        let cosine_light = dot(light_normal, -area_wi);
        if (load_area_two_sided(light_payload)) {
            if (abs(cosine_light) == 0.0) {
                return;
            }
        } else if (cosine_light <= 0.0) {
            return;
        }
        sampled_light_pdf = sampled_light_pdf * triangle_selection.pmf * triangle_sample.w;
    } else {
        return;
    }
    let to_light = light_position - light_sample_origin;
    let distance_squared = dot(to_light, to_light);
    if (distance_squared <= 0.0) {
        return;
    }
    let distance = sqrt(distance_squared);
    let wi = to_light / distance;
    // pbrt-v4 semantics: DiffuseBxDF::f returns R/pi only when wo and wi lie
    // in the same hemisphere of the shading frame (SameHemisphere), and
    // SampleLd weights it with AbsDot(wi, shading.n). This keeps diffuse
    // lighting correct even when the mesh shading normals are globally
    // inverted (e.g. loopsubdiv limit normals wind opposite to the faces).
    let shading_n = surface.normal.xyz;
    let cos_wo = dot(shading_n, wo);
    let cos_wi = dot(shading_n, wi);
    if (cos_wo * cos_wi <= 0.0) {
        return;
    }
    let cosine = abs(cos_wi);
    if (cosine == 0.0) {
        return;
    }
    if (light_kind == LIGHT_KIND_POINT) {
        light_radiance = light_radiance / distance_squared;
    }
    var bsdf_pdf = cosine / PI;
    var f = reflectance / PI;
    if (material_kind == MATERIAL_KIND_CONDUCTOR) {
        let eta = load_conductor_eta(material_index, lambda);
        let k = load_conductor_k(material_index, lambda);
        let h = scattering_local(normalize(wo + wi), shading_n);
        let fresnel = conductor_fresnel(dot(scattering_local(wo, shading_n), h), eta, k);
        let alpha = max(load_conductor_roughness(material_index), 1e-3);
        let cos_h = max(abs(h.z), 1e-5);
        let alpha2 = alpha * alpha;
        let d = alpha2 / (PI * pow(cos_h * cos_h * (alpha2 - 1.0) + 1.0, 2.0));
        let cos_o = max(abs(cos_wo), 1e-5);
        let cos_i = max(abs(cos_wi), 1e-5);
        let g_o = 2.0 * cos_o / (cos_o + sqrt(cos_o * cos_o + alpha2 * (1.0 - cos_o * cos_o)));
        let g_i = 2.0 * cos_i / (cos_i + sqrt(cos_i * cos_i + alpha2 * (1.0 - cos_i * cos_i)));
        f = fresnel * d * g_o * g_i / (4.0 * cos_o * cos_i);
        bsdf_pdf = d * cos_h / max(4.0 * abs(dot(scattering_local(wo, shading_n), h)), 1e-5);
    }
    let surface_kind = load_material_kind(surface.material);
    if (surface_kind == MATERIAL_KIND_COATED_DIFFUSE || surface_kind == MATERIAL_KIND_COATED_CONDUCTOR) {
        let coat = load_attributes_eval_work_item(surface.attributes_eval_work_item + 1u);
        let eta = max(coat.values[0].x, 1.0001);
        let cos_i = clamp(abs(cos_wi), 0.0, 1.0);
        let coat_f = dielectric_fresnel(cos_i, eta);
        f = f * (1.0 - coat_f);
    }
    var mis_weight = 1.0;
    if (light_kind == LIGHT_KIND_AREA) {
        mis_weight = sampled_light_pdf / max(sampled_light_pdf + bsdf_pdf, 1e-7);
    }
    let direct = light_radiance * f * cosine
        / (max(ray.inv_w_u, 1e-7) * sampled_light_pdf)
        * mis_weight;
    let shadow_origin = light_sample_origin;
    var shadow_target = light_position;
    if (light_kind == LIGHT_KIND_AREA) {
        shadow_target = offset_ray_origin(light_position, light_error, light_normal, -wi);
    }
    let shadow_vector = shadow_target - shadow_origin;
    let shadow_distance = length(shadow_vector);
    if (shadow_distance <= 0.0) {
        return;
    }
    append_shadow_ray(
        pixel_index,
        shadow_origin,
        shadow_vector / shadow_distance,
        shadow_distance,
        (ray.throughput * direct),
    );
}
