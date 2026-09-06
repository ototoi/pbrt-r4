@compute @workgroup_size(8, 8, 1)
fn sample_layered_bounce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let ray_index = global_id.y * viewport.width + global_id.x;
    if (ray_index >= current_ray_count()) {
        return;
    }
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let samples = load_ray_samples(pixel_index);
    let surface = surfaces[pixel_index];
    if (surface.hit == 0u || surface.flags != 0u
        || load_material_kind(surface.material) != MATERIAL_KIND_LAYERED) {
        return;
    }

    let layered = load_layered_bxdf(load_layered_data_index(surface.material));
    if (layered.max_depth == 0u || layered.n_samples == 0u
        || max(layered.albedo.x, max(layered.albedo.y, layered.albedo.z)) > 0.0) {
        atomicStore(&wavefront_queue[RENDER_ERROR], 1u);
        return;
    }
    let model_index = load_material_model(surface.material);
    let root = scene_data[scene.scattering_model_offset_words + model_index * 4u];
    let top = load_scattering_child(root, 0u);
    if (load_scattering_node_word(top, 0u) != 1u) {
        atomicStore(&wavefront_queue[RENDER_ERROR], 1u);
        return;
    }
    let eta = load_dielectric_eta(load_scattering_node_word(top, 2u));
    if (!(eta > 0.0) || eta != eta) {
        atomicStore(&wavefront_queue[RENDER_ERROR], 1u);
        return;
    }

    let wo = normalize(-ray.direction.xyz);
    var normal = normalize(surface.normal.xyz);
    var eta_i = 1.0;
    var eta_t = eta;
    if (dot(wo, normal) < 0.0) {
        normal = -normal;
        eta_i = eta;
        eta_t = 1.0;
    }
    let cos_i = clamp(dot(wo, normal), 0.0, 1.0);
    let eta_ratio = eta_i / eta_t;
    let sin2_t = eta_ratio * eta_ratio * max(0.0, 1.0 - cos_i * cos_i);
    var fresnel = 1.0;
    if (sin2_t < 1.0) {
        let cos_t = sqrt(max(0.0, 1.0 - sin2_t));
        let r_parallel = (eta_t * cos_i - eta_i * cos_t)
            / (eta_t * cos_i + eta_i * cos_t);
        let r_perpendicular = (eta_i * cos_i - eta_t * cos_t)
            / (eta_i * cos_i + eta_t * cos_t);
        fresnel = 0.5 * (r_parallel * r_parallel + r_perpendicular * r_perpendicular);
    }

    var direction = reflect(-wo, normal);
    var next_throughput = ray.throughput;
    var next_pdf = max(fresnel, 1e-7);
    if (samples.indirect.x >= fresnel && sin2_t < 1.0) {
        let tangent = make_tangent(normal);
        let bitangent = cross(normal, tangent);
        let u = vec2<f32>(samples.indirect.y, samples.indirect.z);
        let radius = sqrt(u.x);
        let phi = 2.0 * PI * u.y;
        let local = vec3<f32>(
            radius * cos(phi),
            radius * sin(phi),
            sqrt(max(0.0, 1.0 - u.x)),
        );
        direction = normalize(tangent * local.x + bitangent * local.y + normal * local.z);
        let bottom_reflectance = load_layered_bottom_reflectance(surface.material);
        next_throughput = next_throughput * vec4<f32>(bottom_reflectance, 1.0);
        next_pdf = max(1.0 - fresnel, 1e-7) * max(dot(normal, direction), 0.0) / PI;
    }
    direction = normalize(direction);
    let next_ray = RayWorkItem(
        vec4<f32>(offset_ray_origin(surface.position.xyz, surface.position_error.xyz,
            surface.geometric_normal.xyz, direction), 1.0),
        vec4<f32>(direction, 0.0),
        next_throughput,
        surface.position,
        surface.position_error,
        surface.geometric_normal,
        vec4<f32>(normal, 0.0),
        pixel_index,
        ray.depth + 1u,
        ray.inv_w_u,
        ray.inv_w_u / next_pdf,
        next_pdf,
        vec3<u32>(0u, 0u, 0u),
    );
    let next_index = atomicAdd(&wavefront_queue[NEXT_COUNT], 1u);
    if (next_index >= pixel_count()) {
        atomicStore(&wavefront_queue[NEXT_OVERFLOW], 1u);
        return;
    }
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, ray.depth + 1u));
    store_next_ray(next_index, next_ray);
}
