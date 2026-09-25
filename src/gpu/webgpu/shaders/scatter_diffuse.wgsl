@compute @workgroup_size(8, 8, 1)
fn scatter_diffuse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let queue_index = global_id.y * viewport.width + global_id.x;
    if (queue_index >= scatter_diffuse_count()) {
        return;
    }
    let ray_index = load_scatter_diffuse_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    let evaluated = resolve_attributes_eval_work_item(surface.attributes_eval_work_item);

    // Direct lighting.
    let light_sample = direct_light_samples[pixel_index];
    if (light_sample.valid != 0u) {
        let reflectance = evaluated.values[0];
        let wo = -ray.direction.xyz;
        let wi = light_sample.direction_pdf.xyz;
        // pbrt-v4 semantics: DiffuseBxDF::f returns R/pi only when wo and wi lie
        // in the same hemisphere of the shading frame (SameHemisphere), and
        // SampleLd weights it with AbsDot(wi, shading.n). This keeps diffuse
        // lighting correct even when the mesh shading normals are globally
        // inverted (e.g. loopsubdiv limit normals wind opposite to the faces).
        let shading_n = surface.normal.xyz;
        let cos_wo = dot(shading_n, wo);
        let cos_wi = dot(shading_n, wi);
        if (cos_wo * cos_wi > 0.0) {
            let cosine = abs(cos_wi);
            if (cosine > 0.0) {
                let bsdf_pdf = cosine / PI;
                let f = reflectance / PI;
                add_direct_lighting(ray, surface, light_sample, f, bsdf_pdf, cosine);
            }
        }
    }

    // Indirect bounce.
    let normal = surface.normal.xyz;
    var tangent = make_tangent(normal);
    let bitangent = cross(normal, tangent);
    let wo = -ray.direction.xyz;
    let reflectance = evaluated.values[0];
    let samples = load_ray_samples(pixel_index);
    let u = vec2<f32>(samples.indirect.y, samples.indirect.z);
    let radius = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    var local = vec3<f32>(
        radius * cos(phi),
        radius * sin(phi),
        sqrt(max(0.0, 1.0 - u.x)),
    );
    // pbrt-v4 DiffuseBxDF::Sample_f: sample the cosine hemisphere on the
    // side of the shading frame that contains wo (`if (wo.z < 0) wi.z *= -1`),
    // and use AbsCosTheta for the pdf. This bounces outward even when the
    // mesh shading normals are globally inverted.
    if (dot(normal, wo) < 0.0) {
        local.z = -local.z;
    }
    let direction = normalize(tangent * local.x + bitangent * local.y + normal * local.z);
    let next_pdf = abs(dot(normal, direction)) / PI;
    var next_throughput = ray.throughput * reflectance;
    if (ray.depth >= 1u) {
        let rr_beta = max(
            max_spectrum(next_throughput),
            0.0,
        ) / max(ray.inv_w_u, 1e-7);
        let q = max(0.0, 1.0 - rr_beta);
        if (samples.indirect.w < q) {
            return;
        }
        next_throughput = next_throughput / max(1.0 - q, 1e-7);
    }
    let next_ray = RayWorkItem(
        vec4<f32>(offset_ray_origin(surface.position.xyz, surface.position_error.xyz, surface.geometric_normal.xyz, direction), 1.0),
        vec4<f32>(direction, 0.0),
        next_throughput,
        surface.position,
        surface.position_error,
        surface.geometric_normal,
        vec4<f32>(normal, 0.0),
        pixel_index,
        ray.depth + 1u,
        ray.inv_w_u,
        ray.inv_w_u / max(next_pdf, 1e-7),
        next_pdf,
        0u, 0u, 0u,
    );
    let next_index = atomicAdd(&queue_counters.next.count, 1u);
    if (next_index >= pixel_count()) {
        atomicStore(&queue_counters.next.overflow, 1u);
        return;
    }
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, ray.depth + 1u));
    store_next_ray(next_index, next_ray);
}
