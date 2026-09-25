fn current_ray_count() -> u32 {
    return atomicLoad(&queue_counters.current.count);
}

fn next_ray_count() -> u32 {
    return atomicLoad(&queue_counters.next.count);
}

fn shadow_ray_count() -> u32 {
    return atomicLoad(&queue_counters.shadow.count);
}

fn append_shadow_ray(pixel_index: u32, origin: vec3<f32>, direction: vec3<f32>, t: f32, direct: vec4<f32>) {
    let index = atomicAdd(&queue_counters.shadow.count, 1u);
    if (index < queue_counters.shadow.capacity) {
        shadow_rays[index] = ShadowRayWorkItem(
            vec4<f32>(origin, 0.0),
            vec4<f32>(direction, 0.0),
            t,
            0u, 0u, 0u,
            direct,
            pixel_index,
            0u, 0u, 0u,
        );
    } else {
        atomicStore(&queue_counters.shadow.overflow, 1u);
    }
}

fn load_shadow_pixel(index: u32) -> u32 {
    return shadow_rays[index].pixel_index;
}

fn load_shadow_t(index: u32) -> f32 {
    return shadow_rays[index].max_t;
}

fn load_shadow_direct(index: u32) -> vec4<f32> {
    return shadow_rays[index].direct;
}

fn load_shadow_origin(index: u32) -> vec3<f32> {
    return shadow_rays[index].origin.xyz;
}

fn load_shadow_direction(index: u32) -> vec3<f32> {
    return shadow_rays[index].direction.xyz;
}

fn append_material_eval(ray_index: u32) -> u32 {
    let index = atomicAdd(&queue_counters.material.count, 1u);
    if (index < queue_counters.material.capacity) {
        material_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.material.overflow, 1u);
    }
    return index;
}

fn material_eval_count() -> u32 {
    return atomicLoad(&queue_counters.material.count);
}

fn load_material_eval_ray(index: u32) -> u32 {
    return material_ray_indices[index];
}

fn append_hit_area_light(ray_index: u32) {
    let index = atomicAdd(&queue_counters.hit_area.count, 1u);
    if (index < queue_counters.hit_area.capacity) {
        hit_area_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.hit_area.overflow, 1u);
    }
}

fn hit_area_light_count() -> u32 {
    return atomicLoad(&queue_counters.hit_area.count);
}

fn load_hit_area_ray(index: u32) -> u32 {
    return hit_area_ray_indices[index];
}

fn escaped_ray_count() -> u32 {
    return atomicLoad(&queue_counters.escaped.count);
}

fn append_escaped_ray(ray_index: u32) {
    let index = atomicAdd(&queue_counters.escaped.count, 1u);
    if (index < queue_counters.escaped.capacity) {
        escaped_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.escaped.overflow, 1u);
    }
}

fn load_current_ray(index: u32) -> RayWorkItem {
    return current_rays[index];
}

fn load_next_ray(index: u32) -> RayWorkItem {
    return next_rays[index];
}

fn store_current_ray(index: u32, ray: RayWorkItem) {
    current_rays[index] = ray;
}

fn store_next_ray(index: u32, ray: RayWorkItem) {
    next_rays[index] = ray;
}

fn pixel_count() -> u32 {
    return viewport.width * viewport.height;
}

fn append_direct_eval(ray_index: u32) {
    let index = atomicAdd(&queue_counters.direct.count, 1u);
    if (index < queue_counters.direct.capacity) {
        direct_eval_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.direct.overflow, 1u);
    }
}

fn direct_eval_count() -> u32 {
    return atomicLoad(&queue_counters.direct.count);
}

fn load_direct_eval_ray(index: u32) -> u32 {
    return direct_eval_ray_indices[index];
}

fn append_scatter_diffuse(ray_index: u32) {
    let index = atomicAdd(&queue_counters.scatter_diffuse.count, 1u);
    if (index < queue_counters.scatter_diffuse.capacity) {
        scatter_diffuse_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.scatter_diffuse.overflow, 1u);
    }
}

fn scatter_diffuse_count() -> u32 {
    return atomicLoad(&queue_counters.scatter_diffuse.count);
}

fn load_scatter_diffuse_ray(index: u32) -> u32 {
    return scatter_diffuse_ray_indices[index];
}

fn append_scatter_diffuse_transmission(ray_index: u32) {
    let index = atomicAdd(&queue_counters.scatter_diffuse_transmission.count, 1u);
    if (index < queue_counters.scatter_diffuse_transmission.capacity) {
        scatter_diffuse_transmission_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.scatter_diffuse_transmission.overflow, 1u);
    }
}

fn scatter_diffuse_transmission_count() -> u32 {
    return atomicLoad(&queue_counters.scatter_diffuse_transmission.count);
}

fn load_scatter_diffuse_transmission_ray(index: u32) -> u32 {
    return scatter_diffuse_transmission_ray_indices[index];
}

fn append_scatter_conductor(ray_index: u32) {
    let index = atomicAdd(&queue_counters.scatter_conductor.count, 1u);
    if (index < queue_counters.scatter_conductor.capacity) {
        scatter_conductor_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.scatter_conductor.overflow, 1u);
    }
}

fn scatter_conductor_count() -> u32 {
    return atomicLoad(&queue_counters.scatter_conductor.count);
}

fn load_scatter_conductor_ray(index: u32) -> u32 {
    return scatter_conductor_ray_indices[index];
}

fn append_scatter_dielectric(ray_index: u32) {
    let index = atomicAdd(&queue_counters.scatter_dielectric.count, 1u);
    if (index < queue_counters.scatter_dielectric.capacity) {
        scatter_dielectric_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.scatter_dielectric.overflow, 1u);
    }
}

fn scatter_dielectric_count() -> u32 {
    return atomicLoad(&queue_counters.scatter_dielectric.count);
}

fn load_scatter_dielectric_ray(index: u32) -> u32 {
    return scatter_dielectric_ray_indices[index];
}

fn append_scatter_thin_dielectric(ray_index: u32) {
    let index = atomicAdd(&queue_counters.scatter_thin_dielectric.count, 1u);
    if (index < queue_counters.scatter_thin_dielectric.capacity) {
        scatter_thin_dielectric_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.scatter_thin_dielectric.overflow, 1u);
    }
}

fn scatter_thin_dielectric_count() -> u32 {
    return atomicLoad(&queue_counters.scatter_thin_dielectric.count);
}

fn load_scatter_thin_dielectric_ray(index: u32) -> u32 {
    return scatter_thin_dielectric_ray_indices[index];
}

fn append_scatter_measured(ray_index: u32) {
    let index = atomicAdd(&queue_counters.scatter_measured.count, 1u);
    if (index < queue_counters.scatter_measured.capacity) {
        scatter_measured_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.scatter_measured.overflow, 1u);
    }
}

fn scatter_measured_count() -> u32 {
    return atomicLoad(&queue_counters.scatter_measured.count);
}

fn load_scatter_measured_ray(index: u32) -> u32 {
    return scatter_measured_ray_indices[index];
}

fn append_scatter_coated(ray_index: u32) {
    let index = atomicAdd(&queue_counters.scatter_coated.count, 1u);
    if (index < queue_counters.scatter_coated.capacity) {
        scatter_coated_ray_indices[index] = ray_index;
    } else {
        atomicStore(&queue_counters.scatter_coated.overflow, 1u);
    }
}

fn scatter_coated_count() -> u32 {
    return atomicLoad(&queue_counters.scatter_coated.count);
}

fn load_scatter_coated_ray(index: u32) -> u32 {
    return scatter_coated_ray_indices[index];
}

// Shared tail of every non-specular scatter kind's direct-lighting
// evaluation: MIS weight against the light sample's own pdf, the direct
// contribution, and the shadow ray. `f`, `bsdf_pdf`, and `cosine` are the
// BSDF's own f/pdf/AbsCosTheta at the sampled light direction.
fn add_direct_lighting(
    ray: RayWorkItem,
    surface: SurfaceWorkItem,
    light_sample: DirectLightSample,
    f: vec4<f32>,
    bsdf_pdf: f32,
    cosine: f32,
) {
    let wi = light_sample.direction_pdf.xyz;
    let sampled_light_pdf = light_sample.direction_pdf.w;
    let light_kind = light_sample.light_kind;
    var mis_weight = 1.0;
    if (light_sample.use_mis != 0u) {
        let light_pdf2 = sampled_light_pdf * sampled_light_pdf;
        let bsdf_pdf2 = bsdf_pdf * bsdf_pdf;
        mis_weight = light_pdf2 / max(light_pdf2 + bsdf_pdf2, 1e-7);
    }
    let direct = light_sample.radiance * f * cosine
        / (max(ray.inv_w_u, 1e-7) * sampled_light_pdf)
        * mis_weight;
    // Spawn the shadow ray from the side containing the sampled light.
    let shadow_origin = offset_ray_origin(
        surface.position.xyz,
        surface.position_error.xyz,
        surface.geometric_normal.xyz,
        wi,
    );
    var shadow_direction = wi;
    var shadow_distance = RAY_T_MAX;
    if (light_kind == LIGHT_KIND_AREA) {
        let shadow_target = offset_ray_origin(
            light_sample.position, light_sample.position_error, light_sample.normal, -wi,
        );
        let shadow_vector = shadow_target - shadow_origin;
        shadow_distance = length(shadow_vector);
        shadow_direction = shadow_vector / shadow_distance;
    } else if (light_kind != LIGHT_KIND_DISTANT && !is_infinite_light_kind(light_kind)) {
        let shadow_vector = light_sample.position - shadow_origin;
        shadow_distance = length(shadow_vector);
        shadow_direction = shadow_vector / shadow_distance;
    }
    append_shadow_ray(
        ray.pixel_index,
        shadow_origin,
        shadow_direction,
        shadow_distance,
        (ray.throughput * direct),
    );
}
