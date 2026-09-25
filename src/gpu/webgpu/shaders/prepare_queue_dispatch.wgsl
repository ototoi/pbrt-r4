// Rewrites every indirect-dispatch argument slot from the queues' current
// counts. Called several times per depth, each time a batch of queues'
// counts becomes final for this depth (after intersect_primary_rays, after
// shade_surface, after classify_surface_scatter, after the scatter stages,
// and after the ray queue swap). Each call recomputes every slot; a slot
// whose queue has not been (re)populated since the last reset simply reads
// its already-zero count again, which is harmless.
fn write_dispatch_args(slot: u32, count: u32, capacity: u32) {
    let clamped = min(count, capacity);
    let n = (clamped + INDIRECT_WORKGROUP_SIZE - 1u) / INDIRECT_WORKGROUP_SIZE;
    var x = n;
    var y = 1u;
    if (n > INDIRECT_MAX_WORKGROUPS_X) {
        x = INDIRECT_MAX_WORKGROUPS_X;
        y = (n + INDIRECT_MAX_WORKGROUPS_X - 1u) / INDIRECT_MAX_WORKGROUPS_X;
    }
    queue_dispatch_args[slot] = DispatchIndirectArgs(x, y, 1u);
}

@compute @workgroup_size(1)
fn prepare_queue_dispatch() {
    write_dispatch_args(
        QUEUE_DISPATCH_SLOT_MATERIAL_EVAL, material_eval_count(), queue_counters.material.capacity,
    );
    write_dispatch_args(
        QUEUE_DISPATCH_SLOT_DIRECT_EVAL, direct_eval_count(), queue_counters.direct.capacity,
    );
    write_dispatch_args(
        QUEUE_DISPATCH_SLOT_SCATTER_DIFFUSE,
        scatter_diffuse_count(),
        queue_counters.scatter_diffuse.capacity,
    );
    write_dispatch_args(
        QUEUE_DISPATCH_SLOT_SCATTER_DIFFUSE_TRANSMISSION,
        scatter_diffuse_transmission_count(),
        queue_counters.scatter_diffuse_transmission.capacity,
    );
    write_dispatch_args(
        QUEUE_DISPATCH_SLOT_SCATTER_CONDUCTOR,
        scatter_conductor_count(),
        queue_counters.scatter_conductor.capacity,
    );
    write_dispatch_args(
        QUEUE_DISPATCH_SLOT_SCATTER_DIELECTRIC,
        scatter_dielectric_count(),
        queue_counters.scatter_dielectric.capacity,
    );
    write_dispatch_args(
        QUEUE_DISPATCH_SLOT_SCATTER_THIN_DIELECTRIC,
        scatter_thin_dielectric_count(),
        queue_counters.scatter_thin_dielectric.capacity,
    );
    write_dispatch_args(
        QUEUE_DISPATCH_SLOT_SCATTER_MEASURED,
        scatter_measured_count(),
        queue_counters.scatter_measured.capacity,
    );
    write_dispatch_args(
        QUEUE_DISPATCH_SLOT_SCATTER_COATED,
        scatter_coated_count(),
        queue_counters.scatter_coated.capacity,
    );
    write_dispatch_args(
        QUEUE_DISPATCH_SLOT_CURRENT_RAY, current_ray_count(), queue_counters.current.capacity,
    );
    write_dispatch_args(
        QUEUE_DISPATCH_SLOT_ESCAPED, escaped_ray_count(), queue_counters.escaped.capacity,
    );
    write_dispatch_args(
        QUEUE_DISPATCH_SLOT_HIT_AREA, hit_area_light_count(), queue_counters.hit_area.capacity,
    );
    write_dispatch_args(
        QUEUE_DISPATCH_SLOT_SHADOW, shadow_ray_count(), queue_counters.shadow.capacity,
    );
    write_dispatch_args(
        QUEUE_DISPATCH_SLOT_NEXT_RAY, next_ray_count(), queue_counters.next.capacity,
    );
}
