@compute @workgroup_size(8, 8, 1)
fn reset_classification_queues(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u && global_id.y == 0u) {
        atomicStore(&queue_counters.material.count, 0u);
        atomicStore(&queue_counters.material.overflow, 0u);
        atomicStore(&queue_counters.hit_area.count, 0u);
        atomicStore(&queue_counters.hit_area.overflow, 0u);
        atomicStore(&queue_counters.escaped.count, 0u);
        atomicStore(&queue_counters.escaped.overflow, 0u);
        atomicStore(&queue_counters.direct.count, 0u);
        atomicStore(&queue_counters.direct.overflow, 0u);
        atomicStore(&queue_counters.scatter_diffuse.count, 0u);
        atomicStore(&queue_counters.scatter_diffuse.overflow, 0u);
        atomicStore(&queue_counters.scatter_diffuse_transmission.count, 0u);
        atomicStore(&queue_counters.scatter_diffuse_transmission.overflow, 0u);
        atomicStore(&queue_counters.scatter_conductor.count, 0u);
        atomicStore(&queue_counters.scatter_conductor.overflow, 0u);
        atomicStore(&queue_counters.scatter_dielectric.count, 0u);
        atomicStore(&queue_counters.scatter_dielectric.overflow, 0u);
        atomicStore(&queue_counters.scatter_thin_dielectric.count, 0u);
        atomicStore(&queue_counters.scatter_thin_dielectric.overflow, 0u);
        atomicStore(&queue_counters.scatter_measured.count, 0u);
        atomicStore(&queue_counters.scatter_measured.overflow, 0u);
        atomicStore(&queue_counters.scatter_coated.count, 0u);
        atomicStore(&queue_counters.scatter_coated.overflow, 0u);
    }
}
