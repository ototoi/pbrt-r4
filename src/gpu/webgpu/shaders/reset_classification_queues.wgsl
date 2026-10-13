@compute @workgroup_size(1)
fn reset_classification_queues() {
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
