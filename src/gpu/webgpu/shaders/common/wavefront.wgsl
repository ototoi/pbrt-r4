const WAVEFRONT_STATE_INACTIVE: u32 = 0u;
const WAVEFRONT_STATE_RAY: u32 = 1u;
const WAVEFRONT_STATE_MISS: u32 = 2u;
const WAVEFRONT_STATE_HIT: u32 = 3u;
const WAVEFRONT_STATE_EMISSIVE: u32 = 4u;

struct WavefrontSlot {
    ray_origin: vec4<f32>,
    ray_direction: vec4<f32>,
    intersection_ids: vec4<u32>,
    intersection_data: vec4<f32>,
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
    contribution: vec4<f32>,
    radiance: vec4<f32>,
    throughput: vec4<f32>,
    path_info: vec4<u32>,
    surface_position: vec4<f32>,
    surface_position_error: vec4<f32>,
    surface_geometric_normal: vec4<f32>,
    surface_shading_normal: vec4<f32>,
    surface_uv: vec4<f32>,
    previous_surface_position: vec4<f32>,
    previous_surface_shading_normal: vec4<f32>,
    previous_bsdf_pdf: vec4<f32>,
    material_parameters: vec4<f32>,
};

struct WavefrontControl {
    sample_index: u32,
    active_count: atomic<u32>,
    next_count: atomic<u32>,
    overflow: atomic<u32>,
};

struct WavefrontQueueHeader {
    count: atomic<u32>,
    capacity: u32,
    offset: u32,
    overflow: atomic<u32>,
};

struct WavefrontArena {
    control: WavefrontControl,
    ray_queue_a: WavefrontQueueHeader,
    ray_queue_b: WavefrontQueueHeader,
    escaped_ray_queue: WavefrontQueueHeader,
    hit_area_light_queue: WavefrontQueueHeader,
    surface_queue: WavefrontQueueHeader,
    shadow_queue: WavefrontQueueHeader,
    slots: array<WavefrontSlot>,
};

@group(0) @binding(10)
var<storage, read_write> wavefront: WavefrontArena;

// The index arena shares binding 1 with the film buffer. Each queue index uses
// one vec4 slot. Ray, classification, and shadow queues use disjoint regions
// so their work items can coexist between stage dispatches. The regions are
// reused after prepare_next_bounce resets the classification and shadow queues.
fn wavefront_queue_index_base() -> u32 {
    return u32(camera.viewport.z * camera.viewport.w);
}

fn wavefront_queue_index(queue_offset: u32) -> u32 {
    return bitcast<vec4<u32>>(output[wavefront_queue_index_base() + queue_offset]).x;
}

fn wavefront_store_queue_index(queue_offset: u32, slot_index: u32) {
    output[wavefront_queue_index_base() + queue_offset] = vec4<f32>(
        bitcast<f32>(slot_index),
        0.0,
        0.0,
        0.0,
    );
}

fn wavefront_shadow_queue_index(queue_index: u32) -> u32 {
    return wavefront_queue_index(wavefront.shadow_queue.offset + queue_index);
}

fn wavefront_escaped_ray_queue_index(queue_index: u32) -> u32 {
    return wavefront_queue_index(wavefront.escaped_ray_queue.offset + queue_index);
}

fn wavefront_hit_area_light_queue_index(queue_index: u32) -> u32 {
    return wavefront_queue_index(wavefront.hit_area_light_queue.offset + queue_index);
}

fn wavefront_surface_queue_index(queue_index: u32) -> u32 {
    return wavefront_queue_index(wavefront.surface_queue.offset + queue_index);
}

fn wavefront_enqueue_escaped_ray(slot_index: u32) {
    let queue_index = atomicAdd(&wavefront.escaped_ray_queue.count, 1u);
    if (queue_index >= wavefront.escaped_ray_queue.capacity) {
        atomicStore(&wavefront.escaped_ray_queue.overflow, 1u);
        atomicStore(&wavefront.control.overflow, 1u);
    } else {
        wavefront_store_queue_index(wavefront.escaped_ray_queue.offset + queue_index, slot_index);
    }
}

fn wavefront_enqueue_hit_area_light(slot_index: u32) {
    let queue_index = atomicAdd(&wavefront.hit_area_light_queue.count, 1u);
    if (queue_index >= wavefront.hit_area_light_queue.capacity) {
        atomicStore(&wavefront.hit_area_light_queue.overflow, 1u);
        atomicStore(&wavefront.control.overflow, 1u);
    } else {
        wavefront_store_queue_index(wavefront.hit_area_light_queue.offset + queue_index, slot_index);
    }
}

fn wavefront_enqueue_surface(slot_index: u32) {
    let queue_index = atomicAdd(&wavefront.surface_queue.count, 1u);
    if (queue_index >= wavefront.surface_queue.capacity) {
        atomicStore(&wavefront.surface_queue.overflow, 1u);
        atomicStore(&wavefront.control.overflow, 1u);
    } else {
        wavefront_store_queue_index(wavefront.surface_queue.offset + queue_index, slot_index);
    }
}

fn wavefront_enqueue_shadow(slot_index: u32) {
    let queue_index = atomicAdd(&wavefront.shadow_queue.count, 1u);
    if (queue_index >= wavefront.shadow_queue.capacity) {
        atomicStore(&wavefront.shadow_queue.overflow, 1u);
        atomicStore(&wavefront.control.overflow, 1u);
    } else {
        wavefront_store_queue_index(wavefront.shadow_queue.offset + queue_index, slot_index);
    }
}
