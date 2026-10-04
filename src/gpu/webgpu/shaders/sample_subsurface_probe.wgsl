fn bssrdf_segment_seed(segment: BSSRDFSegment) -> u32 {
    var h = 0u;
    for (var i = 0u; i < 3u; i++) {
        h = hash_u32(h ^ bitcast<u32>(segment.start[i]));
        h = hash_u32(h ^ bitcast<u32>(segment.end[i]));
    }
    return h;
}
@compute @workgroup_size(64)
fn sample_subsurface_probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.y * INDIRECT_ROW_ITEMS + id.x;
    if (index >= atomicLoad(&bssrdf_work.state.count)) { return; }
    if (index >= bssrdf_work.state.capacity || index >= arrayLength(&bssrdf_work.items)
        || atomicLoad(&bssrdf_work.state.overflow) != 0u) { set_render_error(); return; }
    if (index >= arrayLength(&bssrdf_results)) { set_render_error(); return; }
    var result: BSSRDFProbeResult;
    bssrdf_results[index] = result;
    let work = bssrdf_work.items[index];
    let segment = bssrdf_sample_segment(work);
    if (segment.valid == 0u) { return; }
    result.segment_valid = 1u;
    result.start = vec4<f32>(segment.start, 1.0);
    result.end = vec4<f32>(segment.end, 1.0);
    var origin = segment.start;
    var seed = bssrdf_segment_seed(segment);
    loop {
        let direction = segment.end - origin;
        if (all(direction == vec3<f32>(0.0))) { break; }
        var query: ray_query;
        // SpawnRayTo uses a segment parameterization and excludes the endpoint.
        rayQueryInitialize(&query, tlas, RayDesc(0u, 0xffu, 0.0, 1.0 - 0.0001, origin, direction));
        while (rayQueryProceed(&query)) {}
        let hit = rayQueryGetCommittedIntersection(&query);
        if (hit.kind == RAY_QUERY_INTERSECTION_NONE) { break; }
        let bary = vec3<f32>(1.0 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics);
        let surface = reconstruct_triangle_surface(hit.instance_custom_data, hit.primitive_index, bary);
        if (surface.valid == 0u) { break; }
        if (instances[hit.instance_custom_data].material_root == work.material_root) {
            result.candidate_count += 1u;
            seed = hash_u32(seed + 0x9e3779b9u);
            let u = f32(seed & 0x00ffffffu) / 16777216.0;
            if (u < 1.0 / f32(result.candidate_count)) {
                result.position = vec4<f32>(surface.position, 1.0);
                result.position_error = vec4<f32>(surface.position_error, 0.0);
                result.normal = vec4<f32>(surface.geometric_normal, 0.0);
                result.barycentric = vec4<f32>(bary, 0.0);
                result.instance_index = hit.instance_custom_data;
                result.primitive_index = hit.primitive_index;
                result.valid = 1u;
            }
        }
        let next_origin = offset_ray_origin(surface.position, surface.position_error, surface.geometric_normal, segment.end - surface.position);
        if (dot(segment.end - next_origin, direction) <= 0.0) { break; }
        if (all(next_origin == origin)) { set_render_error(); break; }
        origin = next_origin;
    }
    if (result.valid != 0u) { result.reservoir_probability = 1.0 / f32(result.candidate_count); }
    bssrdf_results[index] = result;
}
