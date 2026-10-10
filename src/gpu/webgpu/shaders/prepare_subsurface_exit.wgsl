@compute @workgroup_size(64)
fn prepare_subsurface_exit(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.y * INDIRECT_ROW_ITEMS + id.x;
    if (index >= atomicLoad(&bssrdf_work.state.count)) { return; }
    let work = bssrdf_work.items[index];
    var result = bssrdf_results[index];
    if (result.valid == 0u || result.reservoir_probability == 0.0) { return; }
    let sp = tabulated_bssrdf_sr(work, distance(result.position.xyz, work.position.xyz), false);
    let pdf = tabulated_bssrdf_pdf_sp(work, result.position.xyz, result.normal.xyz);
    if (pdf.x <= 0.0 || !any(sp > vec4<f32>(0.0))) { bssrdf_results[index].valid = 0u; return; }
    var ray = load_current_ray(work.ray_index);
    ray.beta *= sp / (result.reservoir_probability * pdf.x);
    ray.r_u *= pdf / pdf.x;
    let frame = reconstruct_triangle_shading_frame(result.instance_index, result.primitive_index,
        result.barycentric.xyz, result.normal.xyz);
    ray.direction = vec4<f32>(-frame[2], 0.0);
    store_current_ray(work.ray_index, ray);
    var surface = surfaces[work.pixel_index];
    surface.position = result.position;
    surface.position_error = result.position_error;
    surface.geometric_normal = result.normal;
    surface.normal = vec4<f32>(frame[2], 0.0);
    surface.tangent = vec4<f32>(frame[0], 0.0);
    surface.instance_custom_data = result.instance_index;
    surface.primitive_index = result.primitive_index;
    surface.flags = SURFACE_FLAG_SUBSURFACE_EXIT;
    surfaces[work.pixel_index] = surface;
    direct_light_samples[work.pixel_index].valid = 0u;
    if (light_table.light_count != 0u) { append_direct_eval(work.ray_index); }
}
