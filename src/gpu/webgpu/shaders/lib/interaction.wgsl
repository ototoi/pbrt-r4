fn instance_orientation_is_reversed(flags: u32) -> bool {
    return (flags & INSTANCE_ORIENTATION_FLAG_REVERSED) != 0u;
}

fn instance_orientation_swaps_handedness(flags: u32) -> bool {
    return (flags & INSTANCE_ORIENTATION_FLAG_TRANSFORM_SWAPS_HANDEDNESS) != 0u;
}

fn reconstruct_triangle_surface(
    instance_index: u32,
    primitive: u32,
    barycentrics: vec3<f32>,
) -> TriangleSurfaceData {
    let invalid_surface = TriangleSurfaceData(
        vec3<f32>(0.0), vec2<f32>(0.0), vec3<f32>(0.0), 0u,
    );
    if (instance_index >= arrayLength(&instances)) {
        set_render_error();
        return invalid_surface;
    }
    let instance = instances[instance_index];
    if (instance.geometry >= arrayLength(&geometries)) {
        set_render_error();
        return invalid_surface;
    }
    let geometry = geometries[instance.geometry];
    let first_index = geometry.index_offset + primitive * 3u;
    if (first_index + 2u >= arrayLength(&indices)) {
        set_render_error();
        return invalid_surface;
    }
    let i0 = geometry.vertex_offset + indices[first_index];
    let i1 = geometry.vertex_offset + indices[first_index + 1u];
    let i2 = geometry.vertex_offset + indices[first_index + 2u];
    if (i0 >= arrayLength(&vertices) || i1 >= arrayLength(&vertices) || i2 >= arrayLength(&vertices)) {
        set_render_error();
        return invalid_surface;
    }
    let p0 = (instance.world_from_object * vertices[i0].position).xyz;
    let p1 = (instance.world_from_object * vertices[i1].position).xyz;
    let p2 = (instance.world_from_object * vertices[i2].position).xyz;
    let b0 = barycentrics.x;
    let b1 = barycentrics.y;
    let b2 = barycentrics.z;
    let position = p0 * b0 + p1 * b1 + p2 * b2;
    let uv = vertices[i0].uv * b0 + vertices[i1].uv * b1 + vertices[i2].uv * b2;
    var geometric_normal = normalize(cross(p1 - p0, p2 - p0));
    if (instance_orientation_is_reversed(instance.orientation_flags)) {
        geometric_normal = -geometric_normal;
    }
    return TriangleSurfaceData(position, uv, geometric_normal, 1u);
}

fn offset_ray_origin(position: vec3<f32>, error: vec3<f32>, normal: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    let offset = normal * dot(abs(normal), error);
    let signed_offset = select(-offset, offset, dot(direction, normal) >= 0.0);
    var result = position + signed_offset;
    if (signed_offset.x > 0.0) {
        result.x = next_float_up(result.x);
    } else if (signed_offset.x < 0.0) {
        result.x = next_float_down(result.x);
    }
    if (signed_offset.y > 0.0) {
        result.y = next_float_up(result.y);
    } else if (signed_offset.y < 0.0) {
        result.y = next_float_down(result.y);
    }
    if (signed_offset.z > 0.0) {
        result.z = next_float_up(result.z);
    } else if (signed_offset.z < 0.0) {
        result.z = next_float_down(result.z);
    }
    return result;
}
