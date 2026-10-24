fn instance_orientation_is_reversed(flags: u32) -> bool {
    return (flags & INSTANCE_ORIENTATION_FLAG_REVERSED) != 0u;
}

fn instance_orientation_swaps_handedness(flags: u32) -> bool {
    return (flags & INSTANCE_ORIENTATION_FLAG_TRANSFORM_SWAPS_HANDEDNESS) != 0u;
}

fn instance_orientation_flips_geometric_normal(flags: u32) -> bool {
    return instance_orientation_is_reversed(flags)
        != instance_orientation_swaps_handedness(flags);
}

// Quadric intersections flip in object space before the normal transform;
// triangle cross products already include the transform's handedness.
fn intersection_flips_geometric_normal(kind: u32, flags: u32) -> bool {
    if (kind == INTERSECTION_NORMAL_KIND_QUADRIC) {
        return instance_orientation_flips_geometric_normal(flags)
            != instance_orientation_shape_swaps_handedness(flags);
    }
    return instance_orientation_flips_geometric_normal(flags);
}

fn instance_orientation_shape_swaps_handedness(flags: u32) -> bool {
    return (flags & INSTANCE_ORIENTATION_FLAG_SHAPE_TRANSFORM_SWAPS_HANDEDNESS) != 0u;
}

fn intersection_flips_vertex_normal(kind: u32, flags: u32) -> bool {
    return kind == INTERSECTION_NORMAL_KIND_QUADRIC
        && instance_orientation_shape_swaps_handedness(flags);
}

// v4 Triangle::InteractionFromIntersection makes vertex shading normals authoritative.
fn triangle_intersection_geometric_normal(
    instance: Instance,
    geometry: Geometry,
    object_normal: vec3<f32>,
    winding_normal: vec3<f32>,
) -> vec3<f32> {
    var normal = winding_normal;
    if (intersection_flips_geometric_normal(geometry.intersection_normal_kind, instance.orientation_flags)) {
        normal = -normal;
    }
    if (geometry.intersection_normal_kind == INTERSECTION_NORMAL_KIND_TRIANGLE) {
        let shading_normal = (instance.normal_from_object * vec4<f32>(object_normal, 0.0)).xyz;
        if (dot(normal, shading_normal) < 0.0) {
            normal = -normal;
        }
    }
    return normal;
}

fn reconstruct_triangle_surface(
    instance_index: u32,
    primitive: u32,
    barycentrics: vec3<f32>,
) -> TriangleSurfaceData {
    let invalid_surface = TriangleSurfaceData(
        vec3<f32>(0.0), vec3<f32>(0.0), vec2<f32>(0.0), vec3<f32>(0.0), 0u,
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
    if (first_index + 2u >= arrayLength(&indices[geometry.page].values)) {
        set_render_error();
        return invalid_surface;
    }
    let i0 = geometry.vertex_offset + indices[geometry.page].values[first_index];
    let i1 = geometry.vertex_offset + indices[geometry.page].values[first_index + 1u];
    let i2 = geometry.vertex_offset + indices[geometry.page].values[first_index + 2u];
    if (i0 >= arrayLength(&vertices[geometry.page].values) || i1 >= arrayLength(&vertices[geometry.page].values) || i2 >= arrayLength(&vertices[geometry.page].values)) {
        set_render_error();
        return invalid_surface;
    }
    let p0 = (instance.world_from_object * vertices[geometry.page].values[i0].position).xyz;
    let p1 = (instance.world_from_object * vertices[geometry.page].values[i1].position).xyz;
    let p2 = (instance.world_from_object * vertices[geometry.page].values[i2].position).xyz;
    let b0 = barycentrics.x;
    let b1 = barycentrics.y;
    let b2 = barycentrics.z;
    let position = p0 * b0 + p1 * b1 + p2 * b2;
    let position_error = (abs(p0 * b0) + abs(p1 * b1) + abs(p2 * b2)) * gamma(7.0)
        + hardware_intersection_error(p0, p1, p2);
    let uv = vertices[geometry.page].values[i0].uv * b0 + vertices[geometry.page].values[i1].uv * b1 + vertices[geometry.page].values[i2].uv * b2;
    let object_normal = vertices[geometry.page].values[i0].normal.xyz * b0
        + vertices[geometry.page].values[i1].normal.xyz * b1
        + vertices[geometry.page].values[i2].normal.xyz * b2;
    let geometric_normal = triangle_intersection_geometric_normal(
        instance, geometry, object_normal, normalize(cross(p1 - p0, p2 - p0)),
    );
    return TriangleSurfaceData(position, position_error, uv, geometric_normal, 1u);
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

fn triangle_shading_normal(instance: Instance, geometry: Geometry, object_normal: vec3<f32>, geometric_normal: vec3<f32>) -> vec3<f32> {
    if (dot(object_normal, object_normal) == 0.0) { return geometric_normal; }
    var normal = (instance.normal_from_object * vec4<f32>(object_normal, 0.0)).xyz;
    if (intersection_flips_vertex_normal(geometry.intersection_normal_kind, instance.orientation_flags)) { normal = -normal; }
    return normalize(normal);
}
fn reconstruct_triangle_shading_frame(instance_index: u32, primitive: u32, bary: vec3<f32>, geometric_normal: vec3<f32>) -> mat3x3<f32> {
    let instance = instances[instance_index];
    let geometry = geometries[instance.geometry];
    let first = geometry.index_offset + primitive * 3u;
    let v0 = vertices[geometry.page].values[geometry.vertex_offset + indices[geometry.page].values[first]];
    let v1 = vertices[geometry.page].values[geometry.vertex_offset + indices[geometry.page].values[first + 1u]];
    let v2 = vertices[geometry.page].values[geometry.vertex_offset + indices[geometry.page].values[first + 2u]];
    let object_normal = v0.normal.xyz*bary.x + v1.normal.xyz*bary.y + v2.normal.xyz*bary.z;
    let normal = triangle_shading_normal(instance, geometry, object_normal, geometric_normal);
    var source = v0.tangent.xyz*bary.x + v1.tangent.xyz*bary.y + v2.tangent.xyz*bary.z;
    if (dot(source, source) == 0.0) {
        let duv02 = v0.uv - v2.uv;
        let duv12 = v1.uv - v2.uv;
        let det = duv02.x*duv12.y - duv02.y*duv12.x;
        if (det != 0.0) { source = (duv12.y*(v0.position.xyz-v2.position.xyz) - duv02.y*(v1.position.xyz-v2.position.xyz)) / det; }
    }
    let tangent = (instance.world_from_object * vec4<f32>(source, 0.0)).xyz;
    let bitangent = cross(normal, tangent);
    var x = coordinate_system_x(normal);
    if (dot(bitangent, bitangent) > 0.0) { x = normalize(cross(bitangent, normal)); }
    return mat3x3<f32>(x, cross(normal, x), normal);
}
