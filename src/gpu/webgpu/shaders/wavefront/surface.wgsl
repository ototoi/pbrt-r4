const MACHINE_EPSILON: f32 = 5.960464477539063e-8;
const TRIANGLE_INTERSECTION_GAMMA: f32 =
    (7.0 * MACHINE_EPSILON) / (1.0 - 7.0 * MACHINE_EPSILON);
const TRANSFORM_GAMMA: f32 =
    (3.0 * MACHINE_EPSILON) / (1.0 - 3.0 * MACHINE_EPSILON);

fn transform_position_error(
    transform: mat4x4<f32>,
    position: vec3<f32>,
    position_error: vec3<f32>,
) -> vec3<f32> {
    let direct_error = TRANSFORM_GAMMA * vec3<f32>(
        abs(transform[0].x * position.x) + abs(transform[1].x * position.y) +
            abs(transform[2].x * position.z) + abs(transform[3].x),
        abs(transform[0].y * position.x) + abs(transform[1].y * position.y) +
            abs(transform[2].y * position.z) + abs(transform[3].y),
        abs(transform[0].z * position.x) + abs(transform[1].z * position.y) +
            abs(transform[2].z * position.z) + abs(transform[3].z),
    );
    let propagated_error = (TRANSFORM_GAMMA + 1.0) * vec3<f32>(
        abs(transform[0].x) * position_error.x + abs(transform[1].x) * position_error.y +
            abs(transform[2].x) * position_error.z,
        abs(transform[0].y) * position_error.x + abs(transform[1].y) * position_error.y +
            abs(transform[2].y) * position_error.z,
        abs(transform[0].z) * position_error.x + abs(transform[1].z) * position_error.y +
            abs(transform[2].z) * position_error.z,
    );
    return direct_error + propagated_error;
}

@compute @workgroup_size(64)
fn evaluate_surface_interaction(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slot_index = global_id.x;
    if (slot_index >= pixel_count() || slot_index >= arena.capacity) {
        return;
    }

    let ray = arena.rays[slot_index];
    if (ray.indices.y != RAY_STATE_HIT) {
        return;
    }

    let primitive_id = ray.indices.z;
    let primitive = primitives[primitive_id];
    let index_offset = primitive.first_index + 3u * ray.indices.w;
    let index0 = indices[index_offset];
    let index1 = indices[index_offset + 1u];
    let index2 = indices[index_offset + 2u];
    let object_p0 = vertices[primitive.first_vertex + index0].position.xyz;
    let object_p1 = vertices[primitive.first_vertex + index1].position.xyz;
    let object_p2 = vertices[primitive.first_vertex + index2].position.xyz;
    let transform = transforms[primitive_id].render_from_object;
    let p0 = (transform * vec4<f32>(object_p0, 1.0)).xyz;
    let p1 = (transform * vec4<f32>(object_p1, 1.0)).xyz;
    let p2 = (transform * vec4<f32>(object_p2, 1.0)).xyz;
    let barycentrics = vec3<f32>(1.0 - ray.hit.y - ray.hit.z, ray.hit.y, ray.hit.z);
    let position = barycentrics.x * p0 + barycentrics.y * p1 + barycentrics.z * p2;
    let object_position =
        barycentrics.x * object_p0 + barycentrics.y * object_p1 + barycentrics.z * object_p2;
    let object_error = TRIANGLE_INTERSECTION_GAMMA * (
        abs(barycentrics.x * object_p0) + abs(barycentrics.y * object_p1) +
            abs(barycentrics.z * object_p2)
    );
    var geometric_normal = normalize(cross(p1 - p0, p2 - p0));
    if (primitive.flags & 1u) != 0u {
        geometric_normal = -geometric_normal;
    }

    arena.rays[slot_index].surface_position = vec4<f32>(position, 1.0);
    arena.rays[slot_index].surface_normal = vec4<f32>(geometric_normal, 0.0);
    arena.rays[slot_index].surface_error =
        vec4<f32>(transform_position_error(transform, object_position, object_error), 0.0);
    let uv0 = vertices[primitive.first_vertex + index0].uv.xy;
    let uv1 = vertices[primitive.first_vertex + index1].uv.xy;
    let uv2 = vertices[primitive.first_vertex + index2].uv.xy;
    arena.rays[slot_index].surface_uv = vec4<f32>(
        barycentrics.x * uv0 + barycentrics.y * uv1 + barycentrics.z * uv2,
        0.0,
        0.0,
    );
    arena.rays[slot_index].indices.y = RAY_STATE_SURFACE;
}
