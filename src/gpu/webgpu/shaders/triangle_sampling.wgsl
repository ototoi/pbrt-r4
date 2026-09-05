struct TriangleVertices {
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
};

fn load_area_triangle(area_index: u32, primitive: u32) -> TriangleVertices {
    let instance = instances[load_area_instance(area_index)];
    let geometry = geometries[instance.geometry];
    let first_index = geometry.index_offset + primitive * 3u;
    let i0 = geometry.vertex_offset + indices[first_index];
    let i1 = geometry.vertex_offset + indices[first_index + 1u];
    let i2 = geometry.vertex_offset + indices[first_index + 2u];
    return TriangleVertices(
        instance.world_from_object * vertices[i0].position,
        instance.world_from_object * vertices[i1].position,
        instance.world_from_object * vertices[i2].position,
    );
}

fn triangle_geometric_normal(triangle: TriangleVertices) -> vec3<f32> {
    return normalize(cross(
        triangle.p1.xyz - triangle.p0.xyz,
        triangle.p2.xyz - triangle.p0.xyz,
    ));
}

fn sample_uniform_triangle(u: vec2<f32>) -> vec3<f32> {
    var b0: f32;
    var b1: f32;
    if (u.x < u.y) {
        b0 = u.x * 0.5;
        b1 = u.y - b0;
    } else {
        b1 = u.y * 0.5;
        b0 = u.x - b1;
    }
    return vec3<f32>(b0, b1, 1.0 - b0 - b1);
}

// xyz contains barycentrics and w contains the solid-angle PDF.
fn sample_uniform_triangle_for_context(
    triangle: TriangleVertices,
    p: vec3<f32>,
    u: vec2<f32>,
    triangle_area: f32,
) -> vec4<f32> {
    let b = sample_uniform_triangle(u);
    let sampled_point = triangle.p0.xyz * b.x + triangle.p1.xyz * b.y + triangle.p2.xyz * b.z;
    let to_light = sampled_point - p;
    let distance_squared = dot(to_light, to_light);
    if (distance_squared == 0.0) {
        return vec4<f32>(0.0);
    }
    let cosine = abs(dot(triangle_geometric_normal(triangle), -normalize(to_light)));
    if (cosine == 0.0 || triangle_area <= 0.0) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(b, distance_squared / (cosine * triangle_area));
}

fn uniform_triangle_pdf_for_context(
    triangle: TriangleVertices,
    p: vec3<f32>,
    light_normal: vec3<f32>,
    wi: vec3<f32>,
    sampled_point: vec3<f32>,
    triangle_area: f32,
) -> f32 {
    let to_light = sampled_point - p;
    let distance_squared = dot(to_light, to_light);
    if (distance_squared == 0.0 || triangle_area <= 0.0) {
        return 0.0;
    }
    let cosine = abs(dot(light_normal, -normalize(wi)));
    if (cosine == 0.0) {
        return 0.0;
    }
    return distance_squared / (cosine * triangle_area);
}
