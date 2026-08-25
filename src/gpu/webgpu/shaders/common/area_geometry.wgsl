struct AreaLightSample {
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    pdf: f32,
    geometry: Light,
    valid: bool,
};

fn light_source_count() -> u32 {
    if (arrayLength(&lights) == 0u) {
        return 0u;
    }
    return min(lights[0].flags >> 16u, arrayLength(&lights));
}

fn area_light_geometry(source: Light, geometry_index: u32) -> Light {
    return lights[source.primitive + geometry_index];
}

fn area_light_geometry_for_triangle(source: Light, triangle: u32) -> Light {
    for (var index = 0u; index < source.triangle; index += 1u) {
        let geometry = area_light_geometry(source, index);
        if (geometry.triangle == triangle) {
            return geometry;
        }
    }
    return Light(vec4<f32>(0.0), vec4<f32>(0.0), 3u, 0u, 0u, 0u);
}

fn area_light_triangle(geometry: Light) -> array<vec3<f32>, 3> {
    let primitive = primitives[geometry.primitive];
    let index_offset = primitive.first_index + geometry.triangle * 3u;
    let transform_table = transforms[geometry.primitive];
    let p0 = transform(transform_table.render_from_object, vertices[primitive.first_vertex + indices[index_offset]].position).xyz;
    let p1 = transform(transform_table.render_from_object, vertices[primitive.first_vertex + indices[index_offset + 1u]].position).xyz;
    let p2 = transform(transform_table.render_from_object, vertices[primitive.first_vertex + indices[index_offset + 2u]].position).xyz;
    return array<vec3<f32>, 3>(p0, p1, p2);
}

fn area_light_uv(geometry: Light, barycentrics: vec3<f32>) -> vec2<f32> {
    let primitive = primitives[geometry.primitive];
    let index_offset = primitive.first_index + geometry.triangle * 3u;
    let uv0 = vertices[primitive.first_vertex + indices[index_offset]].uv.xy;
    let uv1 = vertices[primitive.first_vertex + indices[index_offset + 1u]].uv.xy;
    let uv2 = vertices[primitive.first_vertex + indices[index_offset + 2u]].uv.xy;
    return uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;
}

fn area_light_normal(geometry: Light, barycentrics: vec3<f32>, triangle: array<vec3<f32>, 3>) -> vec3<f32> {
    let primitive = primitives[geometry.primitive];
    var normal = normalize(cross(triangle[1] - triangle[0], triangle[2] - triangle[0]));
    if ((primitive.flags & 1u) != 0u) {
        normal = -normal;
    }
    let index_offset = primitive.first_index + geometry.triangle * 3u;
    let n0 = vertices[primitive.first_vertex + indices[index_offset]].normal.xyz;
    let n1 = vertices[primitive.first_vertex + indices[index_offset + 1u]].normal.xyz;
    let n2 = vertices[primitive.first_vertex + indices[index_offset + 2u]].normal.xyz;
    var shading_normal = n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z;
    if (dot(shading_normal, shading_normal) > 1.0e-20) {
        let orientation_sign = select(1.0, -1.0, (primitive.flags & 1u) != 0u);
        shading_normal = normalize(transform(transforms[geometry.primitive].normal_from_object, vec4<f32>(orientation_sign * shading_normal, 0.0)).xyz);
        if (dot(normal, shading_normal) < 0.0) {
            normal = -normal;
        }
    }
    return normal;
}

fn angle_between(a: vec3<f32>, b: vec3<f32>) -> f32 {
    if (dot(a, b) < 0.0) {
        return 3.141592653589793 - 2.0 * asin(clamp(0.5 * length(a + b), 0.0, 1.0));
    }
    return 2.0 * asin(clamp(0.5 * length(b - a), 0.0, 1.0));
}

fn spherical_triangle_area(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    return abs(2.0 * atan2(
        dot(a, cross(b, c)),
        1.0 + dot(a, b) + dot(a, c) + dot(b, c),
    ));
}

fn sample_linear(u: f32, a: f32, b: f32) -> f32 {
    if (u == 0.0 && a == 0.0) {
        return 0.0;
    }
    let value = u * (a + b) / (a + sqrt(mix(a * a, b * b, u)));
    return min(value, 0.9999999403953552);
}

fn sample_bilinear(u: vec2<f32>, weights: vec4<f32>) -> vec2<f32> {
    let y = sample_linear(u.y, weights.x + weights.y, weights.z + weights.w);
    let x = sample_linear(
        u.x,
        mix(weights.x, weights.z, y),
        mix(weights.y, weights.w, y),
    );
    return vec2<f32>(x, y);
}

fn bilinear_pdf(p: vec2<f32>, weights: vec4<f32>) -> f32 {
    let sum = weights.x + weights.y + weights.z + weights.w;
    if (sum == 0.0) {
        return 1.0;
    }
    return 4.0 * (
        (1.0 - p.x) * (1.0 - p.y) * weights.x
        + p.x * (1.0 - p.y) * weights.y
        + (1.0 - p.x) * p.y * weights.z
        + p.x * p.y * weights.w
    ) / sum;
}

struct SphericalTriangleSample {
    barycentrics: vec3<f32>,
    pdf: f32,
    valid: bool,
};
