fn sample_spherical_triangle(
    triangle: array<vec3<f32>, 3>,
    context_position: vec3<f32>,
    u: vec2<f32>,
) -> SphericalTriangleSample {
    var a = triangle[0] - context_position;
    var b = triangle[1] - context_position;
    var c = triangle[2] - context_position;
    if (dot(a, a) == 0.0 || dot(b, b) == 0.0 || dot(c, c) == 0.0) {
        return SphericalTriangleSample(vec3<f32>(0.0), 0.0, false);
    }
    a = normalize(a);
    b = normalize(b);
    c = normalize(c);
    var n_ab = cross(a, b);
    var n_bc = cross(b, c);
    var n_ca = cross(c, a);
    if (dot(n_ab, n_ab) == 0.0 || dot(n_bc, n_bc) == 0.0 || dot(n_ca, n_ca) == 0.0) {
        return SphericalTriangleSample(vec3<f32>(0.0), 0.0, false);
    }
    n_ab = normalize(n_ab);
    n_bc = normalize(n_bc);
    n_ca = normalize(n_ca);
    let alpha = angle_between(n_ab, -n_ca);
    let beta = angle_between(n_bc, -n_ab);
    let gamma = angle_between(n_ca, -n_bc);
    let area_pi = alpha + beta + gamma;
    let area = area_pi - 3.141592653589793;
    if (area <= 0.0) {
        return SphericalTriangleSample(vec3<f32>(0.0), 0.0, false);
    }
    let sampled_area_pi = mix(3.141592653589793, area_pi, u.x);
    let cos_alpha = cos(alpha);
    let sin_alpha = sin(alpha);
    let sin_phi = sin(sampled_area_pi) * cos_alpha - cos(sampled_area_pi) * sin_alpha;
    let cos_phi = cos(sampled_area_pi) * cos_alpha + sin(sampled_area_pi) * sin_alpha;
    let k1 = cos_phi + cos_alpha;
    let k2 = sin_phi - sin_alpha * dot(a, b);
    let numerator = k2 + fma(k2, cos_phi, -k1 * sin_phi) * cos_alpha;
    let denominator = fma(k2, sin_phi, k1 * cos_phi) * sin_alpha;
    if (denominator == 0.0) {
        return SphericalTriangleSample(vec3<f32>(0.0), 0.0, false);
    }
    let cos_beta_prime = clamp(numerator / denominator, -1.0, 1.0);
    let sin_beta_prime = sqrt(max(0.0, 1.0 - cos_beta_prime * cos_beta_prime));
    let c_orthogonal = c - dot(c, a) * a;
    if (dot(c_orthogonal, c_orthogonal) == 0.0) {
        return SphericalTriangleSample(vec3<f32>(0.0), 0.0, false);
    }
    let c_prime = cos_beta_prime * a + sin_beta_prime * normalize(c_orthogonal);
    let cos_theta = 1.0 - u.y * (1.0 - dot(c_prime, b));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let cp_orthogonal = c_prime - dot(c_prime, b) * b;
    if (dot(cp_orthogonal, cp_orthogonal) == 0.0) {
        return SphericalTriangleSample(vec3<f32>(0.0), 0.0, false);
    }
    let direction = cos_theta * b + sin_theta * normalize(cp_orthogonal);
    let edge1 = triangle[1] - triangle[0];
    let edge2 = triangle[2] - triangle[0];
    let s1 = cross(direction, edge2);
    let divisor = dot(s1, edge1);
    if (divisor == 0.0) {
        return SphericalTriangleSample(vec3<f32>(1.0 / 3.0), 1.0 / area, true);
    }
    let inverse_divisor = 1.0 / divisor;
    let relative = context_position - triangle[0];
    var b1 = clamp(dot(relative, s1) * inverse_divisor, 0.0, 1.0);
    var b2 = clamp(dot(direction, cross(relative, edge1)) * inverse_divisor, 0.0, 1.0);
    if (b1 + b2 > 1.0) {
        let sum = b1 + b2;
        b1 /= sum;
        b2 /= sum;
    }
    return SphericalTriangleSample(vec3<f32>(1.0 - b1 - b2, b1, b2), 1.0 / area, true);
}

fn invert_spherical_triangle_sample(
    triangle: array<vec3<f32>, 3>,
    context_position: vec3<f32>,
    direction: vec3<f32>,
) -> vec2<f32> {
    var a = normalize(triangle[0] - context_position);
    var b = normalize(triangle[1] - context_position);
    var c = normalize(triangle[2] - context_position);
    var n_ab = normalize(cross(a, b));
    var n_bc = normalize(cross(b, c));
    var n_ca = normalize(cross(c, a));
    let alpha = angle_between(n_ab, -n_ca);
    let beta = angle_between(n_bc, -n_ab);
    let gamma = angle_between(n_ca, -n_bc);
    var c_prime = cross(cross(b, direction), cross(c, a));
    if (dot(c_prime, c_prime) == 0.0) {
        return vec2<f32>(0.5);
    }
    c_prime = normalize(c_prime);
    if (dot(c_prime, a + c) < 0.0) {
        c_prime = -c_prime;
    }
    var u0 = 0.0;
    if (dot(a, c_prime) <= 0.99999847691) {
        var n_cpb = cross(c_prime, b);
        var n_acp = cross(a, c_prime);
        if (dot(n_cpb, n_cpb) == 0.0 || dot(n_acp, n_acp) == 0.0) {
            return vec2<f32>(0.5);
        }
        n_cpb = normalize(n_cpb);
        n_acp = normalize(n_acp);
        let sub_area = alpha + angle_between(n_ab, n_cpb)
            + angle_between(n_acp, -n_cpb) - 3.141592653589793;
        let area = alpha + beta + gamma - 3.141592653589793;
        if (area != 0.0) {
            u0 = sub_area / area;
        }
    }
    let denominator = 1.0 - dot(c_prime, b);
    var u1 = 0.0;
    if (denominator != 0.0) {
        u1 = (1.0 - dot(direction, b)) / denominator;
    }
    return clamp(vec2<f32>(u0, u1), vec2<f32>(0.0), vec2<f32>(1.0));
}

fn area_light_warp_weights(
    triangle: array<vec3<f32>, 3>,
    context_position: vec3<f32>,
    context_shading_normal: vec3<f32>,
) -> vec4<f32> {
    let wi0 = normalize(triangle[0] - context_position);
    let wi1 = normalize(triangle[1] - context_position);
    let wi2 = normalize(triangle[2] - context_position);
    return vec4<f32>(
        max(0.01, abs(dot(context_shading_normal, wi1))),
        max(0.01, abs(dot(context_shading_normal, wi1))),
        max(0.01, abs(dot(context_shading_normal, wi0))),
        max(0.01, abs(dot(context_shading_normal, wi2))),
    );
}

fn area_light_solid_angle(triangle: array<vec3<f32>, 3>, context_position: vec3<f32>) -> f32 {
    return spherical_triangle_area(
        normalize(triangle[0] - context_position),
        normalize(triangle[1] - context_position),
        normalize(triangle[2] - context_position),
    );
}

fn sample_uniform_triangle(u: vec2<f32>) -> vec3<f32> {
    var b0: f32;
    var b1: f32;
    if (u.x < u.y) {
        b0 = 0.5 * u.x;
        b1 = u.y - b0;
    } else {
        b1 = 0.5 * u.y;
        b0 = u.x - b1;
    }
    return vec3<f32>(b0, b1, 1.0 - b0 - b1);
}

fn sample_area_light_uniform(light: Light, context_position: vec3<f32>, u: vec2<f32>) -> AreaLightSample {
    let triangle = area_light_triangle(light);
    let barycentrics = sample_uniform_triangle(u);
    let position = triangle[0] * barycentrics.x + triangle[1] * barycentrics.y + triangle[2] * barycentrics.z;
    let normal = area_light_normal(light, barycentrics, triangle);
    let to_light = position - context_position;
    let distance_squared = dot(to_light, to_light);
    let area = 0.5 * length(cross(triangle[1] - triangle[0], triangle[2] - triangle[0]));
    if (distance_squared == 0.0 || area == 0.0) {
        return AreaLightSample(position, normal, vec2<f32>(0.0), 0.0, false);
    }
    let wi = normalize(to_light);
    let cosine = abs(dot(normal, -wi));
    if (cosine == 0.0) {
        return AreaLightSample(position, normal, vec2<f32>(0.0), 0.0, false);
    }
    let pdf = distance_squared / (cosine * area);
    let emitted = (light.flags & 1u) != 0u || dot(normal, -wi) >= 0.0;
    return AreaLightSample(
        position,
        normal,
        area_light_uv(light, barycentrics),
        pdf,
        emitted && pdf >= 0.0 && pdf <= 3.402823466e38,
    );
}

fn area_light_uniform_pdf(light: Light, context_position: vec3<f32>, hit_position: vec3<f32>, wi: vec3<f32>) -> f32 {
    let triangle = area_light_triangle(light);
    let normal = area_light_normal(light, vec3<f32>(1.0 / 3.0), triangle);
    let area = 0.5 * length(cross(triangle[1] - triangle[0], triangle[2] - triangle[0]));
    let cosine = abs(dot(normal, -wi));
    if (area == 0.0 || cosine == 0.0) {
        return 0.0;
    }
    return dot(hit_position - context_position, hit_position - context_position) / (cosine * area);
}
fn sample_area_light(
    light: Light,
    context_position: vec3<f32>,
    context_shading_normal: vec3<f32>,
    u: vec2<f32>,
) -> AreaLightSample {
    let triangle = area_light_triangle(light);
    let solid_angle = area_light_solid_angle(triangle, context_position);
    if (solid_angle < 3.0e-4 || solid_angle > 6.22) {
        return sample_area_light_uniform(light, context_position, u);
    }
    var warped_u = u;
    var warp_pdf = 1.0;
    if (dot(context_shading_normal, context_shading_normal) != 0.0) {
        let weights = area_light_warp_weights(
            triangle,
            context_position,
            context_shading_normal,
        );
        warped_u = sample_bilinear(u, weights);
        warp_pdf = bilinear_pdf(warped_u, weights);
    }
    let spherical_sample = sample_spherical_triangle(triangle, context_position, warped_u);
    if (!spherical_sample.valid || spherical_sample.pdf == 0.0) {
        return AreaLightSample(vec3<f32>(0.0), vec3<f32>(0.0), vec2<f32>(0.0), 0.0, false);
    }
    let barycentrics = spherical_sample.barycentrics;
    let position = triangle[0] * barycentrics.x
        + triangle[1] * barycentrics.y
        + triangle[2] * barycentrics.z;
    let normal = area_light_normal(light, barycentrics, triangle);
    let wi = normalize(position - context_position);
    let emitted = (light.flags & 1u) != 0u || dot(normal, -wi) >= 0.0;
    return AreaLightSample(
        position,
        normal,
        area_light_uv(light, barycentrics),
        spherical_sample.pdf * warp_pdf,
        emitted,
    );
}

fn area_light_pdf(
    light: Light,
    context_position: vec3<f32>,
    context_shading_normal: vec3<f32>,
    hit_position: vec3<f32>,
    direction: vec3<f32>,
) -> f32 {
    let triangle = area_light_triangle(light);
    let solid_angle = area_light_solid_angle(triangle, context_position);
    if (solid_angle < 3.0e-4 || solid_angle > 6.22) {
        return area_light_uniform_pdf(
            light,
            context_position,
            hit_position,
            direction,
        );
    }
    var pdf = 1.0 / solid_angle;
    if (dot(context_shading_normal, context_shading_normal) != 0.0) {
        let weights = area_light_warp_weights(
            triangle,
            context_position,
            context_shading_normal,
        );
        let u = invert_spherical_triangle_sample(
            triangle,
            context_position,
            direction,
        );
        pdf *= bilinear_pdf(u, weights);
    }
    return pdf;
}
