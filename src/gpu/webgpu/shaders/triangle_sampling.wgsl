const MIN_SPHERICAL_SAMPLE_AREA: f32 = 3e-4;
const MAX_SPHERICAL_SAMPLE_AREA: f32 = 6.22;

struct TriangleVertices {
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
};

fn load_area_triangle(area_index: u32) -> TriangleVertices {
    let instance = instances[load_area_instance(area_index)];
    let geometry = geometries[instance.geometry];
    let first_index = geometry.index_offset + load_area_primitive(area_index) * 3u;
    let i0 = geometry.vertex_offset + indices[first_index];
    let i1 = geometry.vertex_offset + indices[first_index + 1u];
    let i2 = geometry.vertex_offset + indices[first_index + 2u];
    return TriangleVertices(
        instance.world_from_object * vertices[i0].position,
        instance.world_from_object * vertices[i1].position,
        instance.world_from_object * vertices[i2].position,
    );
}

fn spherical_triangle_area(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    return abs(2.0 * atan2(dot(a, cross(b, c)),
        1.0 + dot(a, b) + dot(a, c) + dot(b, c)));
}

fn triangle_solid_angle(triangle: TriangleVertices, p: vec3<f32>) -> f32 {
    let a = triangle.p0.xyz - p;
    let b = triangle.p1.xyz - p;
    let c = triangle.p2.xyz - p;
    if (dot(a, a) == 0.0 || dot(b, b) == 0.0 || dot(c, c) == 0.0) {
        return 0.0;
    }
    return spherical_triangle_area(normalize(a), normalize(b), normalize(c));
}

fn angle_between(a: vec3<f32>, b: vec3<f32>) -> f32 {
    if (dot(a, b) < 0.0) {
        return PI - 2.0 * asin(clamp(length(a + b) * 0.5, 0.0, 1.0));
    }
    return 2.0 * asin(clamp(length(b - a) * 0.5, 0.0, 1.0));
}

fn difference_of_products(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let cd = c * d;
    return fma(a, b, -cd) + fma(-c, d, cd);
}

fn sum_of_products(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let cd = c * d;
    return fma(a, b, cd) + fma(c, d, -cd);
}

fn sample_linear(u: f32, a: f32, b: f32) -> f32 {
    if (u == 0.0 && a == 0.0) {
        return 0.0;
    }
    let x = u * (a + b) / (a + sqrt(mix(a * a, b * b, u)));
    return min(x, 0.99999994);
}

fn sample_bilinear(u: vec2<f32>, weights: vec4<f32>) -> vec2<f32> {
    let y = sample_linear(u.y, weights.x + weights.y, weights.z + weights.w);
    let x = sample_linear(u.x, mix(weights.x, weights.z, y), mix(weights.y, weights.w, y));
    return vec2<f32>(x, y);
}

fn bilinear_pdf(p: vec2<f32>, weights: vec4<f32>) -> f32 {
    if (any(p < vec2<f32>(0.0)) || any(p > vec2<f32>(1.0))) {
        return 0.0;
    }
    let sum = weights.x + weights.y + weights.z + weights.w;
    if (sum == 0.0) {
        return 1.0;
    }
    return 4.0 * ((1.0 - p.x) * (1.0 - p.y) * weights.x
        + p.x * (1.0 - p.y) * weights.y
        + (1.0 - p.x) * p.y * weights.z
        + p.x * p.y * weights.w) / sum;
}

fn triangle_context_weights(
    triangle: TriangleVertices,
    p: vec3<f32>,
    shading_normal: vec3<f32>,
) -> vec4<f32> {
    let wi0 = normalize(triangle.p0.xyz - p);
    let wi1 = normalize(triangle.p1.xyz - p);
    let wi2 = normalize(triangle.p2.xyz - p);
    return vec4<f32>(
        max(0.01, abs(dot(shading_normal, wi1))),
        max(0.01, abs(dot(shading_normal, wi1))),
        max(0.01, abs(dot(shading_normal, wi0))),
        max(0.01, abs(dot(shading_normal, wi2))),
    );
}

// xyz contains barycentrics and w contains the solid-angle PDF.
fn sample_spherical_triangle(
    triangle: TriangleVertices,
    p: vec3<f32>,
    u: vec2<f32>,
) -> vec4<f32> {
    let a = normalize(triangle.p0.xyz - p);
    let b = normalize(triangle.p1.xyz - p);
    let c = normalize(triangle.p2.xyz - p);
    var n_ab = cross(a, b);
    var n_bc = cross(b, c);
    var n_ca = cross(c, a);
    if (dot(n_ab, n_ab) == 0.0 || dot(n_bc, n_bc) == 0.0 || dot(n_ca, n_ca) == 0.0) {
        return vec4<f32>(0.0);
    }
    n_ab = normalize(n_ab);
    n_bc = normalize(n_bc);
    n_ca = normalize(n_ca);
    let alpha = angle_between(n_ab, -n_ca);
    let beta = angle_between(n_bc, -n_ab);
    let gamma_angle = angle_between(n_ca, -n_bc);
    let area_plus_pi = alpha + beta + gamma_angle;
    let area = area_plus_pi - PI;
    if (area <= 0.0) {
        return vec4<f32>(0.0);
    }
    let ap_pi = mix(PI, area_plus_pi, u.x);
    let cos_alpha = cos(alpha);
    let sin_alpha = sin(alpha);
    let sin_phi = sin(ap_pi) * cos_alpha - cos(ap_pi) * sin_alpha;
    let cos_phi = cos(ap_pi) * cos_alpha + sin(ap_pi) * sin_alpha;
    let k1 = cos_phi + cos_alpha;
    let k2 = sin_phi - sin_alpha * dot(a, b);
    let denominator = sum_of_products(k2, sin_phi, k1, cos_phi) * sin_alpha;
    if (denominator == 0.0) {
        return vec4<f32>(0.0);
    }
    var cos_bp = (k2 + difference_of_products(k2, cos_phi, k1, sin_phi) * cos_alpha)
        / denominator;
    cos_bp = clamp(cos_bp, -1.0, 1.0);
    let ac_perpendicular = c - dot(c, a) * a;
    if (dot(ac_perpendicular, ac_perpendicular) == 0.0) {
        return vec4<f32>(0.0);
    }
    let cp = cos_bp * a + sqrt(max(0.0, 1.0 - cos_bp * cos_bp))
        * normalize(ac_perpendicular);
    let cos_theta = 1.0 - u.y * (1.0 - dot(cp, b));
    let bp_perpendicular = cp - dot(cp, b) * b;
    if (dot(bp_perpendicular, bp_perpendicular) == 0.0) {
        return vec4<f32>(0.0);
    }
    let w = cos_theta * b + sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        * normalize(bp_perpendicular);
    let e1 = triangle.p1.xyz - triangle.p0.xyz;
    let e2 = triangle.p2.xyz - triangle.p0.xyz;
    let s1 = cross(w, e2);
    let divisor = dot(s1, e1);
    if (divisor == 0.0) {
        return vec4<f32>(vec3<f32>(1.0 / 3.0), 1.0 / area);
    }
    let s = p - triangle.p0.xyz;
    var b1 = clamp(dot(s, s1) / divisor, 0.0, 1.0);
    var b2 = clamp(dot(w, cross(s, e1)) / divisor, 0.0, 1.0);
    if (b1 + b2 > 1.0) {
        let sum = b1 + b2;
        b1 = b1 / sum;
        b2 = b2 / sum;
    }
    return vec4<f32>(1.0 - b1 - b2, b1, b2, 1.0 / area);
}

fn invert_spherical_triangle_sample(
    triangle: TriangleVertices,
    p: vec3<f32>,
    w: vec3<f32>,
) -> vec2<f32> {
    let a = normalize(triangle.p0.xyz - p);
    let b = normalize(triangle.p1.xyz - p);
    let c = normalize(triangle.p2.xyz - p);
    var n_ab = cross(a, b);
    var n_bc = cross(b, c);
    var n_ca = cross(c, a);
    if (dot(n_ab, n_ab) == 0.0 || dot(n_bc, n_bc) == 0.0 || dot(n_ca, n_ca) == 0.0) {
        return vec2<f32>(0.5);
    }
    n_ab = normalize(n_ab);
    n_bc = normalize(n_bc);
    n_ca = normalize(n_ca);
    let alpha = angle_between(n_ab, -n_ca);
    let beta = angle_between(n_bc, -n_ab);
    let gamma_angle = angle_between(n_ca, -n_bc);
    var cp = cross(cross(b, w), cross(c, a));
    if (dot(cp, cp) == 0.0) {
        return vec2<f32>(0.5);
    }
    cp = normalize(cp);
    if (dot(cp, a + c) < 0.0) {
        cp = -cp;
    }
    var u0 = 0.0;
    if (dot(a, cp) <= 0.99999847691) {
        var n_cpb = cross(cp, b);
        var n_acp = cross(a, cp);
        if (dot(n_cpb, n_cpb) == 0.0 || dot(n_acp, n_acp) == 0.0) {
            return vec2<f32>(0.5);
        }
        n_cpb = normalize(n_cpb);
        n_acp = normalize(n_acp);
        let area_p = alpha + angle_between(n_ab, n_cpb) + angle_between(n_acp, -n_cpb) - PI;
        let area = alpha + beta + gamma_angle - PI;
        if (area == 0.0) {
            return vec2<f32>(0.5);
        }
        u0 = area_p / area;
    }
    let denominator = 1.0 - dot(cp, b);
    if (denominator == 0.0) {
        return vec2<f32>(0.5);
    }
    let u1 = (1.0 - dot(w, b)) / denominator;
    return clamp(vec2<f32>(u0, u1), vec2<f32>(0.0), vec2<f32>(1.0));
}

// xyz contains barycentrics and w contains the selected triangle's solid-angle PDF.
fn sample_triangle_for_context(
    triangle: TriangleVertices,
    p: vec3<f32>,
    shading_normal: vec3<f32>,
    light_normal: vec3<f32>,
    total_area: f32,
    u: vec2<f32>,
) -> vec4<f32> {
    let solid_angle = triangle_solid_angle(triangle, p);
    if (solid_angle < MIN_SPHERICAL_SAMPLE_AREA || solid_angle > MAX_SPHERICAL_SAMPLE_AREA) {
        let su = sqrt(u.x);
        let b = vec3<f32>(1.0 - su, su * (1.0 - u.y), su * u.y);
        let sampled_point = triangle.p0.xyz * b.x + triangle.p1.xyz * b.y + triangle.p2.xyz * b.z;
        let to_light = sampled_point - p;
        let distance_squared = dot(to_light, to_light);
        if (distance_squared == 0.0) {
            return vec4<f32>(0.0);
        }
        let cosine = abs(dot(light_normal, -normalize(to_light)));
        if (cosine == 0.0) {
            return vec4<f32>(0.0);
        }
        return vec4<f32>(b, distance_squared / (cosine * total_area));
    }
    var warped_u = u;
    var warp_pdf = 1.0;
    if (dot(shading_normal, shading_normal) > 0.0) {
        let weights = triangle_context_weights(triangle, p, shading_normal);
        warped_u = sample_bilinear(u, weights);
        warp_pdf = bilinear_pdf(warped_u, weights);
    }
    let sample = sample_spherical_triangle(triangle, p, warped_u);
    return vec4<f32>(sample.xyz, sample.w * warp_pdf);
}

fn triangle_pdf_for_context(
    triangle: TriangleVertices,
    p: vec3<f32>,
    shading_normal: vec3<f32>,
    light_normal: vec3<f32>,
    sampled_point: vec3<f32>,
    total_area: f32,
) -> f32 {
    let to_light = sampled_point - p;
    let distance_squared = dot(to_light, to_light);
    if (distance_squared == 0.0) {
        return 0.0;
    }
    let wi = normalize(to_light);
    let solid_angle = triangle_solid_angle(triangle, p);
    if (solid_angle < MIN_SPHERICAL_SAMPLE_AREA || solid_angle > MAX_SPHERICAL_SAMPLE_AREA) {
        let cosine = abs(dot(light_normal, -wi));
        if (cosine == 0.0) {
            return 0.0;
        }
        return distance_squared / (cosine * total_area);
    }
    var pdf = 1.0 / solid_angle;
    if (dot(shading_normal, shading_normal) > 0.0) {
        let weights = triangle_context_weights(triangle, p, shading_normal);
        let warped_u = invert_spherical_triangle_sample(triangle, p, wi);
        pdf = pdf * bilinear_pdf(warped_u, weights);
    }
    return pdf;
}
