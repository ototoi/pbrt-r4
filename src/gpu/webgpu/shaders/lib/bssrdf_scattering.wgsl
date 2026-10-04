fn bssrdf_fresnel_moment1(eta: f32) -> f32 {
    let e2 = eta * eta; let e3 = e2 * eta; let e4 = e3 * eta; let e5 = e4 * eta;
    if (eta < 1.0) { return 0.45966 - 1.73965*eta + 3.37668*e2 - 3.904945*e3 + 2.49277*e4 - 0.68441*e5; }
    return -4.61686 + 11.1136*eta - 10.4646*e2 + 5.11455*e3 - 1.27198*e4 + 0.12746*e5;
}
fn bssrdf_normalized_fresnel(cosine: f32, eta: f32) -> vec4<f32> {
    if (cosine <= 0.0) { return vec4<f32>(0.0); }
    let c = 1.0 - 2.0 * bssrdf_fresnel_moment1(1.0 / eta);
    return vec4<f32>((1.0 - dielectric_fresnel(cosine, eta)) * eta * eta / (c * PI));
}
fn bssrdf_sample_cosine(u: vec2<f32>) -> vec3<f32> {
    let o = 2.0*u - vec2<f32>(1.0);
    if (all(o == vec2<f32>(0.0))) { return vec3<f32>(0.0, 0.0, 1.0); }
    var r = o.y;
    var theta = PI * 0.5 - PI * 0.25 * (o.x / o.y);
    if (abs(o.x) > abs(o.y)) { r = o.x; theta = PI * 0.25 * (o.y / o.x); }
    let d = r * vec2<f32>(cos(theta), sin(theta));
    return vec3<f32>(d, sqrt(max(0.0, 1.0 - dot(d, d))));
}
