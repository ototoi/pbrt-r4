fn texture_noise_weight(t: f32) -> f32 {
    let t3 = t * t * t;
    let t4 = t3 * t;
    return 6.0 * t4 * t - 15.0 * t4 + 10.0 * t3;
}

fn texture_noise_perm(i: u32) -> u32 {
    return textureLoad(texture_noise_table, vec2<i32>(i32(i & 255u), 0), 0).x;
}

fn texture_noise_pair(a: u32, b: u32) -> u32 {
    return textureLoad(
        texture_noise_table,
        vec2<i32>(i32(a & 255u), i32((b & 255u) + 1u)),
        0,
    ).x;
}

fn texture_noise_grad(x: u32, y: u32, z: u32, dx: f32, dy: f32, dz: f32) -> f32 {
    let a = texture_noise_pair(x, y);
    let h = texture_noise_perm(a + z) & 15u;
    let u = select(dy, dx, h < 8u || h == 12u || h == 13u);
    let v = select(dz, dy, h < 4u || h == 12u || h == 13u);
    return select(u, -u, (h & 1u) != 0u) + select(v, -v, (h & 2u) != 0u);
}

fn texture_noise(p: vec3<f32>) -> f32 {
    let ix = i32(floor(p.x));
    let iy = i32(floor(p.y));
    let iz = i32(floor(p.z));
    let dx = p.x - f32(ix);
    let dy = p.y - f32(iy);
    let dz = p.z - f32(iz);
    let x = u32(ix) & 255u;
    let y = u32(iy) & 255u;
    let z = u32(iz) & 255u;
    let w000 = texture_noise_grad(x, y, z, dx, dy, dz);
    let w100 = texture_noise_grad((x + 1u) & 255u, y, z, dx - 1.0, dy, dz);
    let w010 = texture_noise_grad(x, (y + 1u) & 255u, z, dx, dy - 1.0, dz);
    let w110 = texture_noise_grad((x + 1u) & 255u, (y + 1u) & 255u, z, dx - 1.0, dy - 1.0, dz);
    let w001 = texture_noise_grad(x, y, (z + 1u) & 255u, dx, dy, dz - 1.0);
    let w101 = texture_noise_grad((x + 1u) & 255u, y, (z + 1u) & 255u, dx - 1.0, dy, dz - 1.0);
    let w011 = texture_noise_grad(x, (y + 1u) & 255u, (z + 1u) & 255u, dx, dy - 1.0, dz - 1.0);
    let w111 = texture_noise_grad((x + 1u) & 255u, (y + 1u) & 255u, (z + 1u) & 255u, dx - 1.0, dy - 1.0, dz - 1.0);
    let wx = texture_noise_weight(dx);
    let wy = texture_noise_weight(dy);
    let wz = texture_noise_weight(dz);
    let x0 = mix(w000, w100, wx);
    let x1 = mix(w010, w110, wx);
    let x2 = mix(w001, w101, wx);
    let x3 = mix(w011, w111, wx);
    return mix(mix(x0, x1, wy), mix(x2, x3, wy), wz);
}

fn texture_fbm(p: vec3<f32>, omega: f32, octaves: f32, absolute_value: bool) -> f32 {
    let count = u32(clamp(floor(octaves), 0.0, 8.0));
    var sum = 0.0;
    var frequency = 1.0;
    var weight = 1.0;
    for (var octave = 0u; octave < count; octave++) {
        let value = texture_noise(frequency * p);
        sum += weight * select(value, abs(value), absolute_value);
        frequency *= 1.99;
        weight *= omega;
    }
    if (absolute_value) {
        var tail_weight = weight;
        for (var tail = 0u; tail < count; tail++) {
            sum += tail_weight * 0.2;
            tail_weight *= omega;
        }
    }
    return sum;
}

fn texture_marble(p: vec3<f32>, omega: f32, octaves: f32, scale: f32, variation: f32) -> vec3<f32> {
    let q = scale * p;
    let marble = q.y + variation * texture_fbm(q, omega, octaves, false);
    var t = 0.5 + 0.5 * sin(marble);
    let colors = array<vec3<f32>, 9>(
        vec3<f32>(0.58, 0.58, 0.60), vec3<f32>(0.58, 0.58, 0.60),
        vec3<f32>(0.58, 0.58, 0.60), vec3<f32>(0.50, 0.50, 0.50),
        vec3<f32>(0.60, 0.59, 0.58), vec3<f32>(0.58, 0.58, 0.60),
        vec3<f32>(0.58, 0.58, 0.60), vec3<f32>(0.20, 0.20, 0.33),
        vec3<f32>(0.58, 0.58, 0.60),
    );
    let segments = 6;
    let scaled_t = t * f32(segments);
    let first = min(u32(floor(scaled_t)), 5u);
    t = scaled_t - f32(first);
    let c0 = colors[first];
    let c1 = colors[first + 1u];
    let c2 = colors[first + 2u];
    let c3 = colors[first + 3u];
    let a = mix(c0, c1, t);
    let b = mix(c1, c2, t);
    let c = mix(c2, c3, t);
    let d = mix(a, b, t);
    let e = mix(b, c, t);
    return 1.5 * mix(d, e, t);
}

fn mapped_texture_uv(node: TextureNodeRecord, uv: vec2<f32>) -> vec2<f32> {
    if (node.mapping_kind == 1u) {
        return (node.mapping * vec4<f32>(material_texture_position, 1.0)).xy;
    }
    let p = (node.mapping * vec4<f32>(material_texture_position, 1.0)).xyz;
    if (node.mapping_kind == 2u) {
        let q = normalize(p);
        var phi = atan2(q.y, q.x) / (2.0 * 3.14159265359);
        if (phi < 0.0) { phi = phi + 1.0; }
        return vec2<f32>(acos(clamp(q.z, -1.0, 1.0)) / 3.14159265359, phi);
    }
    if (node.mapping_kind == 3u) {
        let s = (3.14159265359 + atan2(p.y, p.x)) / (2.0 * 3.14159265359);
        return vec2<f32>(s, p.z);
    }
    return (node.mapping * vec4<f32>(uv, 0.0, 1.0)).xy;
}

fn mapped_texture_position(node: TextureNodeRecord) -> vec3<f32> {
    return (node.mapping * vec4<f32>(material_texture_position, 1.0)).xyz;
}

fn sample_texture_leaf(texture_index: u32, uv: vec2<f32>) -> vec3<f32> {
    if (texture_index >= arrayLength(&texture_nodes)) { set_render_error(); return vec3<f32>(0.0); }
    let node = texture_nodes[texture_index];
    if (node.operation == TEXTURE_OPERATION_CONSTANT) { return node.constant_value.rgb; }
    let mapped_uv = mapped_texture_uv(node, uv);
    if ((node.swrap_mode == 2u && (mapped_uv.x < 0.0 || mapped_uv.x > 1.0))
        || (node.twrap_mode == 2u && (mapped_uv.y < 0.0 || mapped_uv.y > 1.0))) {
        return vec3<f32>(0.0);
    }
    // pbrt-v4 image textures use image-space origin at the upper left.
    let image_uv = vec2<f32>(mapped_uv.x, 1.0 - mapped_uv.y);
    var value = textureSampleLevel(
        texture_images[node.texture_index],
        texture_samplers[node.sampler],
        image_uv,
        0.0,
    ).rgb * node.constant_value.x;
    if (node.constant_value.y > 0.5) {
        value = max(vec3<f32>(0.0), vec3<f32>(1.0) - value);
    }
    return value;
}

fn sample_texture_program(root: TextureRootRecord, uv: vec2<f32>) -> vec3<f32> {
    var values: array<vec3<f32>, TEXTURE_PROGRAM_CAPACITY>;
    if (root.instruction_count > TEXTURE_PROGRAM_CAPACITY
        || root.result >= root.instruction_count) {
        set_render_error();
        return vec3<f32>(0.0);
    }
    var local = 0u;
    loop {
        if (local >= root.instruction_count) { break; }
        let node_index = root.texture_node + local;
        if (node_index >= arrayLength(&texture_nodes)) {
            set_render_error();
            return vec3<f32>(0.0);
        }
        let node = texture_nodes[node_index];
        if (node.operation == TEXTURE_OPERATION_IMAGE) {
            values[local] = sample_texture_leaf(node_index, uv);
        } else if (node.operation == TEXTURE_OPERATION_CONSTANT) {
            values[local] = node.constant_value.rgb;
        } else if (node.operation == TEXTURE_OPERATION_SCALE) {
            if ((node.child_count != 1u && node.child_count != 2u)
                || node.first_child >= arrayLength(&texture_child_indices)
                || (node.child_count == 2u
                    && node.first_child + 1u >= arrayLength(&texture_child_indices))) {
                set_render_error();
                return vec3<f32>(0.0);
            }
            let child = texture_child_indices[node.first_child];
            if (child >= local) {
                set_render_error();
                return vec3<f32>(0.0);
            }
            var factor = node.constant_value.x;
            if (node.child_count == 2u) {
                let scale = texture_child_indices[node.first_child + 1u];
                if (scale >= local) {
                    set_render_error();
                    return vec3<f32>(0.0);
                }
                factor = values[scale].x;
            }
            if (factor == 0.0) {
                values[local] = vec3<f32>(0.0);
            } else {
                values[local] = values[child] * factor;
            }
        } else if (node.operation == TEXTURE_OPERATION_MIX) {
            if (node.child_count < 2u
                || node.first_child + 1u >= arrayLength(&texture_child_indices)) {
                set_render_error();
                return vec3<f32>(0.0);
            }
            let first = texture_child_indices[node.first_child];
            let second = texture_child_indices[node.first_child + 1u];
            if (first >= local || second >= local) {
                set_render_error();
                return vec3<f32>(0.0);
            }
            var amount = node.constant_value.x;
            if (node.child_count >= 3u) {
                if (node.first_child + 2u >= arrayLength(&texture_child_indices)) {
                    set_render_error();
                    return vec3<f32>(0.0);
                }
                let amount_slot = texture_child_indices[node.first_child + 2u];
                if (amount_slot >= local) {
                    set_render_error();
                    return vec3<f32>(0.0);
                }
                amount = clamp(values[amount_slot].x, 0.0, 1.0);
            }
            values[local] = mix(values[first], values[second], amount);
        } else if (node.operation == TEXTURE_OPERATION_CHECKERBOARD
            || node.operation == TEXTURE_OPERATION_DIRECTION_MIX
            || node.operation == TEXTURE_OPERATION_BILERP) {
            if (node.first_child + node.child_count > arrayLength(&texture_child_indices)) {
                set_render_error();
                return vec3<f32>(0.0);
            }
            if (node.operation == TEXTURE_OPERATION_BILERP) {
                if (node.child_count != 4u) { set_render_error(); return vec3<f32>(0.0); }
                let v00 = texture_child_indices[node.first_child];
                let v01 = texture_child_indices[node.first_child + 1u];
                let v10 = texture_child_indices[node.first_child + 2u];
                let v11 = texture_child_indices[node.first_child + 3u];
                if (v00 >= local || v01 >= local || v10 >= local || v11 >= local) {
                    set_render_error(); return vec3<f32>(0.0);
                }
                let st = mapped_texture_uv(node, uv);
                values[local] = mix(mix(values[v00], values[v10], st.x),
                                    mix(values[v01], values[v11], st.x), st.y);
            } else {
                if (node.child_count != 2u) { set_render_error(); return vec3<f32>(0.0); }
                let first = texture_child_indices[node.first_child];
                let second = texture_child_indices[node.first_child + 1u];
                if (first >= local || second >= local) {
                    set_render_error(); return vec3<f32>(0.0);
                }
                if (node.operation == TEXTURE_OPERATION_DIRECTION_MIX) {
                    let amount = abs((node.mapping * vec4<f32>(material_texture_normal, 0.0)).x);
                    values[local] = mix(values[second], values[first], amount);
                } else {
                    var odd = false;
                    if (node.mapping_kind == 4u) {
                        let p = (node.mapping * vec4<f32>(material_texture_position, 1.0)).xyz;
                        odd = (i32(floor(p.x)) + i32(floor(p.y)) + i32(floor(p.z))) % 2 != 0;
                    } else {
                        let st = mapped_texture_uv(node, uv);
                        odd = (i32(floor(st.x)) + i32(floor(st.y))) % 2 != 0;
                    }
                    values[local] = select(values[first], values[second], odd);
                }
            }
        // TEXTURE_NOISE_BRANCH_BEGIN
        } else if (node.operation == TEXTURE_OPERATION_DOTS
            || node.operation == TEXTURE_OPERATION_FBM
            || node.operation == TEXTURE_OPERATION_WRINKLED
            || node.operation == TEXTURE_OPERATION_WINDY
            || node.operation == TEXTURE_OPERATION_MARBLE) {
            if (node.operation == TEXTURE_OPERATION_DOTS) {
                if (node.child_count != 2u
                    || node.first_child + 1u >= arrayLength(&texture_child_indices)) {
                    set_render_error(); return vec3<f32>(0.0);
                }
                let outside = texture_child_indices[node.first_child];
                let inside = texture_child_indices[node.first_child + 1u];
                if (outside >= local || inside >= local) {
                    set_render_error(); return vec3<f32>(0.0);
                }
                let st = mapped_texture_uv(node, uv);
                let cell = floor(st + vec2<f32>(0.5));
                var is_inside = false;
                if (texture_noise(vec3<f32>(cell + vec2<f32>(0.5), 0.0)) > 0.0) {
                    let center = cell + 0.15 * vec2<f32>(
                        texture_noise(vec3<f32>(cell.x + 1.5, cell.y + 2.8, 0.0)),
                        texture_noise(vec3<f32>(cell.x + 4.5, cell.y + 9.8, 0.0)),
                    );
                    let delta = st - center;
                    is_inside = dot(delta, delta) < 0.35 * 0.35;
                }
                values[local] = select(values[outside], values[inside], is_inside);
            } else {
                let p = mapped_texture_position(node);
                if (node.operation == TEXTURE_OPERATION_FBM) {
                    values[local] = vec3<f32>(texture_fbm(
                        p, node.constant_value.x, node.constant_value.y, false));
                } else if (node.operation == TEXTURE_OPERATION_WRINKLED) {
                    values[local] = vec3<f32>(texture_fbm(
                        p, node.constant_value.x, node.constant_value.y, true));
                } else if (node.operation == TEXTURE_OPERATION_WINDY) {
                    let wind = texture_fbm(0.1 * p, 0.5, 3.0, false);
                    let wave = texture_fbm(p, 0.5, 6.0, false);
                    values[local] = vec3<f32>(abs(wind) * wave);
                } else {
                    values[local] = texture_marble(
                        p,
                        node.constant_value.x,
                        node.constant_value.y,
                        node.constant_value.z,
                        node.constant_value.w,
                    );
                }
            }
        // TEXTURE_NOISE_BRANCH_END
        } else {
            set_render_error();
            return vec3<f32>(0.0);
        }
        local += 1u;
    }
    return values[root.result];
}

var<private> material_texture_uv: vec2<f32>;
var<private> material_texture_normal: vec3<f32>;
var<private> material_texture_position: vec3<f32>;
var<private> material_texture_eval_base: u32;
