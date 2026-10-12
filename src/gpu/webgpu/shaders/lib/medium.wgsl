fn cloud_dnoise(point: vec3<f32>) -> vec3<f32> {
    let delta = 0.01;
    let value = noise(point);
    return vec3<f32>(
        noise(point + vec3<f32>(delta, 0.0, 0.0)) - value,
        noise(point + vec3<f32>(0.0, delta, 0.0)) - value,
        noise(point + vec3<f32>(0.0, 0.0, delta)) - value,
    ) / delta;
}

fn cloud_density(medium: VolumeMediumRecord, point: vec3<f32>) -> f32 {
    var p = medium.cloud_parameters.z * point;
    if (medium.cloud_parameters.y > 0.0) {
        var omega = 0.05 * medium.cloud_parameters.y;
        var frequency = 10.0;
        for (var octave = 0u; octave < 2u; octave += 1u) {
            p += omega * cloud_dnoise(frequency * p);
            omega *= 0.5;
            frequency *= 1.99;
        }
    }
    var density = 0.0;
    var omega = 0.5;
    var frequency = 1.0;
    for (var octave = 0u; octave < 5u; octave += 1u) {
        density += omega * noise(frequency * p);
        omega *= 0.5;
        frequency *= 1.99;
    }
    density = clamp((1.0 - point.y) * 4.5 * medium.cloud_parameters.x * density, 0.0, 1.0);
    density += 2.0 * max(0.0, 0.5 - point.y);
    return clamp(density, 0.0, 1.0);
}

fn uniform_grid_scalar_value(
    offset: u32,
    resolution_u: vec3<u32>,
    coordinate: vec3<i32>,
) -> f32 {
    let resolution = vec3<i32>(resolution_u);
    if (any(coordinate < vec3<i32>(0)) || any(coordinate >= resolution)) {
        return 0.0;
    }
    let index = u32(coordinate.z) * resolution_u.y * resolution_u.x
        + u32(coordinate.y) * resolution_u.x + u32(coordinate.x);
    return volume_data[offset + index];
}

fn uniform_grid_scalar(
    grid: VolumeMediumRecord,
    offset: u32,
    resolution: vec3<u32>,
    position_medium: vec3<f32>,
) -> f32 {
    let bounds_diagonal = grid.bounds_max.xyz - grid.bounds_min.xyz;
    let unit = (position_medium - grid.bounds_min.xyz) / bounds_diagonal;
    let sample_position = unit * vec3<f32>(resolution) - vec3<f32>(0.5);
    let base = vec3<i32>(floor(sample_position));
    let delta = sample_position - vec3<f32>(base);
    let d00 = mix(
        uniform_grid_scalar_value(offset, resolution, base),
        uniform_grid_scalar_value(offset, resolution, base + vec3<i32>(1, 0, 0)), delta.x,
    );
    let d10 = mix(
        uniform_grid_scalar_value(offset, resolution, base + vec3<i32>(0, 1, 0)),
        uniform_grid_scalar_value(offset, resolution, base + vec3<i32>(1, 1, 0)), delta.x,
    );
    let d01 = mix(
        uniform_grid_scalar_value(offset, resolution, base + vec3<i32>(0, 0, 1)),
        uniform_grid_scalar_value(offset, resolution, base + vec3<i32>(1, 0, 1)), delta.x,
    );
    let d11 = mix(
        uniform_grid_scalar_value(offset, resolution, base + vec3<i32>(0, 1, 1)),
        uniform_grid_scalar_value(offset, resolution, base + vec3<i32>(1, 1, 1)), delta.x,
    );
    return mix(mix(d00, d10, delta.y), mix(d01, d11, delta.y), delta.z);
}

fn uniform_grid_density(grid: VolumeMediumRecord, position_medium: vec3<f32>) -> f32 {
    return uniform_grid_scalar(
        grid, grid.density_offset_count.x, grid.resolution.xyz, position_medium,
    );
}

fn sparse_grid_sample(grid: SparseScalarGridRecord, position_medium: vec3<f32>) -> f32 {
    let point = position_medium - grid.translation.xyz;
    let index = vec3<f32>(
        fma(point.x, grid.inverse_map[0].x,
            fma(point.y, grid.inverse_map[0].y, point.z * grid.inverse_map[0].z)),
        fma(point.x, grid.inverse_map[1].x,
            fma(point.y, grid.inverse_map[1].y, point.z * grid.inverse_map[1].z)),
        fma(point.x, grid.inverse_map[2].x,
            fma(point.y, grid.inverse_map[2].y, point.z * grid.inverse_map[2].z)),
    );
    if (any(index != index) || any(abs(index) > vec3<f32>(RAY_T_MAX))
        || any(index < vec3<f32>(-2147483648.0))
        || any(index >= vec3<f32>(2147483520.0))) {
        set_render_error();
        return 0.0;
    }
    let base = vec3<i32>(floor(index));
    if (any(base == vec3<i32>(2147483647))) {
        set_render_error();
        return 0.0;
    }
    let fraction = index - vec3<f32>(base);
    let v000 = sparse_grid_get_value(grid, base);
    let v001 = sparse_grid_get_value(grid, base + vec3<i32>(0, 0, 1));
    let v010 = sparse_grid_get_value(grid, base + vec3<i32>(0, 1, 0));
    let v011 = sparse_grid_get_value(grid, base + vec3<i32>(0, 1, 1));
    let v100 = sparse_grid_get_value(grid, base + vec3<i32>(1, 0, 0));
    let v101 = sparse_grid_get_value(grid, base + vec3<i32>(1, 0, 1));
    let v110 = sparse_grid_get_value(grid, base + vec3<i32>(1, 1, 0));
    let v111 = sparse_grid_get_value(grid, base + vec3<i32>(1, 1, 1));
    let z00 = v000 + fraction.z * (v001 - v000);
    let z01 = v010 + fraction.z * (v011 - v010);
    let z10 = v100 + fraction.z * (v101 - v100);
    let z11 = v110 + fraction.z * (v111 - v110);
    let y0 = z00 + fraction.y * (z01 - z00);
    let y1 = z10 + fraction.y * (z11 - z10);
    let density = y0 + fraction.x * (y1 - y0);
    if (density != density || abs(density) > RAY_T_MAX) {
        set_render_error();
        return 0.0;
    }
    return density;
}

fn sparse_grid_density(grid: VolumeMediumRecord, position_medium: vec3<f32>) -> f32 {
    let density = sparse_grid_sample(grid.density_grid, position_medium);
    if (density < 0.0) { set_render_error(); return 0.0; }
    return density;
}

fn rgb_grid_coefficient_value(
    grid: VolumeMediumRecord,
    offset_count: vec4<u32>,
    coordinate: vec3<i32>,
    lambda: vec4<f32>,
) -> vec4<f32> {
    if (offset_count.x == 0xffffffffu) { return vec4<f32>(1.0); }
    let resolution = vec3<i32>(grid.resolution.xyz);
    if (any(coordinate < vec3<i32>(0)) || any(coordinate >= resolution)) {
        return vec4<f32>(0.0);
    }
    let index = u32(coordinate.z) * grid.resolution.y * grid.resolution.x
        + u32(coordinate.y) * grid.resolution.x + u32(coordinate.x);
    let offset = offset_count.x + 3u * index;
    let rgb = vec3<f32>(volume_data[offset], volume_data[offset + 1u], volume_data[offset + 2u]);
    return rgb_to_unbounded_spectrum4(rgb, lambda, grid.resolution.w);
}

fn rgb_grid_coefficient(
    grid: VolumeMediumRecord,
    offset_count: vec4<u32>,
    position_medium: vec3<f32>,
    lambda: vec4<f32>,
) -> vec4<f32> {
    if (offset_count.x == 0xffffffffu) { return vec4<f32>(1.0); }
    let bounds_diagonal = grid.bounds_max.xyz - grid.bounds_min.xyz;
    let unit = (position_medium - grid.bounds_min.xyz) / bounds_diagonal;
    let sample_position = unit * vec3<f32>(grid.resolution.xyz) - vec3<f32>(0.5);
    let base = vec3<i32>(floor(sample_position));
    let delta = sample_position - vec3<f32>(base);
    let d00 = mix(
        rgb_grid_coefficient_value(grid, offset_count, base, lambda),
        rgb_grid_coefficient_value(grid, offset_count, base + vec3<i32>(1, 0, 0), lambda),
        delta.x,
    );
    let d10 = mix(
        rgb_grid_coefficient_value(grid, offset_count, base + vec3<i32>(0, 1, 0), lambda),
        rgb_grid_coefficient_value(grid, offset_count, base + vec3<i32>(1, 1, 0), lambda),
        delta.x,
    );
    let d01 = mix(
        rgb_grid_coefficient_value(grid, offset_count, base + vec3<i32>(0, 0, 1), lambda),
        rgb_grid_coefficient_value(grid, offset_count, base + vec3<i32>(1, 0, 1), lambda),
        delta.x,
    );
    let d11 = mix(
        rgb_grid_coefficient_value(grid, offset_count, base + vec3<i32>(0, 1, 1), lambda),
        rgb_grid_coefficient_value(grid, offset_count, base + vec3<i32>(1, 1, 1), lambda),
        delta.x,
    );
    return mix(mix(d00, d10, delta.y), mix(d01, d11, delta.y), delta.z);
}

struct UniformGridDda {
    valid: u32,
    t_min: f32,
    t_max: f32,
    padding: f32,
    next_crossing_t: vec3<f32>,
    delta_t: vec3<f32>,
    step: vec3<i32>,
    voxel_limit: vec3<i32>,
    voxel: vec3<i32>,
}

struct UniformGridMajorantSegment {
    t_min: f32,
    t_max: f32,
    sigma_maj: vec4<f32>,
}

fn uniform_grid_dda_init(
    grid: VolumeMediumRecord,
    origin_world: vec3<f32>,
    direction_world: vec3<f32>,
    t_min: f32,
    t_max: f32,
) -> UniformGridDda {
    var dda: UniformGridDda;
    dda.valid = 0u;
    dda.t_min = t_min;
    dda.t_max = t_min;
    dda.padding = 0.0;
    dda.next_crossing_t = vec3<f32>(bitcast<f32>(0x7f800000u));
    dda.delta_t = vec3<f32>(bitcast<f32>(0x7f800000u));
    dda.step = vec3<i32>(0);
    dda.voxel_limit = vec3<i32>(0);
    dda.voxel = vec3<i32>(0);

    var origin_medium = (grid.medium_from_world * vec4<f32>(origin_world, 1.0)).xyz;
    let direction_medium = (grid.medium_from_world * vec4<f32>(direction_world, 0.0)).xyz;
    // Transform::ApplyInverse(Point3fi) uses this error bound for an exact
    // input point; ApplyInverse(Ray) advances the origin and shortens tMax.
    // v4's MachineEpsilon is half the f32 spacing at one.
    let unit_roundoff = 0.5 * MACHINE_EPSILON;
    let gamma3 = (3.0 * unit_roundoff) / (1.0 - 3.0 * unit_roundoff);
    let origin_error = gamma3 * (
        abs(grid.medium_from_world[0].xyz * origin_world.x)
        + abs(grid.medium_from_world[1].xyz * origin_world.y)
        + abs(grid.medium_from_world[2].xyz * origin_world.z)
    );
    var ray_t_max = t_max;
    let length_squared = dot(direction_medium, direction_medium);
    if (length_squared > 0.0) {
        let dt = dot(abs(direction_medium), origin_error) / length_squared;
        origin_medium += direction_medium * dt;
        ray_t_max -= dt;
    }
    let bounds_diagonal = grid.bounds_max.xyz - grid.bounds_min.xyz;
    let origin_grid = (origin_medium - grid.bounds_min.xyz) / bounds_diagonal;
    let direction_grid = direction_medium / bounds_diagonal;
    var t_enter = t_min;
    var t_exit = ray_t_max;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if (direction_grid[axis] == 0.0) {
            if (origin_grid[axis] < 0.0 || origin_grid[axis] > 1.0) {
                return dda;
            }
        } else {
            let t0 = (0.0 - origin_grid[axis]) / direction_grid[axis];
            let t1 = (1.0 - origin_grid[axis]) / direction_grid[axis];
            t_enter = max(t_enter, min(t0, t1));
            t_exit = min(t_exit, max(t0, t1));
        }
    }
    if (!(t_enter < t_exit)) {
        return dda;
    }

    let point_grid = origin_grid + direction_grid * t_enter;
    dda.t_min = t_enter;
    dda.t_max = t_exit;
    dda.valid = 1u;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let resolution = i32(grid.majorant_resolution[axis]);
        dda.voxel[axis] = clamp(i32(point_grid[axis] * f32(resolution)), 0, resolution - 1);
        if (direction_grid[axis] > 0.0) {
            let next_position = f32(dda.voxel[axis] + 1) / f32(resolution);
            dda.next_crossing_t[axis] = t_enter + (next_position - point_grid[axis]) / direction_grid[axis];
            dda.delta_t[axis] = 1.0 / (abs(direction_grid[axis]) * f32(resolution));
            dda.step[axis] = 1;
            dda.voxel_limit[axis] = resolution;
        } else if (direction_grid[axis] < 0.0) {
            let next_position = f32(dda.voxel[axis]) / f32(resolution);
            dda.next_crossing_t[axis] = t_enter + (next_position - point_grid[axis]) / direction_grid[axis];
            dda.delta_t[axis] = 1.0 / (abs(direction_grid[axis]) * f32(resolution));
            dda.step[axis] = -1;
            dda.voxel_limit[axis] = -1;
        } else {
            dda.voxel_limit[axis] = dda.voxel[axis];
        }
    }
    return dda;
}

fn uniform_grid_dda_next(
    grid: VolumeMediumRecord,
    sigma_t: vec4<f32>,
    dda: ptr<function, UniformGridDda>,
) -> UniformGridMajorantSegment {
    var segment: UniformGridMajorantSegment;
    segment.t_min = (*dda).t_min;
    segment.t_max = (*dda).t_min;
    segment.sigma_maj = vec4<f32>(0.0);
    if ((*dda).valid == 0u || (*dda).t_min >= (*dda).t_max) {
        return segment;
    }
    let crossings = (*dda).next_crossing_t;
    let bits = (select(0u, 1u, crossings.x < crossings.y) << 2u)
        | (select(0u, 1u, crossings.x < crossings.z) << 1u)
        | select(0u, 1u, crossings.y < crossings.z);
    let cmp_to_axis = array<u32, 8>(2u, 1u, 2u, 1u, 2u, 2u, 0u, 0u);
    let axis = cmp_to_axis[bits];
    let t_voxel_exit = min((*dda).t_max, crossings[axis]);
    let resolution = grid.majorant_resolution.xyz;
    let index = u32((*dda).voxel.z) * resolution.y * resolution.x
        + u32((*dda).voxel.y) * resolution.x + u32((*dda).voxel.x);
    segment.t_max = t_voxel_exit;
    segment.sigma_maj = sigma_t * volume_data[grid.majorant_offset_count.x + index];

    (*dda).t_min = t_voxel_exit;
    if (crossings[axis] > (*dda).t_max) {
        (*dda).t_min = (*dda).t_max;
    }
    (*dda).voxel[axis] += (*dda).step[axis];
    if ((*dda).voxel[axis] == (*dda).voxel_limit[axis]) {
        (*dda).t_min = (*dda).t_max;
    }
    (*dda).next_crossing_t[axis] += (*dda).delta_t[axis];
    return segment;
}

// pbrt-v4 NanoVDBMedium::Le; nonpositive Lescale follows r4 CPU.
fn nanovdb_le(grid: VolumeMediumRecord, position_medium: vec3<f32>, lambda: vec4<f32>) -> vec4<f32> {
    let present = grid.temperature_grid.ranges[1].z;
    if (present > 1u) { set_render_error(); return vec4<f32>(0.0); }
    if (present == 0u || grid.temperature_parameters.z <= 0.0) { return vec4<f32>(0.0); }
    let sampled_temperature = sparse_grid_sample(grid.temperature_grid, position_medium);
    let temperature = (sampled_temperature - grid.temperature_parameters.x)
        * grid.temperature_parameters.y;
    if (temperature != temperature || abs(temperature) > RAY_T_MAX) {
        set_render_error();
        return vec4<f32>(0.0);
    }
    if (temperature <= 100.0) { return vec4<f32>(0.0); }
    let le = grid.temperature_parameters.z * blackbody_spectrum(temperature, lambda);
    if (any(le != le) || any(abs(le) > vec4<f32>(RAY_T_MAX))) {
        set_render_error();
        return vec4<f32>(0.0);
    }
    return le;
}

// pbrt-v4 WavefrontPathIntegrator::SampleMediumInteraction emission estimator.
fn medium_emission_contribution(
    beta: vec4<f32>, r_u: vec4<f32>, sigma_maj: vec4<f32>,
    transmittance_ratio: vec4<f32>, sigma_a: vec4<f32>, le: vec4<f32>,
) -> vec4<f32> {
    if (all(le == vec4<f32>(0.0))) { return vec4<f32>(0.0); }
    let emission_probability = sampled_spectrum_average(
        r_u * (sigma_maj / sigma_maj.x) * transmittance_ratio,
    );
    if (emission_probability != emission_probability || abs(emission_probability) > RAY_T_MAX) {
        set_render_error();
        return vec4<f32>(0.0);
    }
    if (emission_probability <= 0.0) { return vec4<f32>(0.0); }
    let contribution = beta * (transmittance_ratio / sigma_maj.x)
        * sigma_a * le / emission_probability;
    if (any(contribution != contribution) || any(abs(contribution) > vec4<f32>(RAY_T_MAX))) {
        set_render_error();
        return vec4<f32>(0.0);
    }
    return contribution;
}
