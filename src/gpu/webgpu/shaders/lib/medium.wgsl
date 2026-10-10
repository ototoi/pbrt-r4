fn uniform_grid_density_value(grid: UniformGridMediumRecord, coordinate: vec3<i32>) -> f32 {
    let resolution = vec3<i32>(grid.resolution.xyz);
    if (any(coordinate < vec3<i32>(0)) || any(coordinate >= resolution)) {
        return 0.0;
    }
    let index = u32(coordinate.z) * grid.resolution.y * grid.resolution.x
        + u32(coordinate.y) * grid.resolution.x + u32(coordinate.x);
    return volume_data[grid.density_offset_count.x + index];
}

fn uniform_grid_density(grid: UniformGridMediumRecord, position_medium: vec3<f32>) -> f32 {
    let bounds_diagonal = grid.bounds_max.xyz - grid.bounds_min.xyz;
    let unit = (position_medium - grid.bounds_min.xyz) / bounds_diagonal;
    let sample_position = unit * vec3<f32>(grid.resolution.xyz) - vec3<f32>(0.5);
    let base = vec3<i32>(floor(sample_position));
    let delta = sample_position - vec3<f32>(base);
    let d00 = mix(
        uniform_grid_density_value(grid, base),
        uniform_grid_density_value(grid, base + vec3<i32>(1, 0, 0)), delta.x,
    );
    let d10 = mix(
        uniform_grid_density_value(grid, base + vec3<i32>(0, 1, 0)),
        uniform_grid_density_value(grid, base + vec3<i32>(1, 1, 0)), delta.x,
    );
    let d01 = mix(
        uniform_grid_density_value(grid, base + vec3<i32>(0, 0, 1)),
        uniform_grid_density_value(grid, base + vec3<i32>(1, 0, 1)), delta.x,
    );
    let d11 = mix(
        uniform_grid_density_value(grid, base + vec3<i32>(0, 1, 1)),
        uniform_grid_density_value(grid, base + vec3<i32>(1, 1, 1)), delta.x,
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
    grid: UniformGridMediumRecord,
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
    grid: UniformGridMediumRecord,
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
