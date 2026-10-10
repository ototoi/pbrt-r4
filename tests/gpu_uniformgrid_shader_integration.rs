use pbrt_r4::gpu::webgpu::abi::UniformGridMediumRecord;
use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::shader::compose_source;
use pbrt_r4::gpu::webgpu::stages::RequiredLimits;
use wgpu::util::DeviceExt;

const RESULT_VECTORS: u64 = 33;

const PROBE_SHADER: &str = r#"
@group(0) @binding(77) var<storage, read_write> output_values: array<vec4<f32>>;

@compute @workgroup_size(1)
fn uniform_grid_probe() {
    let grid = uniform_grid_media[0];
    output_values[0] = vec4<f32>(
        uniform_grid_density(grid, vec3<f32>(1.0, 0.5, 0.5)),
        uniform_grid_density(grid, vec3<f32>(0.0, 0.5, 0.5)),
        uniform_grid_density(grid, vec3<f32>(2.0, 0.5, 0.5)),
        0.0,
    );

    var dda = uniform_grid_dda_init(
        grid, vec3<f32>(-1.0, 0.5, 0.5), vec3<f32>(1.0, 0.0, 0.0), 0.0, 4.0,
    );
    let first = uniform_grid_dda_next(grid, vec4<f32>(1.0), &dda);
    let second = uniform_grid_dda_next(grid, vec4<f32>(1.0), &dda);
    output_values[1] = vec4<f32>(first.t_min, first.t_max, first.sigma_maj.x, 0.0);
    output_values[2] = vec4<f32>(second.t_min, second.t_max, second.sigma_maj.x, 0.0);

    var reverse_dda = uniform_grid_dda_init(
        grid, vec3<f32>(3.0, 0.5, 0.5), vec3<f32>(-1.0, 0.0, 0.0), 0.0, 4.0,
    );
    let reverse_first = uniform_grid_dda_next(grid, vec4<f32>(1.0), &reverse_dda);
    let reverse_second = uniform_grid_dda_next(grid, vec4<f32>(1.0), &reverse_dda);
    output_values[3] = vec4<f32>(
        reverse_first.t_min, reverse_first.t_max, reverse_first.sigma_maj.x, 0.0,
    );
    output_values[4] = vec4<f32>(
        reverse_second.t_min, reverse_second.t_max, reverse_second.sigma_maj.x, 0.0,
    );

    var parallel_dda = uniform_grid_dda_init(
        grid, vec3<f32>(0.5, -1.0, 0.5), vec3<f32>(-0.0, 1.0, 0.0), 0.0, 4.0,
    );
    let parallel = uniform_grid_dda_next(grid, vec4<f32>(1.0), &parallel_dda);
    output_values[5] = vec4<f32>(parallel.t_min, parallel.t_max, parallel.sigma_maj.x, 0.0);

    var transformed_grid = grid;
    transformed_grid.medium_from_world = mat4x4<f32>(
        vec4<f32>(0.0, -1.0 / 3.0, 0.0, 0.0),
        vec4<f32>(0.5, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.25, 0.0),
        vec4<f32>(1.0, 1.0, -1.25, 1.0),
    );
    var transformed_dda = uniform_grid_dda_init(
        transformed_grid, vec3<f32>(1.5, -4.0, 7.0),
        vec3<f32>(0.0, 1.0, 0.0), 0.0, 8.0,
    );
    let transformed_first = uniform_grid_dda_next(grid, vec4<f32>(1.0), &transformed_dda);
    let transformed_second = uniform_grid_dda_next(grid, vec4<f32>(1.0), &transformed_dda);
    output_values[6] = vec4<f32>(
        transformed_first.t_min, transformed_first.t_max, transformed_first.sigma_maj.x, 0.0,
    );
    output_values[7] = vec4<f32>(
        transformed_second.t_min, transformed_second.t_max, transformed_second.sigma_maj.x, 0.0,
    );
    var inside_dda = uniform_grid_dda_init(
        grid, vec3<f32>(0.5), vec3<f32>(1.0, 0.0, 0.0), 0.0, 0.25,
    );
    let inside = uniform_grid_dda_next(grid, vec4<f32>(1.0), &inside_dda);
    output_values[8] = vec4<f32>(inside.t_min, inside.t_max, inside.sigma_maj.x, 0.0);
    let miss = uniform_grid_dda_init(
        grid, vec3<f32>(-1.0, 2.0, 0.5), vec3<f32>(1.0, 0.0, 0.0), 0.0, 4.0,
    );
    output_values[9] = vec4<f32>(f32(miss.valid), 0.0, 0.0, 0.0);

    var corner_grid = grid;
    corner_grid.bounds_max = vec4<f32>(1.0, 1.0, 1.0, 0.0);
    corner_grid.majorant_resolution = vec4<u32>(2u, 2u, 2u, 0u);
    corner_grid.majorant_offset_count = vec4<u32>(4u, 8u, 0u, 0u);
    var corner_dda = uniform_grid_dda_init(
        corner_grid, vec3<f32>(-1.0), normalize(vec3<f32>(1.0)), 0.0, 10.0,
    );
    for (var i = 0u; i < 4u; i += 1u) {
        let segment = uniform_grid_dda_next(corner_grid, vec4<f32>(1.0), &corner_dda);
        output_values[10u + i] = vec4<f32>(
            segment.t_min, segment.t_max, segment.sigma_maj.x, 0.0,
        );
    }
    let exhausted = uniform_grid_dda_next(corner_grid, vec4<f32>(1.0), &corner_dda);
    output_values[14] = vec4<f32>(exhausted.t_max - exhausted.t_min, exhausted.sigma_maj.xyz);

    var translated_grid = grid;
    translated_grid.medium_from_world[3].x = -1000001.0;
    var translated_dda = uniform_grid_dda_init(
        translated_grid, vec3<f32>(1000000.0, 0.5, 0.5),
        vec3<f32>(1.0, 0.0, 0.0), 0.0, 4.0,
    );
    let translated = uniform_grid_dda_next(translated_grid, vec4<f32>(1.0), &translated_dda);
    output_values[15] = vec4<f32>(translated.t_min, translated.t_max, translated.sigma_maj.x, 0.0);
    var clipped_dda = uniform_grid_dda_init(
        translated_grid, vec3<f32>(1000000.0, 0.5, 0.5),
        vec3<f32>(1.0, 0.0, 0.0), 0.0, 1.5,
    );
    let clipped = uniform_grid_dda_next(translated_grid, vec4<f32>(1.0), &clipped_dda);
    output_values[16] = vec4<f32>(clipped.t_min, clipped.t_max, clipped.sigma_maj.x, 0.0);

    for (var axis = 0u; axis < 3u; axis += 1u) {
        var origin = vec3<f32>(0.25);
        var direction = vec3<f32>(0.0);
        origin[axis] = -1.0;
        direction[axis] = 1.0;
        var positive = uniform_grid_dda_init(corner_grid, origin, direction, 0.0, 4.0);
        let positive_first = uniform_grid_dda_next(corner_grid, vec4<f32>(1.0), &positive);
        let positive_second = uniform_grid_dda_next(corner_grid, vec4<f32>(1.0), &positive);
        output_values[17u + 4u * axis] = vec4<f32>(
            positive_first.t_min, positive_first.t_max, positive_first.sigma_maj.x, 0.0,
        );
        output_values[18u + 4u * axis] = vec4<f32>(
            positive_second.t_min, positive_second.t_max, positive_second.sigma_maj.x, 0.0,
        );
        origin[axis] = 2.0;
        direction[axis] = -1.0;
        var negative = uniform_grid_dda_init(corner_grid, origin, direction, 0.0, 4.0);
        let negative_first = uniform_grid_dda_next(corner_grid, vec4<f32>(1.0), &negative);
        let negative_second = uniform_grid_dda_next(corner_grid, vec4<f32>(1.0), &negative);
        output_values[19u + 4u * axis] = vec4<f32>(
            negative_first.t_min, negative_first.t_max, negative_first.sigma_maj.x, 0.0,
        );
        output_values[20u + 4u * axis] = vec4<f32>(
            negative_second.t_min, negative_second.t_max, negative_second.sigma_maj.x, 0.0,
        );
    }
    var hero_zero_dda = uniform_grid_dda_init(
        grid, vec3<f32>(-1.0, 0.5, 0.5), vec3<f32>(1.0, 0.0, 0.0), 0.0, 4.0,
    );
    let hero_zero = uniform_grid_dda_next(grid, vec4<f32>(0.0, 1.0, 2.0, 3.0), &hero_zero_dda);
    output_values[29] = hero_zero.sigma_maj;
    var empty_grid = grid;
    empty_grid.majorant_resolution = vec4<u32>(1u, 1u, 1u, 0u);
    empty_grid.majorant_offset_count = vec4<u32>(12u, 1u, 0u, 0u);
    var empty_dda = uniform_grid_dda_init(
        empty_grid, vec3<f32>(-1.0, 0.5, 0.5), vec3<f32>(1.0, 0.0, 0.0), 0.0, 4.0,
    );
    let empty_segment = uniform_grid_dda_next(empty_grid, vec4<f32>(1.0), &empty_dda);
    output_values[30] = empty_segment.sigma_maj;
    var one_grid = corner_grid;
    one_grid.resolution = vec4<u32>(1u, 1u, 1u, 0u);
    one_grid.density_offset_count = vec4<u32>(0u, 1u, 0u, 0u);
    output_values[31] = vec4<f32>(
        uniform_grid_density(one_grid, vec3<f32>(0.5)),
        uniform_grid_density(one_grid, vec3<f32>(0.0)),
        uniform_grid_density(one_grid, vec3<f32>(-0.5, 0.5, 0.5)), 0.0,
    );
    var two_grid = corner_grid;
    two_grid.resolution = vec4<u32>(2u, 2u, 2u, 0u);
    two_grid.density_offset_count = vec4<u32>(4u, 8u, 0u, 0u);
    output_values[32] = vec4<f32>(
        uniform_grid_density(two_grid, vec3<f32>(0.5)),
        uniform_grid_density(two_grid, vec3<f32>(0.0)),
        uniform_grid_density(two_grid, vec3<f32>(0.25, 0.25, 0.75)),
        uniform_grid_density(two_grid, vec3<f32>(1.0)),
    );
}
"#;

#[test]
#[ignore = "requires a WebGPU adapter"]
fn uniform_grid_lookup_and_dda_match_reference_values() {
    let context = Context::new(
        RequiredLimits {
            storage_buffers_per_shader_stage: 3,
            uniform_buffers_per_shader_stage: 0,
            buffers_and_acceleration_structures_per_shader_stage: 3,
            bind_groups: 1,
        },
        0,
        0,
    )
    .unwrap();
    let device = &context.device;

    // Two x voxels with density values 1 and 3. The 2x1x1 trilinear
    // lookup at (1, 0.5, 0.5) is 2; at either x bound it is half of the
    // boundary voxel because out-of-range samples are zero.
    let grid = UniformGridMediumRecord {
        bounds_min: [0.0, 0.0, 0.0, 0.0],
        bounds_max: [2.0, 1.0, 1.0, 0.0],
        resolution: [2, 1, 1, 0],
        majorant_resolution: [2, 1, 1, 0],
        density_offset_count: [0, 2, 0, 0],
        majorant_offset_count: [2, 2, 0, 0],
        medium_from_world: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };
    let grid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniform grid probe record"),
        contents: bytemuck::bytes_of(&grid),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // Density values followed by the two majorant values.
    let volume_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniform grid probe data"),
        contents: bytemuck::cast_slice(&[
            1.0_f32, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 0.0,
        ]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform grid probe output"),
        size: RESULT_VECTORS * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform grid probe readback"),
        size: RESULT_VECTORS * 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("uniform grid probe layout"),
        entries: &[
            storage_layout(75, true),
            storage_layout(76, true),
            storage_layout(77, false),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("uniform grid probe bind group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 75,
                resource: grid_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 76,
                resource: volume_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 77,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let source = compose_source(PROBE_SHADER);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("uniform grid numerical probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("uniform grid probe pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("uniform grid probe pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("uniform_grid_probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, RESULT_VECTORS * 16);
    context.queue.submit(Some(encoder.finish()));
    let values = read_f32_buffer(device, &readback);

    assert_close(values[0], 2.0);
    assert_close(values[1], 0.5);
    assert_close(values[2], 1.5);
    for (index, expected) in [
        [1.0, 2.0, 4.0],
        [2.0, 3.0, 5.0],
        [1.0, 2.0, 5.0],
        [2.0, 3.0, 4.0],
        [1.0, 2.0, 4.0],
        [2.0, 4.0, 4.0],
        [4.0, 6.0, 5.0],
        [0.0, 0.25, 4.0],
    ]
    .iter()
    .enumerate()
    {
        for lane in 0..3 {
            assert_close(values[(1 + index) * 4 + lane], expected[lane]);
        }
    }
    assert_eq!(values[36], 0.0);
    let entry = 3.0_f32.sqrt();
    let crossing = 1.5 * entry;
    for (index, expected) in [
        [entry, crossing, 10.0],
        [crossing, crossing, 14.0],
        [crossing, crossing, 16.0],
        [crossing, 2.0 * entry, 17.0],
    ]
    .iter()
    .enumerate()
    {
        for lane in 0..3 {
            assert_close(values[(10 + index) * 4 + lane], expected[lane]);
        }
    }
    assert_eq!(&values[56..60], &[0.0; 4]);
    let gamma3 = 3.0 * (f32::EPSILON / 2.0) / (1.0 - 3.0 * (f32::EPSILON / 2.0));
    let dt = gamma3 * 1000000.0;
    assert_close(values[60], 1.0 - dt);
    assert_close(values[61], 2.0 - dt);
    assert_close(values[64], 1.0 - dt);
    assert_close(values[65], 1.5 - dt);
    for (axis, high) in [11.0, 12.0, 14.0].iter().enumerate() {
        let base = 17 + 4 * axis;
        for (index, expected) in [
            [1.0, 1.5, 10.0],
            [1.5, 2.0, *high],
            [1.0, 1.5, *high],
            [1.5, 2.0, 10.0],
        ]
        .iter()
        .enumerate()
        {
            for lane in 0..3 {
                assert_close(values[(base + index) * 4 + lane], expected[lane]);
            }
        }
    }
    assert_eq!(&values[116..120], &[0.0, 4.0, 8.0, 12.0]);
    assert_eq!(&values[120..124], &[0.0; 4]);
    assert_eq!(&values[124..128], &[1.0, 0.125, 0.0, 0.0]);
    assert_eq!(&values[128..132], &[13.5, 1.25, 14.0, 2.125]);
}

fn storage_layout(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn read_f32_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    values
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 1e-6,
        "actual {actual}, expected {expected}"
    );
}
