#![cfg(feature = "webgpu")]

use bytemuck::{cast_slice, Zeroable};
use pbrt_r4::gpu::flat::flatten::nanovdb::{create_nanovdb, world_to_index_f};
use pbrt_r4::gpu::flat::{identity_transform, Medium, MediumData};
use pbrt_r4::gpu::webgpu::abi::{
    RenderError, SparseGridEntryRecord, SparseGridRootRecord, VolumeMediumRecord,
};
use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::scene::medium::convert_media;
use pbrt_r4::gpu::webgpu::shader::compose_source;
use pbrt_r4::gpu::webgpu::stages::RequiredLimits;
use pbrt_r4::paramdict::ParameterDictionary;
use wgpu::util::DeviceExt;

const PROBE_SHADER: &str = r#"
@group(0) @binding(79) var<storage, read_write> output_values: array<f32>;

@compute @workgroup_size(1)
fn sparse_grid_probe() {
    let grid = volume_media[0];
    output_values[0] = sparse_grid_get_value(grid, vec3<i32>(-1, 0, 0));
    output_values[1] = sparse_grid_get_value(grid, vec3<i32>(1, 2, 3));
    output_values[2] = sparse_grid_get_value(grid, vec3<i32>(8, 0, 0));
    output_values[3] = sparse_grid_get_value(grid, vec3<i32>(128, 0, 0));
    output_values[4] = sparse_grid_get_value(grid, vec3<i32>(4096, 0, 0));
    output_values[5] = sparse_grid_density(grid, vec3<f32>(3.0, -3.0, 2.0));
    var dda = uniform_grid_dda_init(
        grid, vec3<f32>(-1.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), 0.0, 4.0,
    );
    let segment = uniform_grid_dda_next(grid, vec4<f32>(1.0), &dda);
    output_values[6] = segment.t_min;
    output_values[7] = segment.t_max;
    output_values[8] = segment.sigma_maj.x;
}
"#;

const ASSET_PROBE_SHADER: &str = r#"
@group(0) @binding(79) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(80) var<storage, read> probe_points: array<vec4<f32>>;

@compute @workgroup_size(64)
fn sparse_grid_asset_probe(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= arrayLength(&probe_points)) { return; }
    output_values[global_id.x] = sparse_grid_density(
        volume_media[0], probe_points[global_id.x].xyz,
    );
}
"#;

#[test]
#[ignore = "requires a WebGPU adapter"]
fn sparse_grid_lookup_and_interpolation_match_flat_values() {
    let context = Context::new(
        RequiredLimits {
            storage_buffers_per_shader_stage: 6,
            uniform_buffers_per_shader_stage: 0,
            buffers_and_acceleration_structures_per_shader_stage: 6,
            bind_groups: 1,
        },
        0,
        0,
    )
    .unwrap();
    let device = &context.device;

    let mut grid = VolumeMediumRecord::zeroed();
    grid.bounds_min = [0.0; 4];
    grid.bounds_max = [2.0; 4];
    grid.majorant_resolution = [1, 1, 1, 0];
    grid.majorant_offset_count = [512, 1, 0, 0];
    grid.sparse_grid_ranges = [[0, 2, 0, 32768 + 4096], [0, 512, 0, 0]];
    grid.sparse_grid_background = [1.0, 0.0, 0.0, 0.0];
    grid.grid_inverse_map = [
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
    ];
    grid.grid_translation = [2.0, -4.0, 1.0, 0.0];
    grid.medium_from_world = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let volume_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("NanoVDB sparse-grid probe volume record"),
        contents: bytemuck::bytes_of(&grid),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let roots = [
        SparseGridRootRecord {
            block: [-1, 0, 0],
            child_offset: u32::MAX,
            value: 9.0,
            padding: [0; 3],
        },
        SparseGridRootRecord {
            block: [0; 3],
            child_offset: 0,
            value: 0.0,
            padding: [0; 3],
        },
    ];
    let roots_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("NanoVDB sparse-grid probe roots"),
        contents: cast_slice(&roots),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let mut entries = vec![
        SparseGridEntryRecord {
            child_offset: u32::MAX,
            value: 2.0,
        };
        32768 + 4096
    ];
    entries[0] = SparseGridEntryRecord {
        child_offset: 32768,
        value: 0.0,
    };
    entries[32768] = SparseGridEntryRecord {
        child_offset: 0,
        value: 0.0,
    };
    entries[32768 + 256] = SparseGridEntryRecord {
        child_offset: u32::MAX,
        value: 3.0,
    };
    let entries_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("NanoVDB sparse-grid probe entries"),
        contents: cast_slice(&entries),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let leaf_values = (0..512)
        .map(|index| {
            let x = (index / 64) as f32;
            let y = ((index / 8) % 8) as f32;
            let z = (index % 8) as f32;
            x + 2.0 * y + 3.0 * z
        })
        .collect::<Vec<_>>();
    let mut volume_values = leaf_values;
    volume_values.push(2.0);
    let volume_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("NanoVDB sparse-grid probe leaf values"),
        contents: cast_slice(&volume_values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let render_error = RenderError::zeroed();
    let render_error_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("NanoVDB sparse-grid probe render error"),
        contents: bytemuck::bytes_of(&render_error),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let error_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("NanoVDB sparse-grid probe error readback"),
        size: std::mem::size_of::<RenderError>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("NanoVDB sparse-grid probe output"),
        size: 9 * std::mem::size_of::<f32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("NanoVDB sparse-grid probe readback"),
        size: 9 * std::mem::size_of::<f32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("NanoVDB sparse-grid probe layout"),
        entries: &[
            storage_layout(11, false),
            storage_layout(75, true),
            storage_layout(76, true),
            storage_layout(77, true),
            storage_layout(78, true),
            storage_layout(79, false),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("NanoVDB sparse-grid probe bind group"),
        layout: &layout,
        entries: &[
            binding(11, &render_error_buffer),
            binding(75, &volume_buffer),
            binding(76, &volume_data_buffer),
            binding(77, &roots_buffer),
            binding(78, &entries_buffer),
            binding(79, &output_buffer),
        ],
    });
    let source = compose_source(PROBE_SHADER);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("NanoVDB sparse-grid numerical probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("NanoVDB sparse-grid probe pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("NanoVDB sparse-grid probe pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("sparse_grid_probe"),
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
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, 36);
    encoder.copy_buffer_to_buffer(
        &render_error_buffer,
        0,
        &error_readback,
        0,
        std::mem::size_of::<RenderError>() as u64,
    );
    context.queue.submit(Some(encoder.finish()));
    let values = read_f32_buffer(device, &readback);

    assert_eq!(&values[..6], &[9.0, 14.0, 3.0, 2.0, 1.0, 3.0]);
    assert!((values[6] - 1.0).abs() < 1.0e-5);
    assert!((values[7] - 3.0).abs() < 1.0e-5);
    assert_eq!(values[8], 2.0);
    let render_error = read_u32_buffer(device, &error_readback);
    assert_eq!(
        render_error[0], 0,
        "valid sparse-grid probe raised render error"
    );

    let mut invalid_roots = roots;
    invalid_roots[1].child_offset = 4097;
    let invalid_roots_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("NanoVDB invalid root child probe"),
        contents: cast_slice(&invalid_roots),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let invalid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("NanoVDB invalid root child bind group"),
        layout: &layout,
        entries: &[
            binding(11, &render_error_buffer),
            binding(75, &volume_buffer),
            binding(76, &volume_data_buffer),
            binding(77, &invalid_roots_buffer),
            binding(78, &entries_buffer),
            binding(79, &output_buffer),
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &invalid_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &render_error_buffer,
        0,
        &error_readback,
        0,
        std::mem::size_of::<RenderError>() as u64,
    );
    context.queue.submit(Some(encoder.finish()));
    assert_eq!(
        read_u32_buffer(device, &error_readback)[0],
        1,
        "out-of-range root child must raise render error"
    );

    let mut invalid_entries = entries;
    invalid_entries[32768].child_offset = 500;
    let invalid_entries_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("NanoVDB invalid leaf child probe"),
        contents: cast_slice(&invalid_entries),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let valid_roots_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("NanoVDB valid roots for invalid leaf probe"),
        contents: cast_slice(&roots),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let cleared_render_error_buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NanoVDB invalid leaf probe render error"),
            contents: bytemuck::bytes_of(&RenderError::zeroed()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let invalid_leaf_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("NanoVDB invalid leaf child bind group"),
        layout: &layout,
        entries: &[
            binding(11, &cleared_render_error_buffer),
            binding(75, &volume_buffer),
            binding(76, &volume_data_buffer),
            binding(77, &valid_roots_buffer),
            binding(78, &invalid_entries_buffer),
            binding(79, &output_buffer),
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &invalid_leaf_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &cleared_render_error_buffer,
        0,
        &error_readback,
        0,
        std::mem::size_of::<RenderError>() as u64,
    );
    context.queue.submit(Some(encoder.finish()));
    assert_eq!(
        read_u32_buffer(device, &error_readback)[0],
        1,
        "out-of-range leaf child must raise render error"
    );
}

#[test]
#[ignore = "requires a WebGPU adapter and NANOVDB_ASSET"]
fn actual_asset_gpu_samples_match_flat_grid_samples() {
    let asset = std::env::var("NANOVDB_ASSET").expect("NANOVDB_ASSET");
    let mut params = ParameterDictionary::new();
    params.add_string("filename", &asset);
    let nanovdb = create_nanovdb(&params).unwrap();
    let medium = Medium {
        name: "NanoVDB GPU probe".to_owned(),
        kind: "nanovdb".to_owned(),
        sigma_a: 0,
        sigma_s: 0,
        le: 0,
        g: 0.0,
        transform: identity_transform(),
        data: MediumData::NanoVDB(nanovdb.clone()),
    };
    let converted = convert_media(&[medium], 1).unwrap();
    let volume = converted.volumes[0];
    let sample_count = 729usize;
    let mut points = Vec::with_capacity(sample_count);
    let mut expected = Vec::with_capacity(sample_count);
    for z in 0..9 {
        for y in 0..9 {
            for x in 0..9 {
                let sample = [x, y, z];
                let mut index = [0.0; 3];
                for axis in 0..3 {
                    let t = sample[axis] as f32 / 8.0;
                    let lower = nanovdb.index_bbox_min[axis] as f32 - 1.25;
                    let upper = nanovdb.index_bbox_max[axis] as f32 + 1.25;
                    index[axis] = lower + t * (upper - lower);
                }
                let mut point_medium = [0.0; 3];
                for axis in 0..3 {
                    let scale = nanovdb.inverse_map_f[axis][axis];
                    point_medium[axis] = nanovdb.translation_f[axis] + index[axis] / scale;
                }
                let mapped_index = world_to_index_f(&nanovdb, point_medium);
                expected.push(sample_sparse_grid(&nanovdb, mapped_index));
                points.push([point_medium[0], point_medium[1], point_medium[2], 0.0]);
            }
        }
    }

    let context = Context::new(
        RequiredLimits {
            storage_buffers_per_shader_stage: 7,
            uniform_buffers_per_shader_stage: 0,
            buffers_and_acceleration_structures_per_shader_stage: 7,
            bind_groups: 1,
        },
        0,
        0,
    )
    .unwrap();
    let device = &context.device;
    assert!(
        converted.volume_data.len() * 4 <= device.limits().max_storage_buffer_binding_size as usize
    );
    let volume_buffer = storage_buffer(device, "NanoVDB asset volume record", &[volume]);
    let roots_buffer = storage_buffer(
        device,
        "NanoVDB asset roots",
        converted.sparse_grid_roots.as_slice(),
    );
    let entries_buffer = storage_buffer(
        device,
        "NanoVDB asset entries",
        converted.sparse_grid_entries.as_slice(),
    );
    let volume_data_buffer = storage_buffer(
        device,
        "NanoVDB asset volume data",
        converted.volume_data.as_slice(),
    );
    let render_error = RenderError::zeroed();
    let render_error_buffer = storage_buffer(
        device,
        "NanoVDB asset render error",
        std::slice::from_ref(&render_error),
    );
    let points_buffer = storage_buffer(device, "NanoVDB asset sample points", points.as_slice());
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("NanoVDB asset sample output"),
        size: (sample_count * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("NanoVDB asset sample readback"),
        size: (sample_count * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let error_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("NanoVDB asset error readback"),
        size: std::mem::size_of::<RenderError>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("NanoVDB asset probe layout"),
        entries: &[
            storage_layout(11, false),
            storage_layout(75, true),
            storage_layout(76, true),
            storage_layout(77, true),
            storage_layout(78, true),
            storage_layout(79, false),
            storage_layout(80, true),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("NanoVDB asset probe group"),
        layout: &layout,
        entries: &[
            binding(11, &render_error_buffer),
            binding(75, &volume_buffer),
            binding(76, &volume_data_buffer),
            binding(77, &roots_buffer),
            binding(78, &entries_buffer),
            binding(79, &output_buffer),
            binding(80, &points_buffer),
        ],
    });
    let source = compose_source(ASSET_PROBE_SHADER);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("NanoVDB asset sample probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("NanoVDB asset probe pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("NanoVDB asset probe pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("sparse_grid_asset_probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(sample_count.div_ceil(64) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, (sample_count * 4) as u64);
    encoder.copy_buffer_to_buffer(
        &render_error_buffer,
        0,
        &error_readback,
        0,
        std::mem::size_of::<RenderError>() as u64,
    );
    context.queue.submit(Some(encoder.finish()));
    let actual = read_f32_buffer(device, &readback);
    let render_error = read_u32_buffer(device, &error_readback);
    assert_eq!(render_error[0], 0);
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-6 * expected.abs().max(1.0),
            "GPU {actual} differs from Flat {expected}"
        );
    }
}

fn sample_sparse_grid(grid: &pbrt_r4::gpu::flat::NanoVDBMedium, point: [f32; 3]) -> f32 {
    let base = point.map(|coordinate| coordinate.floor() as i32);
    let fraction = std::array::from_fn::<_, 3, _>(|axis| point[axis] - base[axis] as f32);
    let value = |x: i32, y: i32, z: i32| grid.density.get_value([x, y, z]).unwrap();
    let v000 = value(base[0], base[1], base[2]);
    let v001 = value(base[0], base[1], base[2] + 1);
    let v010 = value(base[0], base[1] + 1, base[2]);
    let v011 = value(base[0], base[1] + 1, base[2] + 1);
    let v100 = value(base[0] + 1, base[1], base[2]);
    let v101 = value(base[0] + 1, base[1], base[2] + 1);
    let v110 = value(base[0] + 1, base[1] + 1, base[2]);
    let v111 = value(base[0] + 1, base[1] + 1, base[2] + 1);
    let z00 = v000 + fraction[2] * (v001 - v000);
    let z01 = v010 + fraction[2] * (v011 - v010);
    let z10 = v100 + fraction[2] * (v101 - v100);
    let z11 = v110 + fraction[2] * (v111 - v110);
    let y0 = z00 + fraction[1] * (z01 - z00);
    let y1 = z10 + fraction[1] * (z11 - z10);
    y0 + fraction[0] * (y1 - y0)
}

fn storage_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    values: &[T],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
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

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn read_f32_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
    bytemuck::cast_slice(&read_bytes(device, buffer)).to_vec()
}

fn read_u32_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u32> {
    bytemuck::cast_slice(&read_bytes(device, buffer)).to_vec()
}

fn read_bytes(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let bytes = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    bytes
}
