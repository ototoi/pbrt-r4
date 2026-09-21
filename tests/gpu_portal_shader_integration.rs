use std::f32::consts::PI;

use pbrt_r4::gpu::webgpu::abi::{PortalDistributionTexel, PortalImageInfiniteRecord};
use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::shader::compose_source;
use pbrt_r4::gpu::webgpu::stages::RequiredLimits;
use wgpu::util::DeviceExt;

const RESULT_FLOATS: usize = 40;

const PORTAL_TEST_SHADER: &str = r#"
@group(0) @binding(26) var<storage, read_write> output_values: array<vec4<f32>, 10>;

@compute @workgroup_size(1)
fn portal_test() {
    let portal = portal_infinite_lights[0];
    let mapped = portal_image_from_render(portal, vec3<f32>(0.0, 0.0, 1.0));
    let back = portal_image_from_render(portal, vec3<f32>(0.0, 0.0, -1.0));
    let corner = portal_image_from_render(portal, normalize(vec3<f32>(-1.0, -1.0, 1.0)));
    let direction = portal_render_from_image(portal, vec2<f32>(0.5));
    let bounds = portal_image_bounds(portal, vec3<f32>(0.0));
    let sampled = sample_portal_distribution(portal, vec2<f32>(0.5), bounds);
    let pdf = portal_distribution_pdf(portal, vec2<f32>(0.5), bounds);
    let one = portal_infinite_lights[1];
    let one_bounds = portal_image_bounds(one, vec3<f32>(0.0));
    let one_sample = sample_portal_distribution(one, vec2<f32>(0.5), one_bounds);
    let zero = portal_infinite_lights[2];
    let zero_bounds = portal_image_bounds(zero, vec3<f32>(0.0));
    let zero_sample = sample_portal_distribution(zero, vec2<f32>(0.5), zero_bounds);
    let non_finite = portal_infinite_lights[3];
    let non_finite_bounds = portal_image_bounds(non_finite, vec3<f32>(0.0));
    let non_finite_sample = sample_portal_distribution(
        non_finite, vec2<f32>(0.5), non_finite_bounds,
    );
    output_values[0] = vec4<f32>(mapped.uv, mapped.duv_dw, f32(mapped.valid));
    output_values[1] = vec4<f32>(direction.wi, direction.duv_dw);
    output_values[2] = vec4<f32>(bounds.min, bounds.max);
    output_values[3] = vec4<f32>(sampled.uv, sampled.pdf, pdf);
    output_values[4] = vec4<f32>(back.uv, back.duv_dw, f32(back.valid));
    output_values[5] = vec4<f32>(corner.uv, corner.duv_dw, f32(corner.valid));
    output_values[6] = vec4<f32>(one_sample.uv, one_sample.pdf, f32(one_sample.valid));
    output_values[7] = vec4<f32>(zero_sample.uv, zero_sample.pdf, f32(zero_sample.valid));
    output_values[8] = vec4<f32>(
        portal_distribution_pdf(portal, vec2<f32>(0.1), bounds),
        non_finite_sample.pdf,
        f32(non_finite_sample.valid),
        0.0,
    );
    let rounded_cdf = portal_infinite_lights[4];
    let rounded_bounds = PortalBoundsResult(vec2<f32>(0.5, 0.0), vec2<f32>(1.0), 1u);
    let rounded_sample = portal_sample_marginal_x(rounded_cdf, 0.5, rounded_bounds, 1.0);
    output_values[9] = vec4<f32>(rounded_sample.value, f32(rounded_sample.valid), 0.0, 0.0);
}
"#;

#[test]
#[ignore = "requires a WebGPU adapter"]
fn portal_shader_matches_v4_mapping_and_windowed_distribution() {
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
    let portal = PortalImageInfiniteRecord {
        portal: [
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, 1.0],
        ],
        world_to_portal: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        distribution_offset: 0,
        width: 3,
        height: 3,
        reserved: 0,
    };
    let mut portals = [portal; 5];
    portals[1].distribution_offset = 9;
    portals[1].width = 1;
    portals[1].height = 1;
    portals[2].distribution_offset = 10;
    portals[2].width = 1;
    portals[2].height = 1;
    portals[3].distribution_offset = 11;
    portals[3].width = 1;
    portals[3].height = 1;
    portals[4].distribution_offset = 12;
    portals[4].width = 2;
    portals[4].height = 1;
    let distribution = [
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 1.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 2.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 3.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 2.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 4.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 6.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 3.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 6.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 9.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 1.0,
        },
        PortalDistributionTexel {
            function: 0.0,
            summed_area: 0.0,
        },
        PortalDistributionTexel {
            function: f32::NAN,
            summed_area: f32::NAN,
        },
        PortalDistributionTexel {
            function: 1.0e20,
            summed_area: 1.0e20,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 1.0e20,
        },
    ];
    let portal_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("portal test record"),
        contents: bytemuck::cast_slice(&portals),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let distribution_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("portal test distribution"),
        contents: bytemuck::cast_slice(&distribution),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("portal test output"),
        size: (RESULT_FLOATS * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("portal test readback"),
        size: (RESULT_FLOATS * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("portal test layout"),
        entries: &[
            buffer_layout(24, true),
            buffer_layout(25, true),
            buffer_layout(26, false),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("portal test bind group"),
        layout: &layout,
        entries: &[
            binding(24, portal_buffer.as_entire_binding()),
            binding(25, distribution_buffer.as_entire_binding()),
            binding(26, output.as_entire_binding()),
        ],
    });
    let source = compose_source(PORTAL_TEST_SHADER);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("portal numerical test"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("portal numerical test"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("portal numerical test"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("portal_test"),
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
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, (RESULT_FLOATS * 4) as u64);
    context.queue.submit(Some(encoder.finish()));
    let actual = read_f32_buffer(device, &readback);

    for (index, expected) in [
        // Forward mapping.
        0.5,
        0.5,
        PI.powi(2),
        1.0,
        // Inverse mapping.
        0.0,
        0.0,
        1.0,
        PI.powi(2),
        // Projected bounds.
        0.25,
        0.25,
        0.75,
        0.75,
        // 3x3 uniform sample and PDF.
        0.5,
        0.5,
        4.0,
        4.0,
        // Back-facing mapping.
        0.0,
        0.0,
        0.0,
        0.0,
        // Portal corner mapping.
        0.25,
        0.25,
        7.5976253,
        1.0,
        // 1x1 distribution.
        0.5,
        0.5,
        4.0,
        1.0,
        // Zero distribution.
        0.0,
        0.0,
        0.0,
        0.0,
        // PDF remains defined outside the window; non-finite data is invalid.
        4.0,
        0.0,
        0.0,
        0.0,
        // Equal rounded CDF endpoints produce an invalid axis sample.
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    .iter()
    .enumerate()
    {
        assert!((actual[index] - expected).abs() < 2e-4, "value {index}");
    }
}

fn buffer_layout(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn binding(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
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
