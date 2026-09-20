use pbrt_r4::gpu::flat::{HaltonRandomization, RenderSettings, SamplerKind};
use pbrt_r4::gpu::webgpu::abi::{RenderError, ViewportUniform};
use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::sampler::SamplerResources;
use pbrt_r4::gpu::webgpu::shader::compose_source;
use pbrt_r4::gpu::webgpu::stages::RequiredLimits;
use pbrt_r4::samplers::{HaltonSampler, RandomizeStrategy};
use pbrt_r4::util::base::Point2i;
use wgpu::util::DeviceExt;

const RESULT_FLOATS: usize = 8;

#[test]
#[ignore = "requires a WebGPU adapter"]
fn halton_shader_matches_cpu_sampler() {
    let context = Context::new(
        RequiredLimits {
            storage_buffers_per_shader_stage: 2,
            uniform_buffers_per_shader_stage: 2,
            bind_groups: 1,
        },
        1,
        1,
    )
    .unwrap();

    compare_strategy(&context, HaltonRandomization::None, RandomizeStrategy::None);
    compare_strategy(
        &context,
        HaltonRandomization::PermuteDigits,
        RandomizeStrategy::PermuteDigits,
    );
}

fn compare_strategy(
    context: &Context,
    gpu_randomization: HaltonRandomization,
    cpu_randomization: RandomizeStrategy,
) {
    let resolution = [320, 180];
    let pixel = Point2i::new(37, 23);
    let pixel_index = pixel.y as u32 * resolution[0] + pixel.x as u32;
    let sample_index = 11;
    let seed = 17;
    let settings = RenderSettings {
        sampler_kind: SamplerKind::Halton,
        halton_randomization: gpu_randomization,
        samples_per_pixel: 16,
        max_depth: 5,
        seed,
        light_sampler: "bvh".to_string(),
        disable_wavelength_jitter: false,
    };
    let resources =
        SamplerResources::new(&context.device, &context.queue, &settings, resolution).unwrap();
    let viewport = ViewportUniform {
        width: resolution[0],
        height: resolution[1],
        sample_index,
        max_depth: 5,
        seed,
        disable_wavelength_jitter: 0,
        padding: [0; 2],
    };

    let mut cpu = HaltonSampler::new(
        settings.samples_per_pixel,
        Point2i::new(resolution[0] as i32, resolution[1] as i32),
        cpu_randomization,
        seed,
    );
    cpu.start_pixel(&pixel);
    cpu.start_pixel_sample(sample_index, 0);
    let wavelength = cpu.get_1d();
    let pixel_sample = cpu.get_pixel_2d();
    cpu.start_pixel_sample(sample_index, 6);
    let expected = [
        pixel_sample.x,
        pixel_sample.y,
        wavelength,
        cpu.get_1d(),
        cpu.get_1d(),
        cpu.get_1d(),
        cpu.get_1d(),
        cpu.get_1d(),
    ];

    let device = &context.device;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Halton test output"),
        size: (RESULT_FLOATS * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Halton test readback"),
        size: (RESULT_FLOATS * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let viewport_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Halton test viewport"),
        contents: bytemuck::bytes_of(&viewport),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let sampler_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Halton test params"),
        contents: bytemuck::bytes_of(&resources.uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let error_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Halton test error"),
        contents: bytemuck::bytes_of(&RenderError {
            value: 0,
            padding: [0; 3],
        }),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Halton test layout"),
        entries: &[
            buffer_layout(1, wgpu::BufferBindingType::Uniform),
            buffer_layout(9, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer_layout(11, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer_layout(21, wgpu::BufferBindingType::Uniform),
            wgpu::BindGroupLayoutEntry {
                binding: 49,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Halton test bindings"),
        layout: &layout,
        entries: &[
            binding(1, viewport_buffer.as_entire_binding()),
            binding(9, output.as_entire_binding()),
            binding(11, error_buffer.as_entire_binding()),
            binding(21, sampler_buffer.as_entire_binding()),
            wgpu::BindGroupEntry {
                binding: 49,
                resource: wgpu::BindingResource::TextureView(&resources.table_view),
            },
        ],
    });
    let source = compose_source(&format!(
        r#"
@compute @workgroup_size(1)
fn halton_test() {{
    let pixel = sampler_get_pixel_2d({pixel_index}u);
    framebuffer[0] = vec4<f32>(
        pixel,
        sampler_get_1d({pixel_index}u, 0u),
        sampler_get_1d({pixel_index}u, 6u),
    );
    framebuffer[1] = vec4<f32>(
        sampler_get_1d({pixel_index}u, 7u),
        sampler_get_1d({pixel_index}u, 8u),
        sampler_get_1d({pixel_index}u, 9u),
        sampler_get_1d({pixel_index}u, 10u),
    );
}}
"#
    ));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Halton numerical test"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Halton numerical test"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Halton numerical test"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("halton_test"),
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
    for (index, expected) in expected.iter().enumerate() {
        assert!(
            (actual[index] - expected).abs() <= 2e-6,
            "strategy={gpu_randomization:?}, value={index}: GPU {} != CPU {expected}",
            actual[index]
        );
    }
}

fn buffer_layout(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
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
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU readback poll failed");
    receiver
        .recv()
        .expect("GPU callback failed")
        .expect("GPU mapping failed");
    let mapped = slice.get_mapped_range().expect("GPU readback unavailable");
    let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    values
}
