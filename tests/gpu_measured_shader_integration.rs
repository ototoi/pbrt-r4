use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

use pbrt_r4::base::bxdf::{TransportMode, BXDF_REFL_TRANS_REFLECTION};
use pbrt_r4::bxdfs::{MeasuredBxDF, MeasuredBxDFData};
use pbrt_r4::gpu::flat::MeasuredBsdfLibrary;
use pbrt_r4::gpu::webgpu::abi::{material_table_uniform, MeasuredBsdfRecord, MeasuredTableRecord};
use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::stages::RequiredLimits;
use pbrt_r4::util::base::{Point2f, Vector3f};
use pbrt_r4::util::spectrum::SampledWavelengths;
use wgpu::util::DeviceExt;

const RESULT_FLOATS: usize = 16;

#[test]
#[ignore = "requires a WebGPU adapter with binding-array support"]
fn measured_shader_matches_cpu_f_pdf_and_sample_f() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/bsdfs/paper_white_spec.bsdf");
    let mut library = MeasuredBsdfLibrary::default();
    library.intern(&path).unwrap();
    let resources = library.finish().unwrap();
    assert_eq!(resources.atlas_pages.len(), 1);

    let bsdfs = resources
        .bsdfs
        .iter()
        .map(|record| MeasuredBsdfRecord {
            ndf: record.ndf,
            sigma: record.sigma,
            vndf: record.vndf,
            luminance: record.luminance,
            spectra: record.spectra,
            isotropic: u32::from(record.isotropic),
            padding: [0; 2],
        })
        .collect::<Vec<_>>();
    let tables = resources
        .tables
        .iter()
        .map(|record| MeasuredTableRecord {
            size: record.size,
            parameter_count: record.parameter_count,
            padding0: 0,
            parameter_sizes: record.parameter_sizes,
            padding1: 0,
            parameter_strides: record.parameter_strides,
            padding2: 0,
            parameter_value_offsets: record.parameter_value_offsets,
            padding3: 0,
            data_offset: record.data_offset,
            marginal_cdf_offset: record.marginal_cdf_offset,
            conditional_cdf_offset: record.conditional_cdf_offset,
            padding4: 0,
        })
        .collect::<Vec<_>>();

    let wavelengths = SampledWavelengths::sample_uniform(0.25, 360.0, 830.0);
    let wo = Vector3f::new(0.2, 0.3, 0.93).normalize();
    let wi = Vector3f::new(-0.1, 0.25, 0.96).normalize();
    let u = Point2f::new(0.37, 0.61);
    let cpu = MeasuredBxDF::new(
        Arc::new(MeasuredBxDFData::try_from_file(path.to_str().unwrap()).unwrap()),
        &wavelengths,
    );
    let expected_f = cpu.f(&wo, &wi, TransportMode::Radiance);
    let expected_pdf = cpu.pdf(
        &wo,
        &wi,
        TransportMode::Radiance,
        BXDF_REFL_TRANS_REFLECTION,
    );
    let expected_sample = cpu
        .sample_f(
            &wo,
            0.5,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("reference sample must be valid");

    let context = Context::new(
        RequiredLimits {
            storage_buffers_per_shader_stage: 3,
            uniform_buffers_per_shader_stage: 1,
            bind_groups: 2,
        },
        1,
        1,
    )
    .unwrap();
    let device = &context.device;
    let queue = &context.queue;
    let bsdf_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("measured test bsdfs"),
        contents: bytemuck::cast_slice(&bsdfs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("measured test tables"),
        contents: bytemuck::cast_slice(&tables),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut material_table = material_table_uniform(0).unwrap();
    material_table.measured_texture_width = resources.atlas_pages[0].resolution[0];
    material_table.measured_texture_height = resources.atlas_pages[0].resolution[1];
    material_table.measured_texture_count = 1;
    let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("measured test material table"),
        contents: bytemuck::bytes_of(&material_table),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("measured test output"),
        size: (RESULT_FLOATS * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("measured test readback"),
        size: (RESULT_FLOATS * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let page = &resources.atlas_pages[0];
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("measured test atlas"),
        size: wgpu::Extent3d {
            width: page.resolution[0],
            height: page.resolution[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(&page.texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(page.resolution[0] * 16),
            rows_per_image: Some(page.resolution[1]),
        },
        texture.size(),
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("measured test buffers"),
        entries: &[
            buffer_layout(0, wgpu::BufferBindingType::Uniform, false),
            buffer_layout(
                1,
                wgpu::BufferBindingType::Storage { read_only: true },
                false,
            ),
            buffer_layout(
                2,
                wgpu::BufferBindingType::Storage { read_only: true },
                false,
            ),
            buffer_layout(
                3,
                wgpu::BufferBindingType::Storage { read_only: false },
                false,
            ),
        ],
    });
    let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("measured test textures"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: NonZeroU32::new(1),
        }],
    });
    let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("measured test buffers"),
        layout: &group0_layout,
        entries: &[
            binding(0, material_buffer.as_entire_binding()),
            binding(1, bsdf_buffer.as_entire_binding()),
            binding(2, table_buffer.as_entire_binding()),
            binding(3, output.as_entire_binding()),
        ],
    });
    let views = [&view];
    let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("measured test textures"),
        layout: &group1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureViewArray(&views),
        }],
    });

    let lambda = wavelengths.lambda();
    let source = format!(
        "enable wgpu_binding_array;\n{}\n{}\n{}",
        include_str!("../src/gpu/webgpu/shaders/types.wgsl"),
        r#"
@group(0) @binding(0) var<uniform> material_table: MaterialTableUniform;
@group(0) @binding(1) var<storage, read> measured_bsdfs: array<MeasuredBsdfRecord>;
@group(0) @binding(2) var<storage, read> measured_tables: array<MeasuredTableRecord>;
@group(0) @binding(3) var<storage, read_write> output_values: array<vec4<f32>, 4>;
@group(1) @binding(0) var texture_images: binding_array<texture_2d<f32>, 1>;
fn set_render_error() {}
fn load_material_attribute(material_node: u32, ordinal: u32) -> AttributeRef {
    return AttributeRef(3u, 0u);
}
"#,
        include_str!("../src/gpu/webgpu/shaders/measured.wgsl")
    ) + &format!(
        r#"
@compute @workgroup_size(1)
fn measured_test() {{
    let wo = normalize(vec3<f32>({0}, {1}, {2}));
    let wi = normalize(vec3<f32>({3}, {4}, {5}));
    let lambda = vec4<f32>({6}, {7}, {8}, {9});
    let f = measured_f(0u, wo, wi, lambda);
    let pdf = measured_pdf(0u, wo, wi);
    let sampled = measured_sample_f(0u, wo, vec2<f32>({10}, {11}), lambda);
    output_values[0] = f;
    output_values[1] = vec4<f32>(pdf, sampled.pdf, f32(sampled.valid), 0.0);
    output_values[2] = sampled.f;
    output_values[3] = vec4<f32>(sampled.wi, 0.0);
}}
"#,
        wo.x, wo.y, wo.z, wi.x, wi.y, wi.z, lambda[0], lambda[1], lambda[2], lambda[3], u.x, u.y,
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("measured numerical test"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("measured numerical test"),
        bind_group_layouts: &[Some(&group0_layout), Some(&group1_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("measured numerical test"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("measured_test"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group0, &[]);
        pass.set_bind_group(1, &group1, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, (RESULT_FLOATS * 4) as u64);
    queue.submit(Some(encoder.finish()));
    let actual = read_f32_buffer(device, &readback);

    for (index, expected) in expected_f.values().iter().enumerate() {
        assert_close(actual[index], *expected, "f");
    }
    assert_close(actual[4], expected_pdf, "pdf");
    assert_close(actual[5], expected_sample.pdf, "sample pdf");
    assert_eq!(actual[6], 1.0);
    for (index, expected) in expected_sample.f.values().iter().enumerate() {
        assert_close(actual[8 + index], *expected, "sample f");
    }
    assert_close(actual[12], expected_sample.wi.x, "sample wi.x");
    assert_close(actual[13], expected_sample.wi.y, "sample wi.y");
    assert_close(actual[14], expected_sample.wi.z, "sample wi.z");
}

fn buffer_layout(
    binding: u32,
    ty: wgpu::BufferBindingType,
    has_dynamic_offset: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset,
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

fn assert_close(actual: f32, expected: f32, label: &str) {
    let tolerance = 2e-4 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: GPU {actual} != CPU/pbrt-v4 {expected}"
    );
}
