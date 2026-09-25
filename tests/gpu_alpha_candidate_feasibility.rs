//! Hardware feasibility check for alpha-mask ray-query candidates and the texture VM.
//! Run the feasibility probe with `cargo test --test gpu_alpha_candidate_feasibility
//! non_opaque_candidates_can_be_rejected_or_confirmed_using_a_texture -- --ignored`.

use std::num::NonZeroU32;
use std::process::Command;
use std::sync::mpsc;

use pbrt_r4::gpu::webgpu::abi::{RenderError, TextureNodeRecord, TEXTURE_OPERATION_IMAGE};
use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::shader::{compose_source_with_noise, resource_bindings};
use pbrt_r4::gpu::webgpu::stages::RequiredLimits;
use pbrt_r4::util::imageio::read_image::read_image;
use wgpu::util::DeviceExt;

const SHADER: &str = r#"
@group(0) @binding(60) var<storage, read_write> results: array<vec4<u32>, 2>;

@compute @workgroup_size(2)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    var query: ray_query;
    rayQueryInitialize(
        &query, tlas,
        RayDesc(0u, 0xffu, 0.0, 10.0, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)),
    );
    var candidates = 0u;
    var sampled_alpha = 0.0;
    while (rayQueryProceed(&query)) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        if (candidate.kind != 1u) { continue; }
        candidates += 1u;
        // Call the renderer's texture VM inside traversal, including its
        // binding-array image and sampler resources.
        material_texture_position = vec3<f32>(0.0);
        material_texture_normal = vec3<f32>(0.0, 0.0, 1.0);
        sampled_alpha = sample_texture_program(
            TextureRootRecord(id.x, 1u, 0u, 0u), vec2<f32>(0.5, 0.5),
        ).x;
        if (candidate.primitive_index == 1u || sampled_alpha > 0.5) {
            rayQueryConfirmIntersection(&query);
        }
    }
    let committed = rayQueryGetCommittedIntersection(&query);
    results[id.x] = vec4<u32>(
        candidates,
        committed.kind,
        committed.primitive_index,
        bitcast<u32>(sampled_alpha),
    );
}
"#;

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

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn non_opaque_candidates_can_be_rejected_or_confirmed_using_a_texture() {
    let context = Context::new(
        RequiredLimits {
            storage_buffers_per_shader_stage: 9,
            uniform_buffers_per_shader_stage: 0,
            buffers_and_acceleration_structures_per_shader_stage: 10,
            bind_groups: 2,
        },
        2,
        1,
    )
    .unwrap();
    let device = &context.device;
    let vertices: [[f32; 3]; 6] = [
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [0.0, 1.0, 1.0],
        [-1.0, -1.0, 2.0],
        [1.0, -1.0, 2.0],
        [0.0, 1.0, 2.0],
    ];
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("alpha candidate vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    let triangle_size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: 6,
        index_format: None,
        index_count: None,
        flags: wgpu::AccelerationStructureGeometryFlags::empty(),
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("alpha candidate BLAS"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![triangle_size.clone()],
        },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("alpha candidate TLAS"),
        max_instances: 1,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });
    tlas[0] = Some(wgpu::TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        0,
        0xff,
    ));

    let textures: Vec<_> = [0u8, 255u8]
        .into_iter()
        .map(|alpha| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("alpha candidate image"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            context.queue.write_texture(
                texture.as_image_copy(),
                &[alpha, alpha, alpha, 255],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                texture.size(),
            );
            texture
        })
        .collect();
    let views: Vec<_> = textures
        .iter()
        .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
        .collect();
    let view_refs: Vec<_> = views.iter().collect();
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("alpha candidate results"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("alpha candidate readback"),
        size: 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let nodes = [0u32, 1u32].map(|texture_index| TextureNodeRecord {
        kind: 0,
        first_child: 0,
        child_count: 0,
        implementation_hash: 0,
        swrap_mode: 0,
        twrap_mode: 0,
        color_space: 0,
        texture_index,
        operation: TEXTURE_OPERATION_IMAGE,
        mapping_kind: 0,
        sampler: 0,
        _operation_padding: 0,
        constant_value: [1.0, 0.0, 0.0, 0.0],
        mapping: identity,
    });
    let node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("alpha candidate texture nodes"),
        contents: bytemuck::cast_slice(&nodes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let child_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("alpha candidate texture children"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let error_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("alpha candidate render error"),
        contents: bytemuck::bytes_of(&RenderError {
            value: 0,
            padding: [0; 3],
        }),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // The shared shader composer includes triangle-sampling helpers in every stage.
    // They are not called by this probe, but their read-only bindings remain declared.
    let unused_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("unused triangle-sampling bindings"),
        size: 256,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
    let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("alpha candidate scene layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            },
            storage_layout(3, true),
            storage_layout(4, true),
            storage_layout(5, true),
            storage_layout(6, true),
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_layout(35, true),
            storage_layout(38, true),
            storage_layout(29, true),
            storage_layout(60, false),
        ],
    });
    let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("alpha candidate image layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: NonZeroU32::new(2),
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: NonZeroU32::new(1),
            },
        ],
    });
    let scene_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("alpha candidate scene group"),
        layout: &scene_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::AccelerationStructure(&tlas),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: unused_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: unused_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: unused_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: unused_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: error_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 29,
                resource: unused_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 35,
                resource: node_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 38,
                resource: child_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 60,
                resource: result_buffer.as_entire_binding(),
            },
        ],
    });
    let texture_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("alpha candidate image group"),
        layout: &texture_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureViewArray(&view_refs),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::SamplerArray(&[&sampler]),
            },
        ],
    });
    let source = compose_source_with_noise(SHADER, false)
        .replace(
            "binding_array<texture_2d<f32>>",
            "binding_array<texture_2d<f32>, 2u>",
        )
        .replace("binding_array<sampler>", "binding_array<sampler, 1u>");
    assert_eq!(
        resource_bindings(&source),
        [
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 11),
            (0, 29),
            (0, 35),
            (0, 38),
            (0, 60),
            (1, 0),
            (1, 1),
        ]
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("alpha candidate feasibility shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("alpha candidate pipeline layout"),
        bind_group_layouts: &[Some(&scene_layout), Some(&texture_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("alpha candidate feasibility pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures(
        [wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &triangle_size,
                vertex_buffer: &vertex_buffer,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }]
        .iter(),
        std::iter::once(&tlas),
    );
    context.queue.submit(Some(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &scene_group, &[]);
        pass.set_bind_group(1, &texture_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback, 0, 32);
    context.queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let words: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();

    assert!(
        words[0] >= 2,
        "black image did not expose both candidates: {words:?}"
    );
    assert_eq!(words[1], 1, "black image committed no triangle: {words:?}");
    assert_eq!(
        words[2], 1,
        "black image did not reach the back triangle: {words:?}"
    );
    assert_eq!(f32::from_bits(words[3]), 0.0);
    assert!(
        words[4] >= 1,
        "white image did not expose the front candidate: {words:?}"
    );
    assert_eq!(words[5], 1, "white image committed no triangle: {words:?}");
    assert_eq!(
        words[6], 0,
        "white image did not keep the front triangle: {words:?}"
    );
    assert_eq!(f32::from_bits(words[7]), 1.0);
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn named_zero_alpha_texture_reveals_the_infinite_light() {
    let directory = tempfile::tempdir().unwrap();
    let scene = directory.path().join("alpha-zero.pbrt");
    std::fs::write(
        &scene,
        r#"LookAt 0 0 3  0 0 0  0 1 0
Camera "perspective" "float fov" [35]
Film "rgb" "integer xresolution" [8] "integer yresolution" [8]
Sampler "independent" "integer pixelsamples" [16]
Integrator "path" "integer maxdepth" [1]
WorldBegin
LightSource "infinite" "rgb L" [1 1 1]
Texture "mask" "float" "constant" "float value" [0]
Material "diffuse" "rgb reflectance" [0 0 0]
Shape "trianglemesh"
    "point3 P" [-2 -2 0  2 -2 0  0 2 0]
    "integer indices" [0 1 2]
    "texture alpha" ["mask"]
"#,
    )
    .unwrap();
    let render = |gpu: bool| {
        let output = directory
            .path()
            .join(if gpu { "gpu.exr" } else { "cpu.exr" });
        let mut command = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"));
        if gpu {
            command.arg("--use-gpu");
        }
        let status = command
            .args([
                "--outfile",
                output.to_str().unwrap(),
                scene.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(status.success(), "render failed: gpu={gpu}");
        let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
        assert_eq!([resolution.x, resolution.y], [8, 8]);
        pixels[4 * 8 + 4].to_rgb().into_iter().sum::<f32>()
    };
    let cpu = render(false);
    let gpu = render(true);
    assert!(
        cpu > 0.1,
        "CPU reference did not reveal the infinite light: {cpu}"
    );
    assert!(
        (gpu - cpu).abs() < 0.2 * cpu,
        "GPU zero-alpha result differs from CPU: gpu={gpu}, cpu={cpu}"
    );
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn fractional_scalar_alpha_matches_cpu_coverage() {
    let directory = tempfile::tempdir().unwrap();
    let scene = directory.path().join("alpha-half.pbrt");
    std::fs::write(
        &scene,
        r#"LookAt 0 0 3  0 0 0  0 1 0
Camera "perspective" "float fov" [35]
Film "rgb" "integer xresolution" [16] "integer yresolution" [16]
Sampler "independent" "integer pixelsamples" [64]
Integrator "path" "integer maxdepth" [1]
WorldBegin
LightSource "infinite" "rgb L" [1 1 1]
Material "diffuse" "rgb reflectance" [0 0 0]
Shape "trianglemesh"
    "point3 P" [-2 -2 0  2 -2 0  0 2 0]
    "integer indices" [0 1 2]
    "float alpha" [0.5]
"#,
    )
    .unwrap();
    let render = |gpu: bool| {
        let output = directory
            .path()
            .join(if gpu { "half-gpu.exr" } else { "half-cpu.exr" });
        let mut command = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"));
        if gpu {
            command.arg("--use-gpu");
        }
        let status = command
            .args([
                "--outfile",
                output.to_str().unwrap(),
                scene.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(
            status.success(),
            "fractional-alpha render failed: gpu={gpu}"
        );
        let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
        assert_eq!([resolution.x, resolution.y], [16, 16]);
        pixels
            .iter()
            .map(|pixel| pixel.to_rgb().into_iter().sum::<f32>())
            .sum::<f32>()
            / pixels.len() as f32
    };
    let cpu = render(false);
    let gpu = render(true);
    assert!(cpu > 0.1, "CPU reference has no visible background: {cpu}");
    assert!(
        (gpu - cpu).abs() < 0.1 * cpu,
        "GPU fractional-alpha coverage differs from CPU: gpu={gpu}, cpu={cpu}"
    );
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn constant_zero_alpha_area_light_remains_visible_to_light_sampling() {
    let directory = tempfile::tempdir().unwrap();
    let scene = directory.path().join("zero-alpha-area-light.pbrt");
    std::fs::write(
        &scene,
        r#"LookAt 0 0 5  0 0 0  0 1 0
Camera "perspective" "float fov" [25]
Film "rgb" "integer xresolution" [1] "integer yresolution" [1]
Sampler "independent" "integer pixelsamples" [16]
Integrator "path" "integer maxdepth" [1]
WorldBegin
Texture "mask" "float" "constant" "float value" [0]
AttributeBegin
    AreaLightSource "diffuse" "rgb L" [20 20 20]
    Shape "trianglemesh"
        "point3 P" [-1 -1 2  0 1 2  1 -1 2]
        "integer indices" [0 1 2]
        "texture alpha" ["mask"]
AttributeEnd
Material "diffuse" "rgb reflectance" [0.8 0.8 0.8]
Shape "trianglemesh"
    "point3 P" [-3 -3 0  3 -3 0  0 3 0]
    "integer indices" [0 1 2]
"#,
    )
    .unwrap();
    let render = |gpu: bool| {
        let output = directory
            .path()
            .join(if gpu { "area-gpu.exr" } else { "area-cpu.exr" });
        let mut command = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"));
        if gpu {
            command.arg("--use-gpu");
        }
        let status = command
            .args([
                "--outfile",
                output.to_str().unwrap(),
                scene.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(status.success(), "area-light render failed: gpu={gpu}");
        let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
        assert_eq!([resolution.x, resolution.y], [1, 1]);
        pixels[0].to_rgb().into_iter().sum::<f32>()
    };
    let cpu = render(false);
    let gpu = render(true);
    assert!(
        cpu > 0.1,
        "CPU reference did not sample the area light: {cpu}"
    );
    assert!(
        gpu > 0.1,
        "GPU lost the constant-zero alpha area light: {gpu}"
    );
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn fractional_texture_alpha_area_light_emits_for_accepted_samples() {
    let directory = tempfile::tempdir().unwrap();
    let scene = directory.path().join("fractional-alpha-area-light.pbrt");
    let render = |with_alpha: bool| {
        let scene_text = format!(
            r#"LookAt 0 0 5  0 0 0  0 1 0
Camera "perspective" "float fov" [25]
Film "rgb" "integer xresolution" [16] "integer yresolution" [16]
Sampler "independent" "integer pixelsamples" [4096] "integer seed" [17]
Integrator "path" "integer maxdepth" [1]
WorldBegin
{alpha_texture}
AttributeBegin
    AreaLightSource "diffuse" "rgb L" [20 20 20]
    Shape "trianglemesh"
        "point3 P" [1 -1 2  2 1 2  3 -1 2]
        "integer indices" [0 1 2]
{alpha_attribute}
AttributeEnd
Material "diffuse" "rgb reflectance" [0.8 0.8 0.8]
Shape "trianglemesh"
    "point3 P" [-3 -3 0  3 -3 0  0 3 0]
    "integer indices" [0 1 2]
"#,
            alpha_texture = if with_alpha {
                "Texture \"mask\" \"float\" \"constant\" \"float value\" [0.5]"
            } else {
                ""
            },
            alpha_attribute = if with_alpha {
                "        \"texture alpha\" [\"mask\"]"
            } else {
                ""
            },
        );
        std::fs::write(&scene, scene_text).unwrap();
        let output = directory.path().join(if with_alpha {
            "fractional-area-gpu.exr"
        } else {
            "opaque-area-gpu.exr"
        });
        let mut command = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"));
        let status = command
            .args([
                "--use-gpu",
                "--outfile",
                output.to_str().unwrap(),
                scene.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(
            status.success(),
            "area-light render failed: alpha={with_alpha}"
        );
        let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
        assert_eq!([resolution.x, resolution.y], [16, 16]);
        pixels
            .iter()
            .map(|pixel| pixel.to_rgb().into_iter().sum::<f32>())
            .sum::<f32>()
            / pixels.len() as f32
    };
    let opaque = render(false);
    let fractional = render(true);
    assert!(
        (0.4..0.6).contains(&(fractional / opaque)),
        "fractional alpha should accept approximately half the sampled light points: alpha={fractional}, opaque={opaque}"
    );
}
