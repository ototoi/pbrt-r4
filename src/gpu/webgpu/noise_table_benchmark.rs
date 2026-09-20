//! Ignored benchmark for the WebGPU Noise permutation-table representation.
//!
//! Run explicitly with:
//! `cargo test noise_table_storage -- --ignored --nocapture`

use std::time::Instant;

use crate::textures::noise::{noise, noise_permutation};

const POINT_COUNT: u32 = 262144;

#[derive(Clone, Copy, Debug)]
enum TableStorage {
    Const,
    PackedUniform,
    IntegerTexture,
    IntegerTexture2d,
}

impl TableStorage {
    fn name(self) -> &'static str {
        match self {
            Self::Const => "const-array",
            Self::PackedUniform => "packed-ubo",
            Self::IntegerTexture => "r8uint-texture",
            Self::IntegerTexture2d => "r8uint-combined-2d",
        }
    }
}

#[test]
#[ignore = "requires a WebGPU adapter; compares shader build and execution time"]
fn noise_table_storage() {
    let permutation = permutation_table();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("a WebGPU adapter is required for this benchmark");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("noise table benchmark"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .expect("WebGPU device creation failed");

    let expected = (0..POINT_COUNT)
        .map(|i| {
            let x = 0.125 + i as f32 * 0.03125;
            let point = [x, -0.75 * x, 1.5 + x];
            let mut fbm = 0.0;
            let mut frequency = 1.0;
            let mut weight = 1.0;
            for _ in 0..6 {
                fbm += weight
                    * noise(
                        frequency * point[0],
                        frequency * point[1],
                        frequency * point[2],
                    );
                frequency *= 1.99;
                weight *= 0.5;
            }
            [noise(point[0], point[1], point[2]), fbm]
        })
        .collect::<Vec<_>>();

    let orders = [
        [
            TableStorage::Const,
            TableStorage::PackedUniform,
            TableStorage::IntegerTexture,
            TableStorage::IntegerTexture2d,
        ],
        [
            TableStorage::IntegerTexture,
            TableStorage::IntegerTexture2d,
            TableStorage::Const,
            TableStorage::PackedUniform,
        ],
        [
            TableStorage::PackedUniform,
            TableStorage::IntegerTexture,
            TableStorage::IntegerTexture2d,
            TableStorage::Const,
        ],
        [
            TableStorage::IntegerTexture,
            TableStorage::Const,
            TableStorage::PackedUniform,
            TableStorage::IntegerTexture2d,
        ],
    ];
    for (round, order) in orders.into_iter().enumerate() {
        for storage in order {
            let source = shader_source(storage, &permutation);
            let source_len = source.len();
            let build_start = Instant::now();
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(storage.name()),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(storage.name()),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let build_time = build_start.elapsed();

            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("noise output"),
                size: u64::from(POINT_COUNT * 2 * 4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("noise readback"),
                size: u64::from(POINT_COUNT * 2 * 4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let (table_buffer, table_texture) = match storage {
                TableStorage::PackedUniform => {
                    let mut packed = [0u32; 64];
                    for (index, value) in permutation.iter().copied().enumerate() {
                        packed[index / 4] |= value << ((index % 4) * 8);
                    }
                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("packed noise permutation"),
                        size: packed.len() as u64 * 4,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&packed));
                    (Some(buffer), None)
                }
                TableStorage::IntegerTexture => {
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("noise permutation texture"),
                        size: wgpu::Extent3d {
                            width: 256,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::R8Uint,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    let values = permutation
                        .iter()
                        .map(|value| *value as u8)
                        .collect::<Vec<_>>();
                    queue.write_texture(
                        texture.as_image_copy(),
                        &values,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(256),
                            rows_per_image: Some(1),
                        },
                        wgpu::Extent3d {
                            width: 256,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                    );
                    (None, Some(texture))
                }
                TableStorage::IntegerTexture2d => {
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("noise combined permutation texture"),
                        size: wgpu::Extent3d {
                            width: 256,
                            height: 257,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::R8Uint,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    let mut values = vec![0u8; 256 * 257];
                    for a in 0..256usize {
                        values[a] = permutation[a] as u8;
                        for b in 0..256usize {
                            values[(b + 1) * 256 + a] =
                                permutation[(permutation[a] as usize + b) & 255] as u8;
                        }
                    }
                    queue.write_texture(
                        texture.as_image_copy(),
                        &values,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(256),
                            rows_per_image: Some(257),
                        },
                        wgpu::Extent3d {
                            width: 256,
                            height: 257,
                            depth_or_array_layers: 1,
                        },
                    );
                    (None, Some(texture))
                }
                TableStorage::Const => (None, None),
            };
            let output_view = output.as_entire_buffer_binding();
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(output_view),
            }];
            let table_view = table_texture
                .as_ref()
                .map(|texture| texture.create_view(&Default::default()));
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
            match storage {
                TableStorage::PackedUniform => entries.push(wgpu::BindGroupEntry {
                    binding: 0,
                    resource: table_buffer.as_ref().unwrap().as_entire_binding(),
                }),
                TableStorage::IntegerTexture | TableStorage::IntegerTexture2d => {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(table_view.as_ref().unwrap()),
                    })
                }
                TableStorage::Const => {}
            }
            let layout = pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(storage.name()),
                layout: &layout,
                entries: &entries,
            });
            let dispatch_start = Instant::now();
            for _ in 0..8 {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(storage.name()),
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(storage.name()),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(POINT_COUNT.div_ceil(64), 1, 1);
                }
                queue.submit(Some(encoder.finish()));
            }
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("noise readback"),
            });
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, u64::from(POINT_COUNT * 2 * 4));
            queue.submit(Some(encoder.finish()));
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU poll failed");
            let slice = readback.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).ok();
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU readback poll failed");
            receiver
                .recv()
                .expect("GPU readback callback failed")
                .expect("GPU readback failed");
            let mapped = slice
                .get_mapped_range()
                .expect("GPU readback mapping failed");
            let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
            drop(mapped);
            readback.unmap();
            let execution_time = dispatch_start.elapsed();
            for (index, expected) in expected.iter().enumerate() {
                for (offset, expected) in expected.iter().enumerate() {
                    let actual = values[offset * POINT_COUNT as usize + index];
                    assert!(
                        (actual - expected).abs() < 2e-4,
                        "{}: {actual} != {expected}",
                        storage.name()
                    );
                }
            }
            println!(
                "round={round} {}: source={} bytes, build={:?}, execution(8 dispatches)={:?}",
                storage.name(),
                source_len,
                build_time,
                execution_time
            );
            drop(sampler);
        }
    }
}

#[test]
#[ignore = "requires a WebGPU adapter; validates the 2D permutation texture layout"]
fn noise_pair_texture_upload() {
    let permutation = permutation_table();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("a WebGPU adapter is required for this diagnostic");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("noise pair texture diagnostic"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .expect("WebGPU device creation failed");
    let mut expected = vec![0u8; 256 * 256];
    for a in 0..256usize {
        for b in 0..256usize {
            expected[b * 256 + a] = permutation[(permutation[a] as usize + b) & 255] as u8;
        }
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("noise pair texture"),
        size: wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        &expected,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(256),
            rows_per_image: Some(256),
        },
        wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
    );
    let source = "@group(0) @binding(0) var table: texture_2d<u32>; @group(0) @binding(1) var<storage, read_write> output: array<u32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { let i=id.x; output[i]=textureLoad(table, vec2<i32>(i32(i & 255u), i32(i >> 8u)), 0).x; }";
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("noise pair diagnostic"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("noise pair diagnostic"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("noise pair output"),
        size: 65536 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("noise pair readback"),
        size: 65536 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let view = texture.create_view(&Default::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("noise pair diagnostic"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("noise pair diagnostic"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("noise pair diagnostic"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1024, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 65536 * 4);
    queue.submit(Some(encoder.finish()));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU poll failed");
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU readback poll failed");
    receiver
        .recv()
        .expect("GPU readback callback failed")
        .expect("GPU readback failed");
    let mapped = slice
        .get_mapped_range()
        .expect("GPU readback mapping failed");
    let actual = bytemuck::cast_slice::<u8, u32>(&mapped);
    for (index, value) in actual.iter().copied().enumerate() {
        assert_eq!(
            value,
            u32::from(expected[index]),
            "pair table mismatch at ({}, {})",
            index & 255,
            index >> 8
        );
    }
}

fn permutation_table() -> Vec<u32> {
    (0..256)
        .map(|index| u32::from(noise_permutation(index)))
        .collect()
}

fn shader_source(storage: TableStorage, permutation: &[u32]) -> String {
    let table = permutation
        .iter()
        .map(|value| format!("{value}u"))
        .collect::<Vec<_>>()
        .join(", ");
    let resource = match storage {
        TableStorage::Const => format!("const NOISE_PERM: array<u32, 256> = array<u32, 256>({table});\nfn perm(i: u32) -> u32 {{ return NOISE_PERM[i & 255u]; }}"),
        TableStorage::PackedUniform => "struct NoiseTable { words: array<vec4<u32>, 16>, }; @group(0) @binding(0) var<uniform> noise_table: NoiseTable; fn perm(i: u32) -> u32 { let word = i >> 2u; let packed = noise_table.words[word >> 2u][word & 3u]; return (packed >> ((i & 3u) * 8u)) & 255u; }".to_string(),
        TableStorage::IntegerTexture => "@group(0) @binding(0) var noise_table: texture_2d<u32>; fn perm(i: u32) -> u32 { return textureLoad(noise_table, vec2<i32>(i32(i & 255u), 0), 0).x; }".to_string(),
        TableStorage::IntegerTexture2d => "@group(0) @binding(0) var noise_table: texture_2d<u32>; fn perm(i: u32) -> u32 { return textureLoad(noise_table, vec2<i32>(i32(i & 255u), 0), 0).x; } fn pair(a: u32, b: u32) -> u32 { return textureLoad(noise_table, vec2<i32>(i32(a & 255u), i32((b & 255u) + 1u)), 0).x; }".to_string(),
    };
    let gradient = match storage {
        TableStorage::IntegerTexture2d => "let a=pair(x,y); let h=perm((a+z)&255u)&15u;",
        _ => "let a=(perm(x)+y)&255u; let b=(perm(a)+z)&255u; let h=perm(b)&15u;",
    };
    format!("{resource}\n@group(0) @binding(1) var<storage, read_write> output: array<f32>; fn weight(t:f32)->f32 {{ let t3=t*t*t; let t4=t3*t; return 6.0*t4*t-15.0*t4+10.0*t3; }} fn grad(x:u32,y:u32,z:u32,dx:f32,dy:f32,dz:f32)->f32 {{ {gradient} let u=select(dy,dx,h<8u||h==12u||h==13u); let v=select(dz,dy,h<4u||h==12u||h==13u); return select(u,-u,(h&1u)!=0u)+select(v,-v,(h&2u)!=0u); }} fn eval(p:vec3<f32>)->f32 {{ let i=vec3<i32>(floor(p)); let d=p-vec3<f32>(i); let x=u32(i.x)&255u; let y=u32(i.y)&255u; let z=u32(i.z)&255u; let wx=weight(d.x); let wy=weight(d.y); let wz=weight(d.z); let a=mix(grad(x,y,z,d.x,d.y,d.z),grad((x+1u)&255u,y,z,d.x-1.0,d.y,d.z),wx); let b=mix(grad(x,(y+1u)&255u,z,d.x,d.y-1.0,d.z),grad((x+1u)&255u,(y+1u)&255u,z,d.x-1.0,d.y-1.0,d.z),wx); let c=mix(grad(x,y,(z+1u)&255u,d.x,d.y,d.z-1.0),grad((x+1u)&255u,y,(z+1u)&255u,d.x-1.0,d.y,d.z-1.0),wx); let e=mix(grad(x,(y+1u)&255u,(z+1u)&255u,d.x,d.y-1.0,d.z-1.0),grad((x+1u)&255u,(y+1u)&255u,(z+1u)&255u,d.x-1.0,d.y-1.0,d.z-1.0),wx); return mix(mix(a,b,wy),mix(c,e,wy),wz); }} fn eval_fbm(p:vec3<f32>)->f32 {{ var sum=0.0; var frequency=1.0; var weight_value=1.0; for (var octave=0u; octave<6u; octave++) {{ sum += weight_value * eval(frequency*p); frequency *= 1.99; weight_value *= 0.5; }} return sum; }} @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>) {{ if (id.x < {POINT_COUNT}u) {{ let x=0.125+f32(id.x)*0.03125; let p=vec3<f32>(x,-0.75*x,1.5+x); output[id.x]=eval(p); output[{POINT_COUNT}u+id.x]=eval_fbm(p); }} }}")
}
