#![cfg(feature = "webgpu")]

use std::sync::{Arc, RwLock};

use pbrt_r4::base::filter::Filter as CpuFilter;
use pbrt_r4::displays::{Display, DisplayTile};
use pbrt_r4::gpu::flat::{flatten_filter, FilterKind};
use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::film::Film;
use pbrt_r4::gpu::webgpu::filter::{filter_buffer_data, FilterResources};
use pbrt_r4::gpu::webgpu::shader::compose_source;
use pbrt_r4::gpu::webgpu::stages::RequiredLimits;
use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::util::base::Point2f;
use pbrt_r4::util::error::PbrtError;
use wgpu::util::DeviceExt;

#[test]
fn filter_tables_preserve_cpu_cells_and_signed_lobes() {
    for name in ["box", "triangle", "gaussian", "mitchell", "sinc"] {
        let params = ParameterDictionary::default();
        let flat = flatten_filter(name, &params).unwrap();
        let cpu = CpuFilter::create(name, &params).unwrap();
        assert_eq!(flat.radius, [cpu.radius().x, cpu.radius().y]);
        let words = filter_buffer_data(&flat).unwrap();
        assert_eq!(words[0], flat.kind as u32);
        if let Some(d) = &flat.distribution {
            let [nx, ny] = d.dimensions;
            assert_eq!(nx, (32.0 * flat.radius[0]) as u32);
            assert_eq!(ny, (32.0 * flat.radius[1]) as u32);
            for y in 0..ny {
                for x in 0..nx {
                    let p = Point2f::new(
                        -flat.radius[0] + 2.0 * flat.radius[0] * (x as f32 + 0.5) / nx as f32,
                        -flat.radius[1] + 2.0 * flat.radius[1] * (y as f32 + 0.5) / ny as f32,
                    );
                    let actual = d.values[(y * nx + x) as usize];
                    assert!((actual - cpu.evaluate(&p)).abs() < 2e-6, "{name} {x} {y}");
                }
            }
            if matches!(flat.kind, FilterKind::Mitchell | FilterKind::Sinc) {
                assert!(d.values.iter().any(|v| *v < 0.0));
            }
        } else {
            assert!(matches!(flat.kind, FilterKind::Box | FilterKind::Triangle));
        }
    }
}

#[test]
fn invalid_filters_and_corrupt_layouts_are_errors() {
    for (name, key, value) in [
        ("gaussian", "xradius", 0.01),
        ("gaussian", "xradius", f32::INFINITY),
        ("gaussian", "xradius", 1e20),
        ("gaussian", "sigma", 0.0),
        ("gaussian", "sigma", 1e-20),
        ("gaussian", "sigma", 0.00304),
        ("mitchell", "B", f32::NAN),
        ("box", "yradius", -1.0),
    ] {
        let mut params = ParameterDictionary::default();
        params.add_float(key, value);
        assert!(
            flatten_filter(name, &params).is_err(),
            "{name} {key} {value}"
        );
    }
    let mut filter = flatten_filter("gaussian", &Default::default()).unwrap();
    filter.distribution.as_mut().unwrap().conditional_cdf.pop();
    assert!(filter_buffer_data(&filter).is_err());
}

fn context() -> Context {
    Context::new(
        RequiredLimits {
            storage_buffers_per_shader_stage: 3,
            buffers_and_acceleration_structures_per_shader_stage: 3,
            bind_groups: 1,
            ..Default::default()
        },
        1,
        1,
    )
    .unwrap()
}

#[test]
#[ignore = "requires a WebGPU adapter"]
fn filter_shader_matches_cpu_positions_and_weights() {
    let ctx = context();
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filter sampling comparison"),
            source: wgpu::ShaderSource::Wgsl(
                compose_source(
                    r#"
@group(0) @binding(80) var<storage, read> filter_test_inputs: array<vec2<f32>>;
@compute @workgroup_size(1)
fn test_filter(@builtin(global_invocation_id) id: vec3<u32>) {
    let fs = sample_filter(filter_test_inputs[id.x]);
    framebuffer[id.x] = vec4<f32>(fs.offset, fs.weight, 0.0);
}
"#,
                )
                .into(),
            ),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("filter sampling comparison"),
            layout: None,
            module: &module,
            entry_point: Some("test_filter"),
            compilation_options: Default::default(),
            cache: None,
        });
    for case in 0..3 {
        let custom = case == 1;
        let narrow = case == 2;
        let mut params = ParameterDictionary::default();
        if custom {
            params.add_float("xradius", 1.25);
            params.add_float("yradius", 2.75);
            params.add_float("sigma", 0.7);
            params.add_float("B", 0.2);
            params.add_float("C", 0.4);
            params.add_float("tau", 2.5);
        }
        if narrow {
            params.add_float("sigma", 0.1);
        }
        for name in ["box", "triangle", "gaussian", "mitchell", "sinc"] {
            if narrow && name != "gaussian" {
                continue;
            }
            let filter = flatten_filter(name, &params).unwrap();
            let cpu = CpuFilter::create(name, &params).unwrap();
            let mut inputs = vec![
                [0.0, 0.0],
                [0.5, 0.5],
                [f32::from_bits(1.0f32.to_bits() - 1); 2],
            ];
            for i in 0..1024u32 {
                inputs.push([
                    (i as f32 + 0.5) / 1024.0,
                    ((i * 397) % 1024) as f32 / 1024.0,
                ]);
            }
            if let Some(d) = &filter.distribution {
                // Equal CDF entries and exact interval boundaries must use FindInterval's <= rule.
                for &y in &d.marginal_cdf {
                    if y < 1.0 {
                        inputs.push([0.375, y]);
                    }
                }
                for &x in d.conditional_cdf.iter().step_by(7) {
                    if x < 1.0 {
                        inputs.push([x, 0.375]);
                    }
                }
            }
            let resources = FilterResources::new(&ctx.device, &filter).unwrap();
            let input = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&inputs),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let size = inputs.len() as u64 * 16;
            let output = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bindings = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 79,
                        resource: resources.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 80,
                        resource: input.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups(inputs.len() as u32, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
            ctx.queue.submit(Some(encoder.finish()));
            let slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            ctx.wait().unwrap();
            rx.recv().unwrap().unwrap();
            let bytes = slice.get_mapped_range().unwrap();
            let values: &[f32] = bytemuck::cast_slice(&bytes);
            for (i, u) in inputs.iter().enumerate() {
                let mut expected = cpu.sample(&Point2f::new(u[0], u[1]));
                if narrow {
                    // CPU f/pdf itself loses accuracy when the PDF is subnormal.
                    // The importance weight's exact magnitude is the absolute integral.
                    expected.weight = filter.distribution.as_ref().unwrap().integral
                        * (2.0 * filter.radius[0])
                        * (2.0 * filter.radius[1]);
                }
                for (channel, expected) in [expected.p.x, expected.p.y, expected.weight]
                    .into_iter()
                    .enumerate()
                {
                    let actual = values[4 * i + channel];
                    assert!(actual.is_finite() && (actual-expected).abs() < 1e-5 * expected.abs().max(1.0),
                        "{name} custom={custom} narrow={narrow} u={u:?} channel={channel}: GPU={actual}, CPU={expected}");
                }
            }
        }
    }
}

struct Capture(Arc<RwLock<Vec<f32>>>);
impl Display for Capture {
    fn start(&mut self, _: &str, _: &[usize; 2], _: &[&str]) -> Result<(), PbrtError> {
        Ok(())
    }
    fn update(&mut self, tile: &DisplayTile) -> Result<(), PbrtError> {
        assert_eq!((tile.x, tile.y, tile.width, tile.height), (0, 0, 4, 1));
        *self.0.write().unwrap() = tile.buffer.clone();
        Ok(())
    }
    fn end(&mut self) -> Result<(), PbrtError> {
        Ok(())
    }
}

#[test]
#[ignore = "requires a WebGPU adapter"]
fn film_readback_preserves_signed_and_zero_weight_sums() {
    let ctx = context();
    let mut film = Film::new(
        &ctx.device,
        [4, 1],
        [4, 1],
        [0, 0],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        1.0,
        false,
    )
    .unwrap();
    let captured = Arc::new(RwLock::new(Vec::new()));
    let display: Arc<RwLock<dyn Display>> = Arc::new(RwLock::new(Capture(captured.clone())));
    film.add_display(&display);
    film.start().unwrap();
    let samples: [[f32; 4]; 4] = [
        [-2.0, -4.0, -6.0, -2.0],
        [2.0, 4.0, 6.0, 2.0],
        [3.0, -2.0, 1.0, 0.0],
        [0.0; 4],
    ];
    ctx.queue
        .write_buffer(&film.framebuffer, 0, bytemuck::cast_slice(&samples));
    film.complete_sample().unwrap();
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    film.copy_to_readback(&mut encoder);
    ctx.queue.submit(Some(encoder.finish()));
    film.readback(&ctx.device).unwrap();
    film.update_display().unwrap();
    assert_eq!(
        *captured.read().unwrap(),
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0, -2.0, 1.0, 0.0, 0.0, 0.0]
    );
}
