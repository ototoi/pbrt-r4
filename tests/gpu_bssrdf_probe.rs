#![cfg(feature = "webgpu")]

use bytemuck::{bytes_of, cast_slice, Zeroable};
use pbrt_r4::base::bxdf::TransportMode;
use pbrt_r4::bssrdf::{BSSRDFTable, SubsurfaceInteraction, TabulatedBSSRDF};
use pbrt_r4::bxdfs::NormalizedFresnelBxDF;
use pbrt_r4::gpu::flat::{flatten_node, TabulatedBSSRDFTable};
use pbrt_r4::gpu::node::*;
use pbrt_r4::gpu::webgpu::abi::{
    BSSRDFProbeResult, BSSRDFProbeWorkItem, BSSRDFTableRecord, QueueState,
};
use pbrt_r4::gpu::webgpu::bssrdf::{probe_shader_source, BSSRDFProbePipeline, BSSRDFTableData};
use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::scene::Scene;
use pbrt_r4::gpu::webgpu::stages::RequiredLimits;
use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::util::base::{Float, Normal3f, Point3f, Vector3f};
use pbrt_r4::util::interpolation::{
    catmull_rom_weights, invert_catmull_rom, sample_catmull_rom_2d,
};
use pbrt_r4::util::spectrum::{SampledSpectrum, SampledWavelengths};
use std::sync::{mpsc, Arc, RwLock};
use wgpu::util::DeviceExt;

#[test]
fn bssrdf_gpu_table_layout_and_packing() {
    assert_eq!(std::mem::size_of::<BSSRDFTableRecord>(), 32);
    assert_eq!(std::mem::size_of::<BSSRDFProbeWorkItem>(), 96);
    assert_eq!(std::mem::size_of::<BSSRDFProbeResult>(), 128);
    let tables = [
        TabulatedBSSRDFTable::new(0.0, 1.33),
        TabulatedBSSRDFTable::new(0.2, 1.5),
    ];
    let data = BSSRDFTableData::from_flat(&tables).unwrap();
    assert_eq!(data.records.len(), 2);
    for (table, record) in tables.iter().zip(&data.records) {
        for (offset, values) in [
            (record.rho_offset, &table.rho_samples),
            (record.radius_offset, &table.radius_samples),
            (record.profile_offset, &table.profile),
            (record.rho_eff_offset, &table.rho_eff),
            (record.cdf_offset, &table.profile_cdf),
        ] {
            assert_eq!(
                &data.values[offset as usize..offset as usize + values.len()],
                values
            );
        }
    }
    let mut bad = tables[0].clone();
    bad.profile.pop();
    assert!(BSSRDFTableData::from_flat(&[bad]).is_err());
}

fn plane_scene(context: &Context) -> Scene {
    let mut root = Node::new("probe-test");
    root.add_component(Component::Output(OutputComponent {
        output: Output {
            filename: "probe.exr".into(),
        },
    }));
    let mut camera = Node::new("camera");
    camera.add_component(Component::Camera(CameraComponent {
        camera: Camera {
            kind: "perspective".into(),
            params: Default::default(),
            medium: None,
        },
    }));
    let mut film_params = ParameterDictionary::default();
    film_params.add_int("integer xresolution", 1);
    film_params.add_int("integer yresolution", 1);
    camera.add_component(Component::Film(FilmComponent {
        film: Film {
            name: "rgb".into(),
            params: film_params,
        },
    }));
    root.add_child(Arc::new(RwLock::new(camera)));
    let shared = Arc::new(Material {
        name: "matching".into(),
        kind: "diffuse".into(),
        params: Default::default(),
        material_attributes: Vec::new(),
        texture_attributes: Vec::new(),
    });
    for (i, z) in [-0.2, 0.0, 0.2].into_iter().enumerate() {
        let mut plane = Node::new("plane");
        let mat = if i == 1 {
            Arc::new(Material {
                name: "other".into(),
                ..(*shared).clone()
            })
        } else {
            shared.clone()
        };
        plane.add_component(Component::Material(MaterialComponent { material: mat }));
        plane.add_component(Component::Shape(ShapeComponent {
            shape: Shape::TriangleMesh(Box::new(TriangleMeshShape {
                source_shape: "trianglemesh".into(),
                positions: vec![
                    Vec3f([-100.0, -100.0, z]),
                    Vec3f([100.0, -100.0, z]),
                    Vec3f([0.0, 100.0, z]),
                ],
                indices: vec![0, 1, 2],
                normals: Some(vec![Vec3f([0.0, 0.0, 1.0]); 3]),
                tangents: Some(vec![Vec3f([1.0, 0.0, 0.0]); 3]),
                uvs: Some(vec![Vec2f([0.0, 0.0]); 3]),
            })),
            reverse_orientation: false,
        }));
        plane.add_component(Component::Medium(MediumComponent {
            medium_interface: MediumInterface::default(),
        }));
        root.add_child(Arc::new(RwLock::new(plane)));
    }
    Scene::from_flat(
        &context.device,
        &context.queue,
        flatten_node(Arc::new(RwLock::new(root))).unwrap(),
    )
    .unwrap()
}

fn run_probe(
    context: &Context,
    scene: &Scene,
    tables: &[TabulatedBSSRDFTable],
    items: &[BSSRDFProbeWorkItem],
) -> Vec<BSSRDFProbeResult> {
    let pipeline = BSSRDFProbePipeline::new(&context.device).unwrap();
    run_probe_pipeline(context, scene, tables, items, &pipeline)
}

fn run_probe_pipeline(
    context: &Context,
    scene: &Scene,
    tables: &[TabulatedBSSRDFTable],
    items: &[BSSRDFProbeWorkItem],
    pipeline: &BSSRDFProbePipeline,
) -> Vec<BSSRDFProbeResult> {
    let device = &context.device;
    let resources = BSSRDFTableData::from_flat(tables)
        .unwrap()
        .upload(device)
        .unwrap();
    let state = QueueState {
        count: items.len() as u32,
        capacity: items.len() as u32,
        overflow: 0,
        padding: 0,
    };
    let mut input = bytes_of(&state).to_vec();
    input.extend_from_slice(cast_slice(items));
    let work = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &input,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let size = (items.len() * std::mem::size_of::<BSSRDFProbeResult>()) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let error = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytes_of(&0u32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size + 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let group = pipeline
        .bind_group(device, scene, &resources, &work, &output, &error)
        .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    pipeline.encode(&mut encoder, &group, items.len() as u32);
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
    encoder.copy_buffer_to_buffer(&error, 0, &readback, size, 4);
    context.queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (send, recv) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        send.send(result).unwrap();
    });
    context.wait().unwrap();
    recv.recv().unwrap().unwrap();
    let data = slice.get_mapped_range().unwrap();
    assert_eq!(bytemuck::from_bytes::<u32>(&data[size as usize..]), &0);
    let results = cast_slice(&data[..size as usize]).to_vec();
    drop(data);
    readback.unmap();
    results
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn gpu_bssrdf_probe_samples_segments_and_same_material_reservoir() {
    let context = Context::new(
        RequiredLimits {
            storage_buffers_per_shader_stage: 9,
            uniform_buffers_per_shader_stage: 0,
            buffers_and_acceleration_structures_per_shader_stage: 10,
            bind_groups: 1,
        },
        0,
        0,
    )
    .unwrap();
    let scene = plane_scene(&context);
    let tables = [TabulatedBSSRDFTable::new(0.0, 1.33)];
    let work = BSSRDFProbeWorkItem {
        normal: [0.0, 0.0, 1.0, 0.0],
        sigma_t: [1.0; 4],
        rho: [0.8; 4],
        material_root: scene.instances[0].material_root,
        sample: [0.75, 0.4, 0.2, 0.0],
        ..Zeroable::zeroed()
    };
    let mut axis_items = Vec::new();
    for normal in [[0.0, 0.0, 1.0, 0.0], [0.36, 0.48, 0.8, 0.0]] {
        for uc in [0.125, 0.25, 0.5, 0.875] {
            axis_items.push(BSSRDFProbeWorkItem {
                normal,
                sample: [uc, 0.321, 0.271, 0.0],
                sigma_t: [2.0, 10.0, 20.0, 30.0],
                ..work
            });
        }
    }
    let axis_results = run_probe(&context, &scene, &tables, &axis_items);
    let table = &tables[0];
    let as_float = |v: &[f32]| {
        v.iter()
            .map(|&x| x as pbrt_r4::util::base::Float)
            .collect::<Vec<_>>()
    };
    let rho_nodes = as_float(&table.rho_samples);
    let radius_nodes = as_float(&table.radius_samples);
    let profile = as_float(&table.profile);
    let cdf = as_float(&table.profile_cdf);
    for (item, result) in axis_items.iter().zip(axis_results) {
        use pbrt_r4::util::base::Vector3f;
        use pbrt_r4::util::vecmath::Frame;
        let ns = Vector3f::new(
            item.normal[0] as _,
            item.normal[1] as _,
            item.normal[2] as _,
        );
        // pbrt-v4 CoordinateSystem reference bases.
        let (c0, c1) = if item.normal[0] == 0.0 {
            (Vector3f::new(1.0, 0.0, 0.0), Vector3f::new(0.0, 1.0, 0.0))
        } else {
            (
                Vector3f::new(0.928, -0.096, -0.36),
                Vector3f::new(-0.096, 0.872, -0.48),
            )
        };
        let frame = if item.sample[0] < 0.25 {
            Frame {
                x: ns,
                y: c0,
                z: c1,
            }
        } else if item.sample[0] < 0.5 {
            Frame {
                x: c1,
                y: ns,
                z: c0,
            }
        } else {
            Frame {
                x: c0,
                y: c1,
                z: ns,
            }
        };
        let radius = |u: f32| {
            sample_catmull_rom_2d(
                &rho_nodes,
                &radius_nodes,
                &profile,
                &cdf,
                item.rho[0] as _,
                u as _,
            )
            .unwrap()
            .0 / item.sigma_t[0] as pbrt_r4::util::base::Float
        };
        let r = radius(item.sample[1]);
        let phi = 2.0 * pbrt_r4::util::base::PI * item.sample[2] as pbrt_r4::util::base::Float;
        let center = r * (frame.x * phi.cos() + frame.y * phi.sin());
        let actual_start = Vector3f::new(
            result.start[0] as _,
            result.start[1] as _,
            result.start[2] as _,
        );
        let actual_end = Vector3f::new(result.end[0] as _, result.end[1] as _, result.end[2] as _);
        let actual_center = (actual_start + actual_end) * 0.5;
        let half_segment = (actual_end - actual_start) * 0.5;
        assert_eq!(result.segment_valid, 1);
        assert!((actual_center - center).length() < 5e-5);
        assert!((half_segment.normalize() - frame.z).length() < 1e-5);
        let optical_max = (half_segment.length_squared() + actual_center.length_squared()).sqrt()
            * item.sigma_t[0] as pbrt_r4::util::base::Float;
        let (offset, weights) = catmull_rom_weights(&rho_nodes, item.rho[0] as _).unwrap();
        let blend = |values: &[pbrt_r4::util::base::Float], col: usize| {
            (0..4)
                .filter(|&i| weights[i] != 0.0)
                .map(|i| {
                    values[(offset + i as i32) as usize * radius_nodes.len() + col] * weights[i]
                })
                .sum::<pbrt_r4::util::base::Float>()
        };
        let blended_profile = (0..radius_nodes.len())
            .map(|i| blend(&profile, i))
            .collect::<Vec<_>>();
        let density = |x| {
            let (offset, weights) = catmull_rom_weights(&radius_nodes, x).unwrap();
            (0..4)
                .filter(|&i| weights[i] != 0.0)
                .map(|i| blended_profile[(offset + i as i32) as usize] * weights[i])
                .sum::<pbrt_r4::util::base::Float>()
        };
        let idx = radius_nodes
            .partition_point(|&x| x <= optical_max)
            .saturating_sub(1)
            .min(radius_nodes.len() - 2);
        let x0 = radius_nodes[idx];
        let width = radius_nodes[idx + 1] - x0;
        // Simpson integration is exact for the cubic profile within one spline interval.
        let integral = (optical_max - x0) / 6.0
            * (density(x0) + 4.0 * density((x0 + optical_max) * 0.5) + density(optical_max));
        let residual =
            (blend(&cdf, idx) + integral - 0.999 * blend(&cdf, radius_nodes.len() - 1)) / width;
        assert!(residual.abs() < 2e-6, "CDF inversion residual: {residual}");
    }
    let mut items = Vec::new();
    for i in 0..1024 {
        items.push(BSSRDFProbeWorkItem {
            sample: [0.75, 0.4, (i as f32 + 0.5) / 1024.0, 0.0],
            ..work
        });
    }
    let results = run_probe(&context, &scene, &tables, &items);
    let mut counts = [0usize; 3];
    for result in &results {
        assert_eq!(result.segment_valid, 1);
        assert_eq!(result.valid, 1);
        assert_eq!(result.candidate_count, 2);
        assert_eq!(result.reservoir_probability, 0.5);
        counts[result.instance_index as usize] += 1;
    }
    assert_eq!(counts[1], 0);
    assert!((counts[0] as f32 / 1024.0 - 0.5).abs() < 0.06, "{counts:?}");
    items.reverse();
    let reordered = run_probe(&context, &scene, &tables, &items);
    for (a, b) in results.iter().rev().zip(&reordered) {
        assert_eq!(bytes_of(a), bytes_of(b));
    }
    let special = [
        BSSRDFProbeWorkItem {
            sigma_t: [0.0; 4],
            ..work
        },
        BSSRDFProbeWorkItem {
            material_root: u32::MAX,
            ..work
        },
    ];
    let result = run_probe(&context, &scene, &tables, &special);
    assert_eq!(result[0].segment_valid, 0);
    assert_eq!(result[1].segment_valid, 1);
    assert_eq!(result[1].valid, 0);
    assert_eq!(result[1].candidate_count, 0);
    assert_eq!(result[1].reservoir_probability, 0.0);
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn gpu_bssrdf_sp_pdf_and_normalized_fresnel_match_cpu() {
    let context = Context::new(
        RequiredLimits {
            storage_buffers_per_shader_stage: 9,
            uniform_buffers_per_shader_stage: 0,
            buffers_and_acceleration_structures_per_shader_stage: 10,
            bind_groups: 1,
        },
        0,
        0,
    )
    .unwrap();
    let scene = plane_scene(&context);
    let entries = (0..10)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            count: None,
            ty: if binding == 0 {
                wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                }
            } else {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding < 7,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            },
        })
        .collect::<Vec<_>>();
    let layout = context
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
    let pipeline_layout = context
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
    let probe_source = probe_shader_source();
    let source = probe_source.split("fn bssrdf_segment_seed").next().unwrap();
    let scattering = include_str!("../src/gpu/webgpu/shaders/lib/scattering.wgsl");
    let fresnel = scattering
        .split("fn dielectric_fresnel")
        .nth(1)
        .unwrap()
        .split("// pbrt-v4 HG")
        .next()
        .unwrap();
    let source = format!(
        "{source}\nfn dielectric_fresnel{fresnel}\n{}\n{}",
        include_str!("../src/gpu/webgpu/shaders/lib/bssrdf_scattering.wgsl"),
        r#"
        @compute @workgroup_size(64)
        fn evaluate_bssrdf(@builtin(global_invocation_id) id: vec3<u32>) {
            if (id.x >= atomicLoad(&bssrdf_work.state.count)) { return; }
            let work = bssrdf_work.items[id.x];
            var result: BSSRDFProbeResult;
            result.start = bssrdf_sr(work, work.sample.x, false);
            result.end = bssrdf_pdf_sp(work, work.position.xyz + work.sample.yzw, vec3<f32>(0.36, 0.48, 0.8));
            let table = bssrdf_tables[work.table_index];
            for (var i = 0u; i < 4u; i++) { result.position[i] = bssrdf_invert_reflectance(table, work.rho[i]); }
            result.normal = bssrdf_normalized_fresnel(work.position.w, 1.33);
            result.barycentric = vec4<f32>(bssrdf_sample_cosine(work.rho.xy), 0.0);
            bssrdf_results[id.x] = result;
        }
        "#
    );
    let module = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = BSSRDFProbePipeline {
        pipeline: context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("evaluate_bssrdf"),
                compilation_options: Default::default(),
                cache: None,
            }),
    };
    let tables = [
        TabulatedBSSRDFTable::new(0.0, 1.33),
        TabulatedBSSRDFTable::new(0.4, 1.5),
    ];
    let mut items = Vec::new();
    for table_index in 0..2 {
        for radius in [0.0f32, 0.001, 0.03, 0.3, 1.0, 10.0] {
            for normal in [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]] {
                items.push(BSSRDFProbeWorkItem {
                    position: [0.0, 0.0, 0.0, 0.4],
                    normal,
                    sigma_t: [1.0, 2.0, 4.0, 0.0],
                    rho: [0.9, 0.8, 0.5, 0.0],
                    sample: [radius, radius * 0.6, radius * 0.8, 0.0],
                    table_index,
                    ..Zeroable::zeroed()
                });
            }
        }
    }
    let results = run_probe_pipeline(&context, &scene, &tables, &items, &pipeline);
    let close = |actual: f32, expected: Float| {
        assert!(actual.is_finite());
        let error = (actual as Float - expected).abs();
        assert!(
            error < 3e-5 * expected.abs().max(1.0),
            "GPU {actual}, CPU {expected}, error {error}"
        );
    };
    let lambda = SampledWavelengths::sample_visible(0.5);
    for (work, result) in items.iter().zip(results) {
        let table = &tables[work.table_index as usize];
        let convert = |v: &[f32]| v.iter().map(|v| *v as Float).collect::<Vec<_>>();
        let cpu_table = Arc::new(BSSRDFTable {
            rho_samples: convert(&table.rho_samples),
            radius_samples: convert(&table.radius_samples),
            profile: convert(&table.profile),
            rho_eff: convert(&table.rho_eff),
            profile_cdf: convert(&table.profile_cdf),
        });
        let sigma_a = SampledSpectrum::from(std::array::from_fn(|i| {
            (work.sigma_t[i] * (1.0 - work.rho[i])) as Float
        }));
        let sigma_s = SampledSpectrum::from(std::array::from_fn(|i| {
            (work.sigma_t[i] * work.rho[i]) as Float
        }));
        let bssrdf = TabulatedBSSRDF::new(
            Point3f::zero(),
            Normal3f::new(
                work.normal[0] as Float,
                work.normal[1] as Float,
                work.normal[2] as Float,
            ),
            Vector3f::new(0.0, 0.0, 1.0),
            table.eta as Float,
            sigma_a,
            sigma_s,
            cpu_table.clone(),
        );
        let ni = Normal3f::new(0.36, 0.48, 0.8);
        let ssi = SubsurfaceInteraction {
            p: Point3f::new(
                work.sample[1] as Float,
                work.sample[2] as Float,
                work.sample[3] as Float,
            ),
            p_error: Vector3f::zero(),
            n: ni,
            ns: ni,
            dpdu: Vector3f::new(1.0, 0.0, 0.0),
            dpdv: Vector3f::new(0.0, 1.0, 0.0),
            dpdus: Vector3f::new(1.0, 0.0, 0.0),
            dpdvs: Vector3f::new(0.0, 1.0, 0.0),
            time: 0.0,
        };
        if let Some(sample) = bssrdf.probe_intersection_to_sample(&ssi, &lambda) {
            for ch in 0..4 {
                close(result.start[ch], sample.sp[ch]);
                close(result.end[ch], sample.pdf[ch]);
            }
        } else {
            assert!(result.start.iter().all(|v| v.is_finite()));
        }
        for ch in 0..4 {
            close(
                result.position[ch],
                invert_catmull_rom(
                    &cpu_table.rho_samples,
                    &cpu_table.rho_eff,
                    work.rho[ch] as Float,
                ),
            );
        }
        let cosine = work.position[3] as Float;
        let f = NormalizedFresnelBxDF::new(1.33).f(
            &Vector3f::new(0.0, 0.0, 1.0),
            &Vector3f::new((1.0 - cosine * cosine).sqrt(), 0.0, cosine),
            TransportMode::Radiance,
        );
        close(result.normal[0], f[0]);
        close(
            result.barycentric[0] * result.barycentric[0]
                + result.barycentric[1] * result.barycentric[1]
                + result.barycentric[2] * result.barycentric[2],
            1.0,
        );
    }
}
