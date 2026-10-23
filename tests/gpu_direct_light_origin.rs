#![cfg(feature = "webgpu")]

use std::collections::HashMap;
use std::mem::{offset_of, size_of};
use std::sync::mpsc;

use bytemuck::{cast_slice, Pod, Zeroable};
use pbrt_r4::gpu::flat::tabulate_bssrdf;
use pbrt_r4::gpu::webgpu::abi::*;
use pbrt_r4::gpu::webgpu::binding_report::BindingReport;
use pbrt_r4::gpu::webgpu::bssrdf::BSSRDFTableData;
use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::material::MaterialKind;
use pbrt_r4::gpu::webgpu::pipeline::Pipeline;
use pbrt_r4::gpu::webgpu::shader::{compose_source_with_features, specialize_source};
use pbrt_r4::gpu::webgpu::stage::{ComputeStageId, COMPUTE_STAGES};
use pbrt_r4::gpu::webgpu::stages::{canonical_wavefront_bindings, BindingClass, ResourceId};

#[test]
fn prepared_origin_uses_only_surface_padding_and_preserves_every_existing_offset() {
    let source = include_str!("../src/gpu/webgpu/shaders/types.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
    let ty = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("SurfaceWorkItem"))
        .unwrap()
        .1;
    let wgpu::naga::TypeInner::Struct { members, span } = &ty.inner else {
        panic!("expected struct")
    };
    assert_eq!(*span, 256);
    assert_eq!(size_of::<SurfaceWorkItem>(), 256);
    macro_rules! fields {
        ($($field:ident => $expected:expr),* $(,)?) => {
            $(
                assert_eq!(offset_of!(SurfaceWorkItem, $field), $expected);
                assert_eq!(members.iter().find(|m| m.name.as_deref() == Some(stringify!($field))).unwrap().offset, $expected);
            )*
        };
    }
    fields! {
        t => 0, hit => 4, instance_custom_data => 8, primitive_index => 12,
        barycentric => 16, position => 32, position_error => 48,
        normal => 64, geometric_normal => 80, tangent => 96,
        dpdu => 112, dpdv => 128, dndu => 144, dndv => 160,
        dpdx => 176, dpdy => 192, uv_differentials => 208, uv => 224,
        material_root => 232, flags => 236, attributes_eval_work_item => 240,
    }
    assert_eq!(offset_of!(SurfaceWorkItem, light_sample_origin), 244);
    for (name, offset) in [
        ("light_sample_origin_x", 244),
        ("light_sample_origin_y", 248),
        ("light_sample_origin_z", 252),
    ] {
        assert_eq!(
            members
                .iter()
                .find(|m| m.name.as_deref() == Some(name))
                .unwrap()
                .offset,
            offset
        );
    }
}

#[test]
fn direct_sampling_drops_material_dependencies_without_growing_producers() {
    let registry = canonical_wavefront_bindings();
    for noise in [false, true] {
        for nanovdb in [false, true] {
            let report =
                BindingReport::for_stages(&registry, COMPUTE_STAGES, noise, nanovdb).unwrap();
            for (id, storage, uniform) in [
                (ComputeStageId::SelectSurfaceLight, 14, 1),
                (ComputeStageId::SelectMediumLight, 13, 1),
                (ComputeStageId::SampleAreaLight, 18, 0),
                (ComputeStageId::SampleImageInfiniteLight, 20, 0),
                (ComputeStageId::SampleOtherLight, 15, 0),
                (ComputeStageId::ClassifySurfaceScatter, 17, 3),
                (ComputeStageId::PrepareSubsurfaceExit, 12, 1),
                (ComputeStageId::ScatterMedium, 12, 3),
            ] {
                let stage = report
                    .stages
                    .iter()
                    .find(|stage| stage.stage_id == id)
                    .unwrap();
                assert_eq!(
                    stage.required.storage_buffers_per_shader_stage, storage,
                    "{id:?}"
                );
                assert_eq!(
                    stage.required.uniform_buffers_per_shader_stage, uniform,
                    "{id:?}"
                );
                if id == ComputeStageId::SelectSurfaceLight
                    || id == ComputeStageId::SelectMediumLight
                {
                    assert!(stage.bindings.iter().all(|b| !matches!(
                        b.resource,
                        ResourceId::MaterialTable | ResourceId::AttributesEvalWorkItems
                    )));
                }
            }
            for stage in COMPUTE_STAGES {
                let source = specialize_source(
                    compose_source_with_features(stage.source, noise, nanovdb),
                    1,
                    1,
                    1,
                );
                let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
                    panic!("{}: {}", stage.entry_point, e.emit_to_string(&source))
                });
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap_or_else(|e| panic!("{}: {e}", stage.entry_point));
            }
        }
    }
}

// Execute the production pipelines and ABI directly. Unused light-kind buffers
// have dummy storage, while the active point light is fully populated.
struct GpuFixture {
    context: Context,
    pipeline: Pipeline,
    buffers: HashMap<ResourceId, wgpu::Buffer>,
    image: wgpu::TextureView,
    noise: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

impl GpuFixture {
    fn new() -> Self {
        let registry = canonical_wavefront_bindings();
        let report = BindingReport::for_stages(&registry, COMPUTE_STAGES, false, false).unwrap();
        let context =
            Context::with_binding_report_and_geometry_page_count(&report, 1, 1, 2).unwrap();
        let device = &context.device;
        let pipeline =
            Pipeline::new_with_geometry_page_count(device, 1, 1, 1, false, false, 2).unwrap();
        let buffers = registry
            .iter()
            .filter_map(|b| {
                let usage = match b.class {
                    BindingClass::Uniform => wgpu::BufferUsages::UNIFORM,
                    BindingClass::Storage => wgpu::BufferUsages::STORAGE,
                    _ => return None,
                };
                Some((
                    b.resource,
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("origin fixture {:?}", b.resource)),
                        size: if b.class == BindingClass::Uniform {
                            4096
                        } else {
                            256 * 1024
                        },
                        usage: usage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                ))
            })
            .collect();
        let texture = |format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("origin fixture texture"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let image = texture(wgpu::TextureFormat::Rgba8Unorm);
        let noise = texture(wgpu::TextureFormat::Rgba8Uint);
        let sampler = device.create_sampler(&Default::default());
        let fixture = Self {
            context,
            pipeline,
            buffers,
            image,
            noise,
            sampler,
        };
        fixture.write(
            ResourceId::LightSamplingParams,
            &[LightTableUniform {
                light_count: 1,
                light_sampler_kind: LIGHT_SAMPLER_KIND_UNIFORM,
                ..Zeroable::zeroed()
            }],
        );
        fixture.write(
            ResourceId::LightRecord,
            &[LightRecord {
                kind: LIGHT_KIND_POINT,
                attribute_count: 2,
                ..Zeroable::zeroed()
            }],
        );
        fixture.write(
            ResourceId::LightSamplingModel,
            &[LightSamplingModel::zeroed()],
        );
        fixture.write(ResourceId::LightPosition, &[[3.0f32, 4.0, 20.0, 1.0]]);
        fixture.write(
            ResourceId::AttributeRef,
            &[
                AttributeRef { kind: 1, index: 0 },
                AttributeRef { kind: 0, index: 0 },
            ],
        );
        fixture.write(ResourceId::ScalarAttribute, &[1.0f32]);
        fixture.write(
            ResourceId::SpectrumAttribute,
            &[DenseSpectrum {
                samples: [1.0; pbrt_r4::gpu::flat::DENSE_SAMPLE_COUNT],
                flags: 1,
            }],
        );
        fixture
    }

    fn write<T: Pod>(&self, resource: ResourceId, values: &[T]) {
        self.context
            .queue
            .write_buffer(&self.buffers[&resource], 0, cast_slice(values));
    }

    fn read<T: Pod>(&self, resource: ResourceId, count: usize) -> Vec<T> {
        let size = (size_of::<T>() * count) as u64;
        let device = &self.context.device;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("origin fixture readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffers[&resource], 0, &readback, 0, size);
        self.context.queue.submit(Some(encoder.finish()));
        let (tx, rx) = mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.context.wait().unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = readback.slice(..).get_mapped_range().unwrap();
        let values = cast_slice(&mapped).to_vec();
        drop(mapped);
        readback.unmap();
        values
    }

    fn run(&self, id: ComputeStageId, groups: u32) {
        assert_eq!(self.dispatch(id, groups), 0);
    }

    fn dispatch(&self, id: ComputeStageId, groups: u32) -> u32 {
        let stage = self.pipeline.stage(id);
        let views = [&self.image];
        let samplers = [&self.sampler];
        let vertex_buffers = [
            wgpu::BufferBinding {
                buffer: &self.buffers[&ResourceId::Vertex],
                offset: 0,
                size: None,
            },
            wgpu::BufferBinding {
                buffer: &self.buffers[&ResourceId::Vertex],
                offset: 0,
                size: None,
            },
        ];
        let index_buffers = [
            wgpu::BufferBinding {
                buffer: &self.buffers[&ResourceId::Index],
                offset: 0,
                size: None,
            },
            wgpu::BufferBinding {
                buffer: &self.buffers[&ResourceId::Index],
                offset: 0,
                size: None,
            },
        ];
        let registry = canonical_wavefront_bindings();
        let bind_groups = stage
            .bind_group_layouts
            .iter()
            .enumerate()
            .map(|(group, layout)| {
                let entries = registry
                    .iter()
                    .filter(|b| {
                        b.group == group as u32
                            && stage.used_bindings().contains(&(b.group, b.binding))
                    })
                    .map(|b| wgpu::BindGroupEntry {
                        binding: b.binding,
                        resource: match b.resource {
                            ResourceId::TextureImageArray => {
                                wgpu::BindingResource::TextureViewArray(&views)
                            }
                            ResourceId::TextureSamplerArray => {
                                wgpu::BindingResource::SamplerArray(&samplers)
                            }
                            ResourceId::Vertex => {
                                wgpu::BindingResource::BufferArray(&vertex_buffers)
                            }
                            ResourceId::Index => wgpu::BindingResource::BufferArray(&index_buffers),
                            ResourceId::NoiseTable => {
                                wgpu::BindingResource::TextureView(&self.noise)
                            }
                            ResourceId::SamplerTable => {
                                wgpu::BindingResource::TextureView(&self.noise)
                            }
                            _ => self
                                .buffers
                                .get(&b.resource)
                                .unwrap_or_else(|| {
                                    panic!("missing fixture buffer for {:?}", b.resource)
                                })
                                .as_entire_binding(),
                        },
                    })
                    .collect::<Vec<_>>();
                self.context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("origin fixture"),
                        layout,
                        entries: &entries,
                    })
            })
            .collect::<Vec<_>>();
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&stage.pipeline);
            for (i, group) in bind_groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[]);
            }
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));
        self.context.wait().unwrap();
        self.read::<RenderError>(ResourceId::RenderError, 1)[0].value
    }

    fn assert_point_sample(&self, pixel: usize, origin: [f32; 3]) {
        let sample =
            self.read::<DirectLightSample>(ResourceId::DirectLightSample, pixel + 1)[pixel];
        assert_eq!(sample.valid, 1);
        assert_eq!(sample.direction_pdf[3], 1.0);
        let vector = [3.0 - origin[0], 4.0 - origin[1], 20.0 - origin[2]];
        let length = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        for i in 0..3 {
            assert!(
                (sample.direction_pdf[i] - vector[i] / length).abs() < 1e-6,
                "pixel {pixel}: origin {origin:?}, sample {:?}",
                sample.direction_pdf
            );
        }
        assert!(sample.radiance.iter().all(|v| v.is_finite() && *v > 0.0));
    }
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn typed_light_queue_overflow_preserves_regions_and_dispatch_is_clamped() {
    let gpu = GpuFixture::new();
    let sentinel = LightSampleWorkItem {
        ray_index: 0xdead_beef,
        source_kind: 0xdead_beef,
        light_index: 0xdead_beef,
        selection_pmf: f32::NAN,
    };
    gpu.write(ResourceId::LightSampleWork, &[sentinel; 6]);
    gpu.write(
        ResourceId::CurrentRay,
        &[
            RayWorkItem {
                pixel_index: 0,
                ..Zeroable::zeroed()
            },
            RayWorkItem {
                pixel_index: 1,
                ..Zeroable::zeroed()
            },
        ],
    );
    gpu.write(ResourceId::DirectEvalQueue, &[0u32, 1]);
    gpu.write(
        ResourceId::PixelSampleState,
        &[
            PixelSampleState {
                direct: [0.1, 0.2, 0.3, 0.0],
                ..Zeroable::zeroed()
            },
            PixelSampleState {
                direct: [0.9, 0.2, 0.3, 0.0],
                ..Zeroable::zeroed()
            },
        ],
    );
    gpu.write(
        ResourceId::QueueCounters,
        &[QueueCounters {
            direct: state(2, 2),
            sample_area_light: state(0, 1),
            sample_image_infinite_light: state(0, 1),
            sample_other_light: state(0, 1),
            ..Zeroable::zeroed()
        }],
    );
    assert_eq!(gpu.dispatch(ComputeStageId::SelectSurfaceLight, 1), 1);
    let counters = gpu.read::<QueueCounters>(ResourceId::QueueCounters, 1)[0];
    assert_eq!(counters.sample_other_light.count, 2);
    assert_eq!(counters.sample_other_light.overflow, 1);
    let work = gpu.read::<LightSampleWorkItem>(ResourceId::LightSampleWork, 6);
    for index in [0, 1, 3, 4, 5] {
        assert_eq!(work[index].ray_index, sentinel.ray_index);
        assert_eq!(work[index].source_kind, sentinel.source_kind);
        assert_eq!(work[index].light_index, sentinel.light_index);
        assert_eq!(
            work[index].selection_pmf.to_bits(),
            sentinel.selection_pmf.to_bits()
        );
    }
    assert_eq!(work[2].ray_index, 0);
    assert_eq!(work[2].source_kind, 0);

    for (count, expected) in [
        (0, [0, 1, 1]),
        (1, [1, 1, 1]),
        (64, [1, 1, 1]),
        (65, [2, 1, 1]),
        (4_194_241, [65_535, 2, 1]),
    ] {
        gpu.write(
            ResourceId::QueueCounters,
            &[QueueCounters {
                sample_other_light: state(count, count),
                ..Zeroable::zeroed()
            }],
        );
        assert_eq!(gpu.dispatch(ComputeStageId::PrepareQueueDispatch, 1), 1);
        let args = gpu.read::<DispatchIndirectArgs>(ResourceId::QueueDispatchArgs, 18);
        assert_eq!(
            [
                args[QUEUE_DISPATCH_SLOT_SAMPLE_OTHER_LIGHT as usize].x,
                args[QUEUE_DISPATCH_SLOT_SAMPLE_OTHER_LIGHT as usize].y,
                args[QUEUE_DISPATCH_SLOT_SAMPLE_OTHER_LIGHT as usize].z
            ],
            expected,
            "count={count}"
        );
    }
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn production_selector_routes_every_supported_light_family() {
    let gpu = GpuFixture::new();
    let kinds = [1, 5, 6, 0, 2, 3, 4]; // area, image, portal, point, spot, distant, uniform infinite
    let count = kinds.len() as u32;
    gpu.write(
        ResourceId::LightSamplingParams,
        &[LightTableUniform {
            light_count: count,
            light_sampler_kind: LIGHT_SAMPLER_KIND_UNIFORM,
            ..Zeroable::zeroed()
        }],
    );
    gpu.write(
        ResourceId::LightRecord,
        &kinds
            .iter()
            .map(|&kind| LightRecord {
                kind,
                ..Zeroable::zeroed()
            })
            .collect::<Vec<_>>(),
    );
    gpu.write(
        ResourceId::CurrentRay,
        &(0..count)
            .map(|pixel_index| RayWorkItem {
                pixel_index,
                ..Zeroable::zeroed()
            })
            .collect::<Vec<_>>(),
    );
    gpu.write(ResourceId::DirectEvalQueue, &(0..count).collect::<Vec<_>>());
    gpu.write(
        ResourceId::Surface,
        &vec![SurfaceWorkItem::zeroed(); count as usize],
    );
    gpu.write(
        ResourceId::PixelSampleState,
        &(0..count)
            .map(|index| PixelSampleState {
                direct: [((index as f32) + 0.5) / count as f32, 0.25, 0.75, 0.0],
                ..Zeroable::zeroed()
            })
            .collect::<Vec<_>>(),
    );
    gpu.write(
        ResourceId::QueueCounters,
        &[QueueCounters {
            direct: state(count, count),
            sample_area_light: state(0, count),
            sample_image_infinite_light: state(0, count),
            sample_other_light: state(0, count),
            ..Zeroable::zeroed()
        }],
    );
    gpu.run(ComputeStageId::SelectSurfaceLight, 1);
    let counters = gpu.read::<QueueCounters>(ResourceId::QueueCounters, 1)[0];
    assert_eq!(counters.sample_area_light.count, 1);
    assert_eq!(counters.sample_image_infinite_light.count, 2);
    assert_eq!(counters.sample_other_light.count, 4);
    let items = gpu.read::<LightSampleWorkItem>(ResourceId::LightSampleWork, count as usize * 3);
    assert_eq!(items[0].light_index, 0);
    assert_eq!(items[count as usize].light_index, 1);
    assert_eq!(items[count as usize + 1].light_index, 2);
    for (slot, indices) in [(2, [3, 4, 5, 6])].into_iter() {
        for (offset, light_index) in indices.into_iter().enumerate() {
            let item = items[slot * count as usize + offset];
            assert_eq!(item.light_index, light_index as u32);
            assert_eq!(item.ray_index, light_index as u32);
        }
    }
}

fn state(count: u32, capacity: u32) -> QueueState {
    QueueState {
        count,
        capacity,
        ..Zeroable::zeroed()
    }
}

fn evaluated(kind: MaterialKind) -> AttributesEvalWorkItem {
    let mut item = AttributesEvalWorkItem {
        bxdf_kind: kind.tag(),
        ..Zeroable::zeroed()
    };
    item.values[0][0] = 1.5; // Non-unit eta for rough dielectric.
    item.values[1][0] = 0.1;
    item.values[2][0] = 0.1; // Rough conductor / anisotropic dielectric.
    item
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn surface_producers_prepare_origins_for_the_real_light_sampler() {
    let gpu = GpuFixture::new();
    let cases = [
        (MaterialKind::Diffuse, true),
        (MaterialKind::DiffuseTransmission, true),
        (MaterialKind::ConductorEtaK, true),
        (MaterialKind::ConductorReflectance, true),
        (MaterialKind::Measured, true),
        (MaterialKind::CoatedDiffuse, true),
        (MaterialKind::CoatedConductor, true),
        (MaterialKind::Dielectric, true),
        (MaterialKind::Subsurface, true),
        (MaterialKind::ThinDielectric, false),
        (MaterialKind::Normal, false),
        (MaterialKind::Uv, false),
        (MaterialKind::Mix, true),
        (MaterialKind::Mix, true),
        (MaterialKind::Diffuse, false),
        (MaterialKind::Dielectric, false),
        (MaterialKind::ConductorEtaK, false),
        (MaterialKind::Diffuse, false),
        (MaterialKind::Diffuse, false),
    ];
    let count = cases.len() as u32;
    gpu.write(
        ResourceId::SampleParams,
        &[ViewportUniform {
            tile_width: count,
            tile_height: 1,
            max_depth: 2,
            ..Zeroable::zeroed()
        }],
    );
    gpu.write(
        ResourceId::MaterialTable,
        &[MaterialTableUniform {
            attributes_eval_stride: 3,
            ..Zeroable::zeroed()
        }],
    );
    for round in 0..2 {
        let mut surfaces = Vec::new();
        let mut rays = Vec::new();
        let mut attributes = Vec::new();
        let mut expected = Vec::new();
        for (i, &(kind, direct)) in cases.iter().enumerate() {
            let positive = (i + round) % 2 == 0;
            let z = 10.0 + round as f32 * 2.0;
            surfaces.push(SurfaceWorkItem {
                hit: u32::from(i != 18),
                position: [1.0, 2.0, z, 1.0],
                position_error: [0.125, 0.125, 0.25, 0.0],
                normal: [1.0, 0.0, 0.0, 0.0], // Distinct from geometric normal (bump).
                geometric_normal: [0.0, 0.0, 1.0, 0.0],
                flags: u32::from(i == 14),
                attributes_eval_work_item: (i * 3) as u32,
                light_sample_origin: [99.0, 98.0, 97.0],
                ..Zeroable::zeroed()
            });
            rays.push(RayWorkItem {
                pixel_index: i as u32,
                direction: [0.5, 0.0, if positive { -2.0 } else { 2.0 }, 0.0],
                depth: if i == 17 { 2 } else { 0 },
                ..Zeroable::zeroed()
            });
            let mut item = evaluated(kind);
            if i == 15 {
                item.values[1][0] = 0.0;
                item.values[2][0] = 0.0;
            }
            if i == 16 {
                item.values[2][0] = 0.0;
            }
            if kind == MaterialKind::Mix {
                item.selected_child_work_item = (i * 3 + 1) as u32;
            }
            attributes.push(item);
            let mut child = evaluated(if i == 13 {
                MaterialKind::Mix
            } else {
                MaterialKind::DiffuseTransmission
            });
            child.selected_child_work_item = (i * 3 + 2) as u32;
            attributes.push(child);
            attributes.push(evaluated(MaterialKind::DiffuseTransmission));
            let transmission =
                kind == MaterialKind::DiffuseTransmission || kind == MaterialKind::Mix;
            let origin_z = if matches!(kind, MaterialKind::Dielectric | MaterialKind::Subsurface) {
                z
            } else if positive != transmission {
                f32::from_bits((z + 0.25).to_bits() + 1)
            } else {
                f32::from_bits((z - 0.25).to_bits() - 1)
            };
            expected.push(direct.then_some([1.0, 2.0, origin_z]));
        }
        let mut counters = QueueCounters {
            material: state(count, count),
            direct: state(0, count),
            sample_area_light: state(0, count),
            sample_image_infinite_light: state(0, count),
            sample_other_light: state(0, count),
            ..Zeroable::zeroed()
        };
        // Every scatter queue uses the same capacity, including non-direct kinds.
        for queue in [
            &mut counters.scatter_diffuse,
            &mut counters.scatter_diffuse_transmission,
            &mut counters.scatter_conductor,
            &mut counters.scatter_dielectric,
            &mut counters.scatter_thin_dielectric,
            &mut counters.scatter_measured,
            &mut counters.scatter_coated,
        ] {
            queue.capacity = count;
        }
        gpu.write(ResourceId::QueueCounters, &[counters]);
        gpu.write(
            ResourceId::MaterialRayQueue,
            &(0..count).collect::<Vec<_>>(),
        );
        gpu.write(ResourceId::Surface, &surfaces);
        gpu.write(ResourceId::CurrentRay, &rays);
        gpu.write(ResourceId::AttributesEvalWorkItems, &attributes);
        gpu.write(
            ResourceId::PixelSampleState,
            &vec![
                PixelSampleState {
                    lambda: [550.0; 4],
                    ..Zeroable::zeroed()
                };
                count as usize
            ],
        );
        gpu.run(ComputeStageId::ClassifySurfaceScatter, 1);
        let actual = gpu.read::<SurfaceWorkItem>(ResourceId::Surface, count as usize);
        assert_eq!(
            gpu.read::<QueueCounters>(ResourceId::QueueCounters, 1)[0]
                .direct
                .count as usize,
            expected.iter().flatten().count()
        );
        for (i, origin) in expected.iter().enumerate() {
            assert_eq!(
                actual[i].light_sample_origin,
                origin.unwrap_or([99.0, 98.0, 97.0]),
                "case {i}, round {round}"
            );
        }
        gpu.run(ComputeStageId::SelectSurfaceLight, 1);
        gpu.run(ComputeStageId::PrepareQueueDispatch, 1);
        gpu.run(ComputeStageId::SampleAreaLight, 1);
        gpu.run(ComputeStageId::SampleImageInfiniteLight, 1);
        gpu.run(ComputeStageId::SampleOtherLight, 1);
        for (i, origin) in expected.iter().enumerate() {
            if let Some(origin) = origin {
                gpu.assert_point_sample(i, *origin);
            }
        }
    }

    // Medium scatter work selects and samples through the same typed queues,
    // then ScatterMedium consumes that sample to publish its shadow ray.
    let medium_origin = [1.0, 2.0, 10.0];
    gpu.write(
        ResourceId::CurrentRay,
        &[RayWorkItem {
            origin: [medium_origin[0], medium_origin[1], medium_origin[2], 1.0],
            direction: [0.0, 0.0, 1.0, 0.0],
            beta: [1.0; 4],
            r_u: [1.0; 4],
            r_l: [1.0; 4],
            pixel_index: 0,
            medium_id: 0,
            ..Zeroable::zeroed()
        }],
    );
    gpu.write(ResourceId::MediumScatterQueue, &[0u32]);
    gpu.write(
        ResourceId::PixelSampleState,
        &[PixelSampleState {
            lambda: [550.0; 4],
            direct: [0.3, 0.4, 0.5, 0.0],
            indirect: [0.2, 0.3, 0.4, 0.0],
            ..Zeroable::zeroed()
        }],
    );
    gpu.write(
        ResourceId::QueueCounters,
        &[QueueCounters {
            medium_scatter: state(1, 3),
            next: state(0, 3),
            shadow: state(0, 3),
            sample_area_light: state(0, 3),
            sample_image_infinite_light: state(0, 3),
            sample_other_light: state(0, 3),
            ..Zeroable::zeroed()
        }],
    );
    gpu.run(ComputeStageId::SelectMediumLight, 1);
    gpu.run(ComputeStageId::PrepareQueueDispatch, 1);
    gpu.run(ComputeStageId::SampleAreaLight, 1);
    gpu.run(ComputeStageId::SampleImageInfiniteLight, 1);
    gpu.run(ComputeStageId::SampleOtherLight, 1);
    gpu.assert_point_sample(0, medium_origin);
    gpu.run(ComputeStageId::PrepareQueueDispatch, 1);
    gpu.run(ComputeStageId::ScatterMedium, 1);
    assert_eq!(
        gpu.read::<QueueCounters>(ResourceId::QueueCounters, 1)[0]
            .shadow
            .count,
        1
    );

    // Surface and medium selectors can fill the same typed batch without
    // overwriting one another; their source kinds preserve distinct origins.
    let surface_origin = [1.0, 2.0, 10.0];
    let mixed_medium_origin = [0.0, 1.0, 10.0];
    gpu.write(
        ResourceId::CurrentRay,
        &[
            RayWorkItem {
                pixel_index: 0,
                ..Zeroable::zeroed()
            },
            RayWorkItem {
                pixel_index: 1,
                origin: [
                    mixed_medium_origin[0],
                    mixed_medium_origin[1],
                    mixed_medium_origin[2],
                    1.0,
                ],
                ..Zeroable::zeroed()
            },
        ],
    );
    gpu.write(
        ResourceId::Surface,
        &[SurfaceWorkItem {
            position: [surface_origin[0], surface_origin[1], surface_origin[2], 1.0],
            light_sample_origin: surface_origin,
            ..Zeroable::zeroed()
        }],
    );
    gpu.write(ResourceId::DirectEvalQueue, &[0u32]);
    gpu.write(ResourceId::MediumScatterQueue, &[1u32]);
    gpu.write(
        ResourceId::PixelSampleState,
        &[
            PixelSampleState {
                lambda: [550.0; 4],
                direct: [0.2, 0.3, 0.4, 0.0],
                ..Zeroable::zeroed()
            },
            PixelSampleState {
                lambda: [550.0; 4],
                direct: [0.8, 0.3, 0.4, 0.0],
                ..Zeroable::zeroed()
            },
        ],
    );
    gpu.write(
        ResourceId::QueueCounters,
        &[QueueCounters {
            direct: state(1, 2),
            medium_scatter: state(1, 2),
            sample_area_light: state(0, 2),
            sample_image_infinite_light: state(0, 2),
            sample_other_light: state(0, 2),
            ..Zeroable::zeroed()
        }],
    );
    gpu.run(ComputeStageId::SelectSurfaceLight, 1);
    gpu.run(ComputeStageId::SelectMediumLight, 1);
    let counters = gpu.read::<QueueCounters>(ResourceId::QueueCounters, 1)[0];
    assert_eq!(counters.sample_other_light.count, 2);
    let work = gpu.read::<LightSampleWorkItem>(ResourceId::LightSampleWork, 6);
    assert_eq!((work[4].ray_index, work[4].source_kind), (0, 0));
    assert_eq!((work[5].ray_index, work[5].source_kind), (1, 1));
    gpu.run(ComputeStageId::PrepareQueueDispatch, 1);
    gpu.run(ComputeStageId::SampleOtherLight, 1);
    gpu.assert_point_sample(0, surface_origin);
    gpu.assert_point_sample(1, mixed_medium_origin);

    // A real BSSRDF exit replaces a different entry surface and deliberately
    // invalid/stale material-work-item index. The consumer must not resolve it.
    let data = BSSRDFTableData::from_flat(&[tabulate_bssrdf(0.0, 1.33)]).unwrap();
    gpu.write(ResourceId::BSSRDFTable, &data.records);
    gpu.write(ResourceId::BSSRDFValues, &data.values);
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    gpu.write(
        ResourceId::Instance,
        &[Instance {
            world_from_object: identity,
            normal_from_object: identity,
            ..Zeroable::zeroed()
        }],
    );
    gpu.write(
        ResourceId::Geometry,
        &[Geometry {
            vertex_count: 3,
            index_count: 3,
            page: 1,
            ..Zeroable::zeroed()
        }],
    );
    gpu.write(ResourceId::Index, &[0u32, 2, 1]);
    gpu.write(
        ResourceId::Vertex,
        &[
            [0.0, 3.0, 0.0, 1.0],
            [4.0, 3.0, 0.0, 1.0],
            [0.0, 3.0, 8.0, 1.0],
        ]
        .map(|position| Vertex {
            position,
            normal: [0.0, 1.0, 0.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 0.0],
            ..Zeroable::zeroed()
        }),
    );
    let works = (0..3)
        .map(|i| BSSRDFProbeWorkItem {
            position: [0.75, 2.5, 4.0, 1.0],
            normal: [0.0, 0.0, 1.0, 0.0],
            sigma_t: [1.0; 4],
            rho: [0.8; 4],
            pixel_index: i,
            ray_index: i,
            ..Zeroable::zeroed()
        })
        .collect::<Vec<_>>();
    gpu.write(ResourceId::BSSRDFWork, &[state(3, 3)]);
    gpu.context.queue.write_buffer(
        &gpu.buffers[&ResourceId::BSSRDFWork],
        16,
        cast_slice(&works),
    );
    gpu.write(
        ResourceId::BSSRDFResults,
        &(0..3)
            .map(|i| BSSRDFProbeResult {
                position: [1.0, 3.0, 4.0, 1.0],
                position_error: [0.125, 0.25, 0.125, 0.0],
                normal: [0.0, 1.0, 0.0, 0.0],
                barycentric: [0.25, 0.5, 0.25, 0.0],
                valid: u32::from(i != 1),
                reservoir_probability: if i == 2 { 0.0 } else { 1.0 },
                ..Zeroable::zeroed()
            })
            .collect::<Vec<_>>(),
    );
    gpu.write(
        ResourceId::Surface,
        &vec![
            SurfaceWorkItem {
                attributes_eval_work_item: u32::MAX,
                position: [-9.0, -8.0, -7.0, 1.0],
                light_sample_origin: [99.0, 98.0, 97.0],
                ..Zeroable::zeroed()
            };
            3
        ],
    );
    gpu.write(
        ResourceId::CurrentRay,
        &(0..3)
            .map(|i| RayWorkItem {
                pixel_index: i,
                beta: [1.0; 4],
                r_u: [1.0; 4],
                ..Zeroable::zeroed()
            })
            .collect::<Vec<_>>(),
    );
    gpu.write(
        ResourceId::QueueCounters,
        &[QueueCounters {
            direct: state(0, 3),
            medium_scatter: state(1, 3),
            sample_area_light: state(0, 3),
            sample_image_infinite_light: state(0, 3),
            sample_other_light: state(0, 3),
            ..Zeroable::zeroed()
        }],
    );
    gpu.run(ComputeStageId::PrepareSubsurfaceExit, 1);
    let surfaces = gpu.read::<SurfaceWorkItem>(ResourceId::Surface, 3);
    let origin = [1.0, f32::from_bits(3.25f32.to_bits() + 1), 4.0];
    assert_eq!(surfaces[0].light_sample_origin, origin);
    assert_eq!(surfaces[0].flags, 16);
    assert_eq!(surfaces[1].light_sample_origin, [99.0, 98.0, 97.0]);
    assert_eq!(surfaces[2].light_sample_origin, [99.0, 98.0, 97.0]);
    assert_eq!(
        gpu.read::<QueueCounters>(ResourceId::QueueCounters, 1)[0]
            .direct
            .count,
        1
    );
    gpu.run(ComputeStageId::SelectSurfaceLight, 1);
    gpu.run(ComputeStageId::PrepareQueueDispatch, 1);
    gpu.run(ComputeStageId::SampleAreaLight, 1);
    gpu.run(ComputeStageId::SampleImageInfiniteLight, 1);
    gpu.run(ComputeStageId::SampleOtherLight, 1);
    gpu.assert_point_sample(0, origin);
    assert_eq!(
        gpu.read::<QueueCounters>(ResourceId::QueueCounters, 1)[0]
            .medium_scatter
            .count,
        1
    );

    // Sample/tile reuse clears all stored origins; light-free classification
    // cannot publish an unprepared or previous-sample origin.
    gpu.run(ComputeStageId::PrepareSample, count.div_ceil(8));
    assert!(gpu
        .read::<SurfaceWorkItem>(ResourceId::Surface, count as usize)
        .iter()
        .all(|s| s.light_sample_origin == [0.0; 3]));
    gpu.write(
        ResourceId::LightSamplingParams,
        &[LightTableUniform {
            light_count: 0,
            ..Zeroable::zeroed()
        }],
    );
    gpu.write(
        ResourceId::QueueCounters,
        &[QueueCounters {
            material: state(1, count),
            direct: state(0, count),
            scatter_diffuse: state(0, count),
            ..Zeroable::zeroed()
        }],
    );
    gpu.write(
        ResourceId::Surface,
        &[SurfaceWorkItem {
            hit: 1,
            ..Zeroable::zeroed()
        }],
    );
    gpu.write(
        ResourceId::AttributesEvalWorkItems,
        &[evaluated(MaterialKind::Diffuse)],
    );
    gpu.write(ResourceId::CurrentRay, &[RayWorkItem::zeroed()]);
    gpu.run(ComputeStageId::ClassifySurfaceScatter, 1);
    assert_eq!(
        gpu.read::<QueueCounters>(ResourceId::QueueCounters, 1)[0]
            .direct
            .count,
        0
    );
}
