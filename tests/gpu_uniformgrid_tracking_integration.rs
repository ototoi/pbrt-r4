#![cfg(feature = "webgpu")]

use std::collections::BTreeMap;
use std::mem::size_of;

use bytemuck::{bytes_of, cast_slice, Zeroable};
use pbrt_r4::gpu::webgpu::abi::{
    DenseSpectrum, Instance, MediumRecord, PixelSampleState, QueueCounters, RayWorkItem,
    RenderError, ShadowRayWorkItem, SurfaceWorkItem, UniformGridMediumRecord, ViewportUniform,
};
use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::shader::{compose_source, resource_bindings};
use pbrt_r4::gpu::webgpu::stages::{
    canonical_wavefront_bindings, Access, BindingClass, BindingSpec, RequiredLimits, ResourceId,
};
use wgpu::util::DeviceExt;

const PRIMARY_SHADER: &str = include_str!("../src/gpu/webgpu/shaders/sample_medium.wgsl");
const SHADOW_SHADER: &str = include_str!("../src/gpu/webgpu/shaders/intersect_shadow.wgsl");

#[derive(Clone, Copy)]
struct ProbeInput {
    a: [f32; 4],
    s: [f32; 4],
    draws: [f32; 4],
    distance: f32,
    majorant: f32,
    cells: u32,
    shadow_weight_scale: f32,
    infinite: bool,
}

impl Default for ProbeInput {
    fn default() -> Self {
        Self {
            a: [0.0; 4],
            s: [0.0; 4],
            draws: [0.9; 4],
            distance: 0.2,
            majorant: 2.0,
            cells: 1,
            shadow_weight_scale: 1.0,
            infinite: false,
        }
    }
}

struct ProbeResult {
    ray: RayWorkItem,
    counters: QueueCounters,
    shadow: ShadowRayWorkItem,
    pixel: PixelSampleState,
}

#[test]
#[ignore = "requires a WebGPU adapter"]
fn primary_uniform_grid_events_match_v4_spectral_weights() {
    let source = fixed_random_source(PRIMARY_SHADER);
    let bindings = probe_bindings(&source);
    let context = Context::new(RequiredLimits::from_bindings(&bindings).unwrap(), 0, 0).unwrap();
    let (layout, pipeline) = probe_pipeline(&context.device, &source, &bindings, "sample_medium");
    let a = [0.25, 0.5, 0.75, 1.0];
    let s = [0.75, 0.5, 1.0, 1.5];
    let first_distance = 1.0 - (-0.2_f32).exp();

    for mode in [0.05, 0.3, 0.9] {
        let input = ProbeInput {
            a,
            s,
            draws: [first_distance, mode, 0.9, 0.9],
            ..Default::default()
        };
        let ProbeResult { ray, counters, .. } =
            run_stage(&context, &layout, &pipeline, &bindings, &input);
        if mode < 0.125 {
            assert_eq!(ray.beta, [0.0; 4]);
            assert_eq!(counters.medium_scatter.count, 0);
        } else if mode < 0.5 {
            assert_eq!(counters.medium_scatter.count, 1);
            assert_close(ray.origin[2], 0.1);
            for lane in 0..4 {
                let sigma_maj = 2.0 * (a[lane] + s[lane]);
                let t_maj = (-0.1 * sigma_maj).exp();
                let pr = (-0.2_f32).exp() * s[0];
                let expected = t_maj * s[lane] / pr;
                assert_close(ray.beta[lane], expected);
                assert_close(ray.r_u[lane], expected);
                assert_close(ray.r_l[lane], 1.0);
            }
        } else {
            assert_eq!(counters.medium_scatter.count, 0);
            for lane in 0..4 {
                let sigma_t = a[lane] + s[lane];
                let sigma_maj = 2.0 * sigma_t;
                let t_maj = (-0.1 * sigma_maj).exp();
                let pr = (-0.2_f32).exp() * (2.0 - a[0] - s[0]);
                let tail_ratio = (-0.1 * sigma_maj).exp() / (-0.2_f32).exp();
                let expected = t_maj * (sigma_maj - sigma_t) / pr * tail_ratio;
                assert_close(ray.beta[lane], expected);
                assert_close(ray.r_u[lane], expected);
                assert_close(ray.r_l[lane], t_maj * sigma_maj / pr * tail_ratio);
            }
        }
    }

    for (a, s, majorant, cells, distance) in [
        (a, s, 2.0, 1, 0.2),
        ([0.0, 0.5, 0.75, 1.0], [0.0, 0.5, 1.0, 1.5], 2.0, 1, 0.2),
        ([0.0; 4], [128.0, 129.0, 130.0, 131.0], 1.0, 16, 2.0),
        (a, s, 0.0, 1, 0.2),
    ] {
        let u = if cells == 16 {
            1.0 - f32::EPSILON / 2.0
        } else {
            0.9
        };
        let input = ProbeInput {
            a,
            s,
            draws: [u, 0.9, u, 0.9],
            distance,
            majorant,
            cells,
            ..Default::default()
        };
        let ProbeResult { ray, counters, .. } =
            run_stage(&context, &layout, &pipeline, &bindings, &input);
        assert_eq!(counters.medium_scatter.count, 0);
        for lane in 0..4 {
            let expected = (distance * majorant * (a[0] + s[0] - a[lane] - s[lane])).exp();
            assert_close(ray.beta[lane], expected);
            assert_close(ray.r_u[lane], expected);
            assert_close(ray.r_l[lane], expected);
        }
    }
}

#[test]
#[ignore = "requires a WebGPU adapter"]
fn shadow_uniform_grid_ratio_tracking_and_roulette_match_v4() {
    // A miss isolates medium tracking from geometry; real boundary traversal
    // is exercised by gpu_medium_segment_integration and the plume renders.
    let mut stage = SHADOW_SHADER.to_owned();
    let start = stage.find("    var query: ray_query;").unwrap();
    let end = stage.find("    let segment_distance = select(").unwrap();
    stage.replace_range(
        start..end,
        r#"
        let intersection = TestShadowIntersection(0u, 0u, vec2<f32>(0.0), 0.0);
        let hit = false;
    "#,
    );
    stage.push_str(
        r#"
        struct TestShadowIntersection {
            instance_custom_data: u32,
            primitive_index: u32,
            barycentrics: vec2<f32>,
            t: f32,
        }
    "#,
    );
    let source = fixed_random_source(&stage);
    let bindings = probe_bindings(&source);
    let context = Context::new(RequiredLimits::from_bindings(&bindings).unwrap(), 0, 0).unwrap();
    let (layout, pipeline) =
        probe_pipeline(&context.device, &source, &bindings, "intersect_shadow");
    let a = [0.25, 0.5, 0.75, 1.0];
    let s = [0.75, 0.5, 1.0, 1.5];
    for infinite in [false, true] {
        let distance = if infinite { 1.0 } else { 0.2 };
        let input = ProbeInput {
            a,
            s,
            draws: [1.0 - (-0.2_f32).exp(), 0.9, 0.9, 0.9],
            distance,
            infinite,
            ..Default::default()
        };
        let result = run_stage(&context, &layout, &pipeline, &bindings, &input);
        for lane in 0..4 {
            let sigma_t = a[lane] + s[lane];
            let sigma_maj = 2.0 * sigma_t;
            let t_maj = (-0.1 * sigma_maj).exp();
            let pr = 2.0 * (-0.2_f32).exp();
            let tail_ratio =
                (-(distance - 0.1) * sigma_maj).exp() / (-2.0 * (distance - 0.1)).exp();
            let expected = t_maj * (sigma_maj - sigma_t) / pr * tail_ratio;
            assert_close(result.shadow.transmittance[lane], expected);
            assert_close(result.shadow.inv_w_u[lane], expected);
            assert_close(
                result.shadow.inv_w_l[lane],
                t_maj * sigma_maj / pr * tail_ratio,
            );
        }
        assert_shadow_radiance(&result);
    }

    for roulette in [0.1, 0.9] {
        let majorant = 1.01;
        let input = ProbeInput {
            a: [0.5; 4],
            s: [0.5; 4],
            majorant,
            draws: [1.0 - (-0.1_f32 * majorant).exp(), 0.9, 0.9, roulette],
            // Incoming direct weights must not be used for the RR threshold.
            shadow_weight_scale: 1e-4,
            ..Default::default()
        };
        let result = run_stage(&context, &layout, &pipeline, &bindings, &input);
        let ratio = (majorant - 1.0) / majorant;
        for lane in 0..4 {
            assert_close(result.shadow.inv_w_u[lane], ratio);
            assert_close(result.shadow.inv_w_l[lane], 1.0);
            assert_close(
                result.shadow.transmittance[lane],
                if roulette < 0.75 { 0.0 } else { ratio / 0.25 },
            );
        }
        assert_shadow_radiance(&result);
    }

    for (a, s, majorant, cells, distance) in [
        (a, s, 2.0, 1, 0.2),
        ([0.0, 0.5, 0.75, 1.0], [0.0, 0.5, 1.0, 1.5], 2.0, 1, 0.2),
        ([0.0; 4], [128.0, 129.0, 130.0, 131.0], 1.0, 16, 2.0),
        (a, s, 0.0, 1, 0.2),
    ] {
        let u = if cells == 16 {
            1.0 - f32::EPSILON / 2.0
        } else {
            0.9
        };
        let input = ProbeInput {
            a,
            s,
            majorant,
            cells,
            distance,
            draws: [u, 0.9, u, 0.9],
            ..Default::default()
        };
        let result = run_stage(&context, &layout, &pipeline, &bindings, &input);
        for lane in 0..4 {
            let expected = (distance * majorant * (a[0] + s[0] - a[lane] - s[lane])).exp();
            assert_close(result.shadow.transmittance[lane], expected);
            assert_close(result.shadow.inv_w_u[lane], expected);
            assert_close(result.shadow.inv_w_l[lane], expected);
        }
        assert_shadow_radiance(&result);
    }
}

fn assert_shadow_radiance(result: &ProbeResult) {
    let denominator = (0..4)
        .map(|lane| {
            result.shadow.r_u[lane] * result.shadow.inv_w_u[lane]
                + result.shadow.r_l[lane] * result.shadow.inv_w_l[lane]
        })
        .sum::<f32>()
        / 4.0;
    for lane in 0..4 {
        assert_close(
            result.pixel.radiance[lane],
            result.shadow.transmittance[lane] / denominator,
        );
    }
}

// Only the random stream is replaced: event selection, lookup, DDA, queue
// writes, and spectral updates execute the production stage unchanged.
fn fixed_random_source(stage: &str) -> String {
    let mut source = compose_source(stage);
    let start = source.find("fn random_medium(").unwrap();
    let body = start + source[start..].find('{').unwrap();
    let mut depth = 1;
    let end = source[body + 1..]
        .char_indices()
        .find_map(|(index, ch)| {
            match ch {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            (depth == 0).then_some(body + 1 + index)
        })
        .unwrap();
    source.replace_range(
        body + 1..end,
        r#"
        if (stream == 0x10000u || stream == 0x20000u) { return test_draws[0].x; }
        if (stream == 0x10001u) { return test_draws[0].y; }
        if (stream >= 0x30000u) { return test_draws[0].w; }
        return test_draws[0].z;
    "#,
    );
    source.push_str("\n@group(0) @binding(77) var<storage, read> test_draws: array<vec4<f32>>;\n");
    source
}

fn probe_bindings(source: &str) -> Vec<BindingSpec> {
    let mut canonical = canonical_wavefront_bindings();
    canonical.push(BindingSpec {
        group: 0,
        binding: 77,
        resource: ResourceId::VolumeData,
        class: BindingClass::Storage,
        access: Access::Read,
    });
    resource_bindings(source)
        .into_iter()
        .map(|(group, binding)| {
            assert_eq!(group, 0);
            *canonical
                .iter()
                .find(|spec| spec.group == group && spec.binding == binding)
                .unwrap()
        })
        .collect()
}

fn probe_pipeline(
    device: &wgpu::Device,
    source: &str,
    bindings: &[BindingSpec],
    entry_point: &str,
) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
    let entries: Vec<_> = bindings
        .iter()
        .map(|spec| wgpu::BindGroupLayoutEntry {
            binding: spec.binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: match spec.class {
                    BindingClass::Uniform => wgpu::BufferBindingType::Uniform,
                    BindingClass::Storage => wgpu::BufferBindingType::Storage {
                        read_only: !spec.access.permits_write(),
                    },
                    _ => panic!("unexpected probe resource"),
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("uniform grid tracking probe"),
        entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("uniform grid tracking probe"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("uniform grid tracking probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("uniform grid tracking probe"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    (layout, pipeline)
}

fn run_stage(
    context: &Context,
    layout: &wgpu::BindGroupLayout,
    pipeline: &wgpu::ComputePipeline,
    bindings: &[BindingSpec],
    input: &ProbeInput,
) -> ProbeResult {
    let ProbeInput {
        a,
        s,
        draws,
        distance,
        majorant,
        cells,
        ..
    } = *input;
    let mut values = BTreeMap::new();
    let mut viewport = ViewportUniform::zeroed();
    viewport.full_width = 1;
    viewport.tile_width = 1;
    viewport.max_depth = 4;
    viewport.medium_scattering_enabled = 1;
    values.insert(1, bytes_of(&viewport).to_vec());
    values.insert(6, bytes_of(&Instance::zeroed()).to_vec());
    let mut surface = SurfaceWorkItem::zeroed();
    surface.hit = 1;
    surface.t = distance;
    values.insert(8, bytes_of(&surface).to_vec());
    let mut counters = QueueCounters::zeroed();
    counters.medium_active.count = 1;
    counters.shadow_active.count = 1;
    counters.medium_scatter.capacity = 1;
    values.insert(10, bytes_of(&counters).to_vec());
    values.insert(11, bytes_of(&RenderError::zeroed()).to_vec());
    let mut pixel = PixelSampleState::zeroed();
    pixel.lambda = [400.0, 500.0, 600.0, 700.0];
    values.insert(12, bytes_of(&pixel).to_vec());
    let mut ray = RayWorkItem::zeroed();
    ray.direction = [0.0, 0.0, 1.0, 0.0];
    ray.beta = [1.0; 4];
    ray.r_u = [1.0; 4];
    ray.r_l = [1.0; 4];
    values.insert(13, bytes_of(&ray).to_vec());
    let mut shadow = ShadowRayWorkItem::zeroed();
    shadow.direction = ray.direction;
    shadow.max_t = if input.infinite { f32::MAX } else { distance };
    shadow.infinite_distance = u32::from(input.infinite);
    shadow.direct = [1.0; 4];
    shadow.r_u = [2.0, 3.0, 4.0, 5.0].map(|value| value * input.shadow_weight_scale);
    shadow.r_l = [5.0, 4.0, 3.0, 2.0].map(|value| value * input.shadow_weight_scale);
    shadow.transmittance = [1.0; 4];
    shadow.inv_w_u = [1.0; 4];
    shadow.inv_w_l = [1.0; 4];
    values.insert(15, bytes_of(&shadow).to_vec());
    let mut spectra = [DenseSpectrum::zeroed(); 2];
    for (spectrum, coefficients) in spectra.iter_mut().zip([a, s]) {
        for (index, value) in [40, 140, 240, 340].into_iter().zip(coefficients) {
            spectrum.samples[index] = value;
        }
    }
    values.insert(34, cast_slice(&spectra).to_vec());
    let mut medium = MediumRecord::zeroed();
    medium.kind = 1;
    medium.sigma_s = 1;
    values.insert(61, bytes_of(&medium).to_vec());
    let grid = UniformGridMediumRecord {
        bounds_min: [-1.0, -1.0, if cells == 16 { 0.0 } else { -1.0 }, 0.0],
        bounds_max: [1.0, 1.0, if cells == 16 { 2.0 } else { 1.0 }, 0.0],
        resolution: [2, 2, 2, 0],
        majorant_resolution: [1, 1, cells, 0],
        density_offset_count: [0, 8, 0, 0],
        majorant_offset_count: [8, cells, 0, 0],
        medium_from_world: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };
    values.insert(75, bytes_of(&grid).to_vec());
    let mut volume = vec![1.0_f32; 8];
    volume.extend(std::iter::repeat_n(majorant, cells as usize));
    values.insert(76, cast_slice(&volume).to_vec());
    values.insert(77, bytes_of(&draws).to_vec());
    let buffers: BTreeMap<_, _> = bindings
        .iter()
        .map(|spec| {
            let data = values
                .remove(&spec.binding)
                .unwrap_or_else(|| vec![0; 1024]);
            let usage = if spec.class == BindingClass::Uniform {
                wgpu::BufferUsages::UNIFORM
            } else {
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
            };
            (
                spec.binding,
                context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("uniform grid probe data"),
                        contents: &data,
                        usage,
                    }),
            )
        })
        .collect();
    let entries: Vec<_> = buffers
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform grid tracking probe"),
            layout,
            entries: &entries,
        });
    let ray_size = size_of::<RayWorkItem>() as u64;
    let counter_size = size_of::<QueueCounters>() as u64;
    let shadow_size = size_of::<ShadowRayWorkItem>() as u64;
    let pixel_size = size_of::<PixelSampleState>() as u64;
    let total = ray_size + counter_size + shadow_size + pixel_size + 16;
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform grid tracking readback"),
        size: total,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = context.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    let mut offset = 0;
    for (binding, size) in [
        (13, ray_size),
        (10, counter_size),
        (15, shadow_size),
        (12, pixel_size),
        (11, 16),
    ] {
        if let Some(buffer) = buffers.get(&binding) {
            encoder.copy_buffer_to_buffer(buffer, 0, &readback, offset, size);
        }
        offset += size;
    }
    context.queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let ray = bytemuck::pod_read_unaligned(&mapped[..ray_size as usize]);
    let counters = bytemuck::pod_read_unaligned(
        &mapped[ray_size as usize..(ray_size + counter_size) as usize],
    );
    let shadow = bytemuck::pod_read_unaligned(
        &mapped
            [(ray_size + counter_size) as usize..(ray_size + counter_size + shadow_size) as usize],
    );
    let pixel = bytemuck::pod_read_unaligned(
        &mapped[(ray_size + counter_size + shadow_size) as usize..(total - 16) as usize],
    );
    let error: RenderError = bytemuck::pod_read_unaligned(&mapped[(total - 16) as usize..]);
    assert_eq!(error.value, 0);
    ProbeResult {
        ray,
        counters,
        shadow,
        pixel,
    }
}

fn assert_close(actual: f32, expected: f32) {
    assert!(actual.is_finite());
    assert!(
        (actual - expected).abs() <= 2e-5 * expected.abs().max(1.0),
        "actual {actual}, expected {expected}"
    );
}
