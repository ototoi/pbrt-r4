#![cfg(feature = "webgpu")]

use bytemuck::{bytes_of, cast_slice, Zeroable};
use pbrt_r4::base::light::LightBounds as CpuBounds;
use pbrt_r4::gpu::flat::{Bounds3, LightBounds};
use pbrt_r4::gpu::webgpu::abi::{
    LightTableUniform, LIGHT_SAMPLER_KIND_EXHAUSTIVE, LIGHT_SAMPLER_KIND_POWER,
};
use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::light_sampler::{pack_exhaustive_bounds, pack_power_weights};
use pbrt_r4::util::base::{Normal3f, Point3f, Vector3f};
use pbrt_r4::util::geometry::Bounds3f;
use wgpu::util::DeviceExt;

const DRAWS: usize = 4096;
const PROBE: &str = r#"
@group(0) @binding(0) var<uniform> light_table: LightTableUniform;
@group(0) @binding(1) var<storage, read> light_sampler_data: array<u32>;
struct ProbeContext { p: vec4<f32>, n: vec4<f32> };
@group(0) @binding(2) var<uniform> ctx: ProbeContext;
@group(0) @binding(3) var<storage, read_write> results: array<vec4<u32>>;
var<private> light_bvh_header: array<u32, 8>;
var<private> light_bvh_nodes: array<u32, 8>;
var<private> light_bvh_leaves: array<u32, 1>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&results)) { return; }
    let u = (f32(id.x) + 0.5) / f32(arrayLength(&results));
    let sample = light_sampler_sample(u, ctx.p.xyz, ctx.n.xyz);
    let pmf = light_sampler_pmf(sample.index, ctx.p.xyz, ctx.n.xyz);
    let query = id.x % (light_table.light_count + 1u);
    results[id.x] = vec4<u32>(sample.index, bitcast<u32>(sample.pmf), bitcast<u32>(pmf),
        bitcast<u32>(light_sampler_pmf(query, ctx.p.xyz, ctx.n.xyz)));
}
"#;

struct Probe {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

impl Probe {
    fn new() -> Self {
        let source = [
            include_str!("../src/gpu/webgpu/shaders/types.wgsl"),
            include_str!("../src/gpu/webgpu/shaders/lib/light_samplers.wgsl"),
            PROBE,
        ]
        .join("\n");
        Self::with_source(source)
    }

    fn with_source(source: String) -> Self {
        let adapter = Context::request_adapter(None).unwrap();
        eprintln!("light sampler numeric probe adapter: {:?}", adapter.info);
        let (device, queue) = pollster::block_on(
            adapter
                .adapter
                .request_device(&wgpu::DeviceDescriptor::default()),
        )
        .unwrap();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("light sampler numeric probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("light sampler numeric probe"),
            layout: None,
            module: &module,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            device,
            queue,
            pipeline,
        }
    }

    fn run(
        &self,
        kind: u32,
        finite: u32,
        infinite: u32,
        data: &[u32],
        p: [f32; 3],
        n: [f32; 3],
    ) -> Vec<[u32; 4]> {
        let table = LightTableUniform {
            light_count: finite + infinite,
            finite_light_count: finite,
            infinite_light_count: infinite,
            light_sampler_kind: kind,
            ..LightTableUniform::zeroed()
        };
        let params = [p[0], p[1], p[2], 0.0, n[0], n[1], n[2], 0.0];
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes_of(&table),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let payload = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let context = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let size = (DRAWS * 16) as u64;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffers = [&uniform, &payload, &context, &output];
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(DRAWS as u32 / 64, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
        self.queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range().unwrap();
        cast_slice::<u8, [u32; 4]>(&mapped).to_vec()
    }
}

fn cpu_importance(bound: LightBounds, p: [f32; 3], n: [f32; 3]) -> f32 {
    let v = |a: [f32; 3]| Vector3f::new(a[0], a[1], a[2]);
    let bound = CpuBounds::new(
        Bounds3f::new(&v(bound.bounds.min), &v(bound.bounds.max)),
        v(bound.direction),
        bound.phi,
        bound.cos_theta_o,
        bound.cos_theta_e,
        bound.two_sided,
    );
    bound.importance(
        Point3f::new(p[0], p[1], p[2]),
        Normal3f::new(n[0], n[1], n[2]),
    )
}

fn assert_probability(actual: f32, expected: f32) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "actual={actual}, expected=NaN");
    } else {
        assert!(
            (actual - expected).abs() < 3e-6,
            "actual={actual}, expected={expected}"
        );
    }
}

fn check_draws(draws: &[[u32; 4]], expected: &[f32]) {
    let mut counts = vec![0; expected.len()];
    for (index, draw) in draws.iter().enumerate() {
        if draw[0] != u32::MAX {
            let handle = draw[0] as usize;
            assert!(handle < expected.len());
            assert!(expected[handle] > 0.0);
            counts[handle] += 1;
            assert_probability(f32::from_bits(draw[1]), expected[handle]);
            assert_probability(f32::from_bits(draw[2]), expected[handle]);
        } else {
            assert_eq!(f32::from_bits(draw[1]), 0.0);
        }
        let handle = index % (expected.len() + 1);
        assert_probability(
            f32::from_bits(draw[3]),
            expected.get(handle).copied().unwrap_or(0.0),
        );
    }
    for (count, probability) in counts.into_iter().zip(expected) {
        if probability.is_nan() {
            assert_eq!(count, 0);
        } else {
            assert!((count as f32 / DRAWS as f32 - probability).abs() <= 2.0 / DRAWS as f32);
        }
    }
}

#[test]
#[ignore = "requires a WebGPU adapter; does not require ray-query support"]
fn wgsl_power_and_exhaustive_samples_and_pmfs_match_v4_distributions() {
    let probe = Probe::new();
    for weights in [
        vec![],
        vec![1.0, 3.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 0.0],
        vec![16777216.0, 1.0, 1.0],
    ] {
        let normalized = if weights.iter().all(|w| *w == 0.0) {
            vec![1.0; weights.len()]
        } else {
            weights.clone()
        };
        let sum = normalized.iter().map(|w| f64::from(*w)).sum::<f64>() as f32;
        let expected = normalized.iter().map(|w| *w / sum).collect::<Vec<_>>();
        let draws = probe.run(
            LIGHT_SAMPLER_KIND_POWER,
            weights.len() as u32,
            0,
            &pack_power_weights(&weights).unwrap(),
            [0.0, 0.0, 3.0],
            [0.0; 3],
        );
        check_draws(&draws, &expected);
    }
    let bound = LightBounds {
        bounds: Bounds3::new([-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]).unwrap(),
        direction: [0.0, 0.0, 1.0],
        phi: 1.0,
        cos_theta_o: 1.0,
        cos_theta_e: 0.0,
        two_sided: false,
    };
    for (bounds, infinite) in [
        (vec![], 0),
        (vec![], 2),
        (vec![bound, LightBounds { phi: 3.0, ..bound }], 0),
        (vec![bound, LightBounds { phi: 3.0, ..bound }], 2),
        (vec![LightBounds { phi: 0.0, ..bound }; 2], 0),
        (vec![LightBounds { phi: 0.0, ..bound }; 2], 1),
        (
            vec![
                bound,
                LightBounds {
                    direction: [0.0, 0.0, -1.0],
                    ..bound
                },
            ],
            0,
        ),
    ] {
        for n in [[0.0; 3], [0.0, 0.0, -1.0]] {
            let p = [0.0, 0.0, 3.0];
            let groups = infinite + u32::from(!bounds.is_empty());
            let p_infinite = infinite as f32 / groups as f32;
            let weights = bounds
                .iter()
                .map(|b| cpu_importance(*b, p, n))
                .collect::<Vec<_>>();
            let sum: f32 = weights.iter().sum();
            let mut expected = weights
                .iter()
                .map(|w| (1.0 - p_infinite) * *w / sum)
                .collect::<Vec<_>>();
            expected.extend((0..infinite).map(|_| p_infinite / infinite as f32));
            let draws = probe.run(
                LIGHT_SAMPLER_KIND_EXHAUSTIVE,
                bounds.len() as u32,
                infinite,
                &pack_exhaustive_bounds(&bounds).unwrap(),
                p,
                n,
            );
            check_draws(&draws, &expected);
        }
    }
}

fn shader_definition<'a>(source: &'a str, declaration: &str) -> &'a str {
    let start = source.find(declaration).unwrap();
    let body = start + source[start..].find('{').unwrap();
    let mut depth = 0;
    for (offset, byte) in source.as_bytes()[body..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=body + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unclosed shader declaration: {declaration}");
}

#[test]
#[ignore = "requires a WebGPU adapter; does not require ray-query support"]
fn wgsl_area_sampling_and_intersection_normals_follow_v4_vertex_orientation() {
    let interaction = include_str!("../src/gpu/webgpu/shaders/lib/interaction.wgsl");
    let triangles = include_str!("../src/gpu/webgpu/shaders/triangle_sampling.wgsl");
    let mut source = include_str!("../src/gpu/webgpu/shaders/types.wgsl").to_owned();
    for name in [
        "instance_orientation_is_reversed",
        "instance_orientation_swaps_handedness",
        "instance_orientation_shape_swaps_handedness",
        "instance_orientation_flips_geometric_normal",
        "intersection_flips_geometric_normal",
        "triangle_intersection_geometric_normal",
    ] {
        source.push_str(shader_definition(interaction, &format!("fn {name}(")));
        source.push('\n');
    }
    source.push_str(shader_definition(triangles, "struct TriangleVertices {"));
    source.push_str(";\n");
    source.push_str(shader_definition(
        triangles,
        "fn triangle_geometric_normal(",
    ));
    source.push_str(r#"
@group(0) @binding(0) var<uniform> light_table: LightTableUniform;
@group(0) @binding(1) var<storage, read> light_sampler_data: array<u32>;
struct ProbeContext { p: vec4<f32>, n: vec4<f32> };
@group(0) @binding(2) var<uniform> ctx: ProbeContext;
@group(0) @binding(3) var<storage, read_write> results: array<vec4<u32>>;
fn vector_at(offset: u32) -> vec3<f32> {
    return vec3<f32>(bitcast<f32>(light_sampler_data[offset]),
        bitcast<f32>(light_sampler_data[offset+1u]), bitcast<f32>(light_sampler_data[offset+2u]));
}
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= light_table.light_count) { return; }
    let transform = mat4x4<f32>(vec4<f32>(vector_at(19u),0.0), vec4<f32>(vector_at(22u),0.0),
        vec4<f32>(vector_at(25u),0.0), vec4<f32>(0.0,0.0,0.0,1.0));
    let tri = TriangleVertices(vec4<f32>(vector_at(0u),1.0), vec4<f32>(vector_at(3u),1.0), vec4<f32>(vector_at(6u),1.0),
        (transform*vec4<f32>(vector_at(9u),0.0)).xyz, (transform*vec4<f32>(vector_at(12u),0.0)).xyz,
        (transform*vec4<f32>(vector_at(15u),0.0)).xyz, light_sampler_data[18u]);
    var normal = triangle_geometric_normal(tri, ctx.p.xyz);
    if (id.x == 1u) {
        var instance: Instance;
        instance.orientation_flags = tri.orientation_flags;
        instance.normal_from_object = transform;
        var geometry: Geometry;
        geometry.intersection_normal_kind = INTERSECTION_NORMAL_KIND_TRIANGLE;
        let object_normal = vector_at(9u)*ctx.p.x + vector_at(12u)*ctx.p.y + vector_at(15u)*ctx.p.z;
        normal = triangle_intersection_geometric_normal(instance, geometry, object_normal,
            normalize(cross(tri.p1.xyz-tri.p0.xyz, tri.p2.xyz-tri.p0.xyz)));
    }
    results[id.x] = vec4<u32>(bitcast<vec3<u32>>(normal), bitcast<u32>(dot(normal,ctx.n.xyz)));
}
"#);
    let probe = Probe::with_source(source);
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let bary = [1.0 / 3.0; 3];
    for (positions, normals, flags, matrix, expected) in [
        (
            positions,
            [[0.0, 0.0, -1.0]; 3],
            0u32,
            identity,
            [0.0, 0.0, -1.0],
        ),
        (
            positions,
            [[0.0, 0.0, 1.0]; 3],
            1,
            identity,
            [0.0, 0.0, 1.0],
        ),
        (positions, [[0.0; 3]; 3], 0, identity, [0.0, 0.0, 1.0]),
        (positions, [[0.0; 3]; 3], 1, identity, [0.0, 0.0, -1.0]),
        (positions, [[0.0; 3]; 3], 2, identity, [0.0, 0.0, -1.0]),
        (positions, [[0.0; 3]; 3], 3, identity, [0.0, 0.0, 1.0]),
        (
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, -1.0]; 3],
            2,
            identity,
            [0.0, 0.0, -1.0],
        ),
        (
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, -1.0]; 3],
            0,
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [-1.0, 0.0, 0.0],
        ),
        (
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
            [[1.0, 0.0, 0.1], [1.0, 0.0, 0.1], [0.0, 0.0, -0.15]],
            0,
            [[1.0 / 3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.5]],
            [0.0, 0.0, 1.0],
        ),
    ] {
        let mut data = positions
            .into_iter()
            .flatten()
            .chain(normals.into_iter().flatten())
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        data.push(flags);
        data.extend(matrix.into_iter().flatten().map(f32::to_bits));
        let output = probe.run(0, 2, 0, &data, bary, expected);
        for row in &output[..2] {
            for (actual, expected) in row[..3].iter().zip(expected) {
                assert!(
                    (f32::from_bits(*actual) - expected).abs() < 1e-6,
                    "normal={row:?}, expected={expected}"
                );
            }
            assert!((f32::from_bits(row[3]) - 1.0).abs() < 1e-6);
        }
    }
}
