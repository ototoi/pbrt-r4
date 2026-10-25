#![cfg(feature = "webgpu")]

//! Parity of `lib/bxdfs/hair.wgsl` against the CPU `HairBxDF`
//! (`pbrt-r4/src/bxdfs/hair.rs`) for f, sample_f and pdf over a parameter
//! grid, using the same shader-on-device harness shape as the bssrdf probe
//! tests. The pigment/reflectance sigma_a resolution is covered by
//! `gpu_hair_material.rs`; here sigma_a arrives already resolved per
//! wavelength, so the probe exercises only HairBxDF's math.

use pbrt_r4::base::bxdf::{TransportMode, BXDF_REFL_TRANS_ALL};
use pbrt_r4::bxdfs::HairBxDF;
use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::stages::RequiredLimits;
use pbrt_r4::util::base::{Float, Point2f, Vector3f};
use pbrt_r4::util::spectrum::SampledSpectrum;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: usize = 64;

/// One probe case: hair BxDF parameters (sigma_a already resolved), one
/// (wo, wi) query pair and Sample_f's u. Laid out as vec4 columns to match
/// the WGSL storage layout exactly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HairProbeInput {
    /// beta_m, beta_n, alpha, eta
    params: [f32; 4],
    sigma_a: [f32; 4],
    /// h, u.x, u.y, 0
    h_and_u: [f32; 4],
    /// xyz direction, w unused
    wo: [f32; 4],
    /// xyz direction, w unused
    wi: [f32; 4],
}

fn probe_source() -> String {
    [
        include_str!("../src/gpu/webgpu/shaders/types.wgsl"),
        include_str!("../src/gpu/webgpu/shaders/lib/vecmath.wgsl"),
        // The pigment/reflance sigma_a resolvers are exercised through the
        // evaluate_attributes pipeline and covered by gpu_hair_material.rs;
        // the probe resolves sigma_a on the CPU side. hair.wgsl still
        // references the RGB conversion and spectrum helpers, so give the
        // probe its own minimal declarations of them: a dummy table binding
        // (never dereferenced by the probe entry) plus the two lane helpers
        // out of spectrum.wgsl.
        r#"
struct HairProbeInput {
    params: vec4<f32>,   // beta_m, beta_n, alpha, eta
    sigma_a: vec4<f32>,
    h_and_u: vec4<f32>,  // h, u.x, u.y, 0
    wo: vec4<f32>,       // xyz direction
    wi: vec4<f32>,       // xyz direction
};
@group(0) @binding(0) var<storage, read> rgb_spectrum_table: array<f32>;
fn sampled_spectrum_average(value: vec4<f32>) -> f32 {
    return dot(value, vec4<f32>(0.25));
}
fn sampled_spectrum_max_component_value(value: vec4<f32>) -> f32 {
    return max(max(value.x, value.y), max(value.z, value.w));
}
// scattering.wgsl's refract() uses the dielectric interface sample helpers
// (lib/bxdfs/dielectric.wgsl), which pull material-table bindings into a
// probe; re-declare the tiny invalid-sample helper instead.
fn invalid_dielectric_interface_sample() -> DielectricInterfaceSample {
    return DielectricInterfaceSample(vec4<f32>(0.0), vec3<f32>(0.0), 0.0, 1.0, 0u, 0u, 0u);
}
@group(0) @binding(1) var<storage, read> hair_probe_inputs: array<HairProbeInput>;
// Three vec4 slots per case keep the storage readback free of WGSL struct
// padding rules: [f, (wi, pdf), (sample_pdf, sample_valid, 0, 0)].
@group(0) @binding(2) var<storage, read_write> hair_probe_outputs: array<vec4<f32>>;
"#,
        include_str!("../src/gpu/webgpu/shaders/lib/scattering.wgsl"),
        include_str!("../src/gpu/webgpu/shaders/lib/color.wgsl"),
        include_str!("../src/gpu/webgpu/shaders/lib/bxdfs/hair.wgsl"),
        r#"
@compute @workgroup_size(64)
fn hair_bxdf_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&hair_probe_inputs)) { return; }
    let input = hair_probe_inputs[index];
    var item = AttributesEvalWorkItem(
        0u, 0xffffffffu, 0xffffffffu, 0u, 0xffffffffu, 0u, 0u, 0u,
        array<vec4<f32>, 10>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
            vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
            vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)));
    item.values[0].x = input.params.x;
    item.values[1].x = input.params.y;
    item.values[2].x = input.params.z;
    item.values[3].x = input.params.w;
    item.values[4] = input.sigma_a;
    let bxdf = hair_bxdf_make(item, input.h_and_u.x);
    let wo = input.wo.xyz;
    let wi = input.wi.xyz;
    let sampled = hair_bxdf_sample_f(bxdf, wo, input.h_and_u.yz);
    let base = index * 3u;
    hair_probe_outputs[base] = hair_bxdf_f(bxdf, wo, wi);
    hair_probe_outputs[base + 1u] = vec4<f32>(sampled.wi, hair_bxdf_pdf(bxdf, wo, wi));
    hair_probe_outputs[base + 2u] = vec4<f32>(sampled.pdf, f32(sampled.valid), 0.0, 0.0);
}
"#,
    ]
    .join("\n")
}

/// The sRGB2Spectrum table never runs in the probe (sigma_a arrives already
/// resolved), so bind a zero-filled buffer with the table's declared size.
fn rgb_table_word_count() -> u32 {
    const RESOLUTION: u32 = 64;
    const SCALE_COUNT: u32 = RESOLUTION;
    const COEFF_COUNT: u32 = 3 * RESOLUTION * RESOLUTION * RESOLUTION * 3;
    // color_space 0..3 => four tables in the shared storage.
    4 * (SCALE_COUNT + COEFF_COUNT)
}

#[test]
#[ignore = "requires a WebGPU adapter"]
fn gpu_hair_bxdf_matches_the_cpu_implementation() {
    let context = Context::new(
        RequiredLimits {
            storage_buffers_per_shader_stage: 2,
            uniform_buffers_per_shader_stage: 0,
            buffers_and_acceleration_structures_per_shader_stage: 2,
            bind_groups: 1,
        },
        0,
        0,
    )
    .expect("a WebGPU adapter is required");
    let device = &context.device;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    const SIGMA_A: [Float; 4] = [0.7, 0.9, 1.2, 1.5];
    const ETA: Float = 1.55;
    const SIGMA_A_F32: [f32; 4] = [0.7, 0.9, 1.2, 1.5];

    let mut cases: Vec<HairProbeInput> = Vec::new();
    let mut expected: Vec<(SampledSpectrum, Float, (Vector3f, Float))> = Vec::new();
    for h in [-0.8, -0.2, 0.0, 0.3, 0.9] {
        for (beta_m, beta_n) in [(0.1, 0.1), (0.3, 0.3), (0.6, 0.25)] {
            for alpha in [0.0, 2.0, 10.0] {
                for wo in [[0.3, 0.9, 0.1], [-0.6, 0.7, -0.2], [0.05, -0.6, 0.8]] {
                    for u in [[0.13, 0.71], [0.555, 0.222]] {
                        let length = Float::sqrt(wo[0] * wo[0] + wo[1] * wo[1] + wo[2] * wo[2]);
                        let wo_local =
                            Vector3f::new(wo[0] / length, wo[1] / length, wo[2] / length);
                        let wi_local = Vector3f::new(-wo[1] / length, wo[0] / length, 0.4);

                        let bxdf = HairBxDF::new(
                            h as Float,
                            ETA,
                            SampledSpectrum::from(SIGMA_A),
                            beta_m,
                            beta_n,
                            alpha,
                        );
                        let f = bxdf.f(&wo_local, &wi_local, TransportMode::Radiance);
                        let pdf = bxdf.pdf(
                            &wo_local,
                            &wi_local,
                            TransportMode::Radiance,
                            BXDF_REFL_TRANS_ALL,
                        );
                        let sample = bxdf
                            .sample_f(
                                &wo_local,
                                0.5,
                                &Point2f::new(u[0], u[1]),
                                TransportMode::Radiance,
                                BXDF_REFL_TRANS_ALL,
                            )
                            .expect("cpu hair sampling should succeed");

                        cases.push(HairProbeInput {
                            params: [beta_m as f32, beta_n as f32, alpha as f32, ETA as f32],
                            sigma_a: SIGMA_A_F32,
                            h_and_u: [h as f32, u[0], u[1], 0.0],
                            wo: [wo_local.x as f32, wo_local.y as f32, wo_local.z as f32, 0.0],
                            wi: [wi_local.x as f32, wi_local.y as f32, wi_local.z as f32, 0.0],
                        });
                        expected.push((f, pdf, (sample.wi, sample.pdf)));
                    }
                }
            }
        }
    }

    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hair probe inputs"),
        contents: bytemuck::cast_slice(&cases),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hair probe outputs"),
        size: (cases.len() * 3 * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hair probe readback"),
        size: output_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hair bxdf probe"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(probe_source())),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hair bxdf probe"),
        layout: None,
        module: &module,
        entry_point: Some("hair_bxdf_probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hair bxdf probe"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    let groups = cases.len().div_ceil(WORKGROUP_SIZE) as u32;
    pass.dispatch_workgroups(groups, 1, 1);
    drop(pass);
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, readback.size());
    context.queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device should stay responsive");
    receiver
        .recv()
        .expect("readback mapping should succeed")
        .expect("mapping should not error");
    let mapped = slice.get_mapped_range().unwrap();
    let floats = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    // wgpu 30 returns an async error scope; poll it to completion like the
    // production pipelines do.
    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("shader must not raise a validation error: {error}");
    }

    let tolerance = 2e-3;
    assert_eq!(floats.len(), expected.len() * 12);
    for (index, (want_f, want_pdf, (want_wi, want_sample_pdf))) in expected.iter().enumerate() {
        let base = index * 12;
        let f_lane = |lane: usize| floats[base + lane];
        for lane in 0..4 {
            let want = want_f[lane] as f32;
            let got = f_lane(lane);
            assert!(
                (got - want).abs() <= tolerance * (1.0 + want.abs()),
                "f mismatch at case {index} lane {lane}: {got} vs {want}"
            );
        }
        let got_wi = [floats[base + 4], floats[base + 5], floats[base + 6]];
        let got_pdf = floats[base + 7];
        let got_sample_pdf = floats[base + 8];
        let got_sample_valid = floats[base + 9];
        assert!(
            (got_pdf - *want_pdf as f32).abs() <= tolerance * (1.0 + *want_pdf as f32),
            "pdf mismatch at case {index}: {got_pdf} vs {want_pdf}"
        );
        assert_eq!(got_sample_valid, 1.0, "sample should succeed at {index}");
        for (axis, want) in [want_wi.x as f32, want_wi.y as f32, want_wi.z as f32]
            .into_iter()
            .enumerate()
        {
            assert!(
                (got_wi[axis] - want).abs() <= tolerance,
                "sampled wi mismatch at case {index} axis {axis}: {} vs {want}",
                got_wi[axis]
            );
        }
        assert!(
            (got_sample_pdf - *want_sample_pdf as f32).abs()
                <= tolerance * (1.0 + *want_sample_pdf as f32),
            "sampled pdf mismatch at case {index}: {got_sample_pdf} vs {want_sample_pdf}"
        );
    }
}
