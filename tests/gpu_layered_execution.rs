#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::webgpu::{
    abi::LayeredBxDFData, context::Context, pipeline::Pipeline, stages::RequiredLimits,
};
use wgpu::util::DeviceExt;

const SAMPLE_COUNT: usize = 8192;
const CASE_COUNT: usize = 5;

#[test]
#[ignore = "requires a Vulkan GPU with experimental ray queries"]
fn mixed_layered_scene_renders_with_real_and_debug_materials() {
    use pbrt_r4::util::imageio::read_image;
    use std::{
        process::Command,
        time::{SystemTime, UNIX_EPOCH},
    };
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut images = Vec::new();
    for mode in [None, Some("normal"), Some("uv"), Some("lambert")] {
        let output = std::env::temp_dir().join(format!(
            "pbrt-layered-{}-{nonce}-{}.exr",
            std::process::id(),
            mode.unwrap_or("real")
        ));
        let mut command = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"));
        command
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .env_remove("PBRT_R4_GPU_DEBUG_MATERIAL")
            .args(["--use-gpu", "--spp", "1", "--outfile"])
            .arg(&output)
            .arg("tests/scenes/gpu-wavefront-layered.pbrt");
        if let Some(mode) = mode {
            command.env("PBRT_R4_GPU_DEBUG_MATERIAL", mode);
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success() && output.exists(),
            "render {mode:?}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
        assert_eq!((resolution.x, resolution.y), (128, 128));
        let rgb: Vec<_> = pixels.iter().flat_map(|p| p.to_rgb()).collect();
        assert!(rgb.iter().all(|v| v.is_finite()));
        assert!(rgb.iter().any(|v| *v > 0.0));
        images.push(rgb);
        std::fs::remove_file(output).unwrap();
    }
    for i in 0..images.len() {
        for j in 0..i {
            assert_ne!(images[i], images[j], "material override modes must differ");
        }
    }
}

// Hardware test: exercises the production WGSL, not a translated CPU copy.
#[test]
#[ignore = "requires a Vulkan GPU with experimental ray queries"]
fn layered_gpu_matches_analytic_limits_and_sampled_energy() {
    let context = Context::new(RequiredLimits::default()).unwrap();
    let device = &context.device;
    Pipeline::new(device).expect("all composed wavefront shaders must validate");
    let cases: Vec<_> = (0..CASE_COUNT)
        .map(|case| LayeredBxDFData {
            thickness: 0.2,
            g: if case == 3 { -0.4 } else { 0.4 },
            max_depth: if case == 0 { 2 } else { 32 },
            n_samples: if case == 4 { 4 } else { 1 },
            albedo: if case >= 2 {
                [0.6, 0.6, 0.6, 0.0]
            } else {
                [0.0; 4]
            },
            two_sided: 1,
            padding: [0; 3],
        })
        .collect();
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("layered test data (48-byte stride)"),
        contents: bytemuck::cast_slice(&cases),
        usage: wgpu::BufferUsages::STORAGE,
    });
    const WORDS: usize = 16;
    let byte_size = (SAMPLE_COUNT * CASE_COUNT * WORDS * 4) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layered test results"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layered readback"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let source = format!(
        "{}\n{}",
        include_str!("../src/gpu/webgpu/shaders/layered.wgsl"),
        include_str!("shaders/layered_probe.wgsl")
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("layered numeric probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("layered numeric probe"),
        layout: None,
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups((SAMPLE_COUNT / 64) as u32, CASE_COUNT as u32, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, byte_size);
    context.queue.submit(Some(encoder.finish()));
    let (send, recv) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).unwrap()
        });
    context.wait().unwrap();
    recv.recv().unwrap().unwrap();
    let mapped = readback.slice(..).get_mapped_range().unwrap();
    let values: &[f32] = bytemuck::cast_slice(&mapped);
    for case in 0..CASE_COUNT {
        let mut sampled = 0.0_f64;
        let mut integrated = 0.0_f64;
        for i in 0..SAMPLE_COUNT {
            let v = &values[(case * SAMPLE_COUNT + i) * WORDS..][..WORDS];
            assert!(
                v.iter().all(|v| v.is_finite()),
                "nonfinite case={case} sample={i}: {v:?}"
            );
            sampled += v[0] as f64;
            integrated += v[1] as f64;
            assert!((v[2] - v[3]).abs() < 1e-6, "two-sided f mismatch");
            assert!(v[4] < 1e-5, "two-sided sampling mismatch");
            assert!(v[5] < 1e-5, "sample direction is not unit length");
            assert_eq!(v[6], 0.0, "opaque bottom cannot transmit");
            assert!(
                (v[7] - v[8]).abs() < 2e-6,
                "TRT PDF mismatch case={case} sample={i}: {v:?}"
            );
            if case == 0 {
                assert!(
                    (v[0] - v[9]).abs() < 2e-5,
                    "eta=1 sample attenuation mismatch"
                );
                assert!((v[2] - v[10]).abs() < 2e-5, "eta=1 f attenuation mismatch");
            } else {
                assert!((v[11] - 1.0).abs() < 1e-5, "entrance specular weight");
                assert!((v[12] - 0.04).abs() < 1e-5, "entrance Fresnel PDF");
                assert_eq!(v[13] as u32, 17, "entrance specular flags");
            }
        }
        sampled /= SAMPLE_COUNT as f64;
        integrated /= SAMPLE_COUNT as f64;
        // f excludes the delta reflection at the top. Include its analytic mass.
        if case != 0 {
            integrated += 0.04;
        }
        eprintln!("case {case}: sampled={sampled:.6}, integrated={integrated:.6}");
        assert!(
            (sampled - integrated).abs() < 0.035,
            "energy mismatch case {case}"
        );
        assert!((0.0..=1.0).contains(&sampled));
    }
}
