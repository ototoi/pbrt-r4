use std::f32::consts::PI;

use pbrt_r4::gpu::webgpu::abi::{PortalDistributionTexel, PortalImageInfiniteRecord};
use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::shader::compose_source;
use pbrt_r4::gpu::webgpu::stages::RequiredLimits;
use wgpu::util::DeviceExt;

const RESULT_FLOATS: usize = 68;

const PORTAL_TEST_SHADER: &str = r#"
@group(0) @binding(26) var<storage, read_write> output_values: array<vec4<f32>, 17>;

@compute @workgroup_size(1)
fn portal_test() {
    let portal = portal_infinite_lights[0];
    let mapped = portal_image_from_render(portal, vec3<f32>(0.0, 0.0, 1.0));
    let back = portal_image_from_render(portal, vec3<f32>(0.0, 0.0, -1.0));
    let corner00 = portal_image_from_render(portal, normalize(vec3<f32>(-1.0, -1.0, 1.0)));
    let corner01 = portal_image_from_render(portal, normalize(vec3<f32>(-1.0, 1.0, 1.0)));
    let corner11 = portal_image_from_render(portal, normalize(vec3<f32>(1.0, 1.0, 1.0)));
    let corner10 = portal_image_from_render(portal, normalize(vec3<f32>(1.0, -1.0, 1.0)));
    let direction = portal_render_from_image(portal, vec2<f32>(0.5));
    let bounds = portal_image_bounds(portal, vec3<f32>(0.0));
    let sampled = sample_portal_distribution(portal, vec2<f32>(0.5), bounds);
    let pdf = portal_distribution_pdf(portal, vec2<f32>(0.5), bounds);
    let one = portal_infinite_lights[1];
    let one_bounds = portal_image_bounds(one, vec3<f32>(0.0));
    let one_sample = sample_portal_distribution(one, vec2<f32>(0.5), one_bounds);
    let zero = portal_infinite_lights[2];
    let zero_bounds = portal_image_bounds(zero, vec3<f32>(0.0));
    let zero_sample = sample_portal_distribution(zero, vec2<f32>(0.5), zero_bounds);
    let non_finite = portal_infinite_lights[3];
    let non_finite_bounds = portal_image_bounds(non_finite, vec3<f32>(0.0));
    let non_finite_sample = sample_portal_distribution(
        non_finite, vec2<f32>(0.5), non_finite_bounds,
    );
    output_values[0] = vec4<f32>(mapped.uv, mapped.duv_dw, f32(mapped.valid));
    output_values[1] = vec4<f32>(direction.wi, direction.duv_dw);
    output_values[2] = vec4<f32>(bounds.min, bounds.max);
    output_values[3] = vec4<f32>(sampled.uv, sampled.pdf, pdf);
    output_values[4] = vec4<f32>(back.uv, back.duv_dw, f32(back.valid));
    output_values[5] = vec4<f32>(corner00.uv, corner00.duv_dw, f32(corner00.valid));
    output_values[6] = vec4<f32>(corner01.uv, corner01.duv_dw, f32(corner01.valid));
    output_values[7] = vec4<f32>(corner11.uv, corner11.duv_dw, f32(corner11.valid));
    output_values[8] = vec4<f32>(corner10.uv, corner10.duv_dw, f32(corner10.valid));
    output_values[9] = vec4<f32>(one_sample.uv, one_sample.pdf, f32(one_sample.valid));
    output_values[10] = vec4<f32>(zero_sample.uv, zero_sample.pdf, f32(zero_sample.valid));
    output_values[11] = vec4<f32>(
        portal_distribution_pdf(portal, vec2<f32>(0.1), bounds),
        non_finite_sample.pdf,
        f32(non_finite_sample.valid),
        0.0,
    );
    let rounded_cdf = portal_infinite_lights[4];
    let rounded_bounds = PortalBoundsResult(vec2<f32>(0.5, 0.0), vec2<f32>(1.0), 1u);
    let rounded_sample = portal_sample_marginal_x(rounded_cdf, 0.5, rounded_bounds, 1.0);
    output_values[12] = vec4<f32>(rounded_sample.value, f32(rounded_sample.valid), 0.0, 0.0);
    let invalid_bounds = portal_image_bounds(portal, vec3<f32>(0.0, 0.0, 2.0));
    let invalid_bounds_sample = sample_portal_distribution(portal, vec2<f32>(0.5), invalid_bounds);
    output_values[13] = vec4<f32>(
        f32(invalid_bounds.valid), invalid_bounds_sample.pdf,
        f32(invalid_bounds_sample.valid), 0.0,
    );
    let nan = bitcast<f32>(0x7fc00000u);
    let invalid_u_sample = sample_portal_distribution(portal, vec2<f32>(nan, 0.5), bounds);
    output_values[14] = vec4<f32>(
        invalid_u_sample.uv, invalid_u_sample.pdf, f32(invalid_u_sample.valid),
    );
    let invalid_direction = portal_render_from_image(portal, vec2<f32>(nan, 0.5));
    output_values[15] = vec4<f32>(
        invalid_direction.wi, f32(invalid_direction.valid),
    );
    let outside_direction = portal_render_from_image(portal, vec2<f32>(0.1));
    let outside_mapping = portal_image_from_render(portal, outside_direction.wi);
    output_values[16] = vec4<f32>(
        portal_pdf_li(portal, outside_mapping, vec3<f32>(0.0)),
        outside_mapping.uv,
        f32(outside_mapping.valid),
    );
}
"#;

#[test]
#[ignore = "requires a WebGPU adapter"]
fn portal_shader_matches_v4_mapping_and_windowed_distribution() {
    let context = Context::new(
        RequiredLimits {
            storage_buffers_per_shader_stage: 3,
            uniform_buffers_per_shader_stage: 0,
            buffers_and_acceleration_structures_per_shader_stage: 3,
            bind_groups: 1,
        },
        0,
        0,
    )
    .unwrap();
    let device = &context.device;
    let portal = PortalImageInfiniteRecord {
        portal: [
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, 1.0],
        ],
        world_to_portal: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        distribution_offset: 0,
        width: 3,
        height: 3,
        reserved: 0,
    };
    let mut portals = [portal; 5];
    portals[1].distribution_offset = 9;
    portals[1].width = 1;
    portals[1].height = 1;
    portals[2].distribution_offset = 10;
    portals[2].width = 1;
    portals[2].height = 1;
    portals[3].distribution_offset = 11;
    portals[3].width = 1;
    portals[3].height = 1;
    portals[4].distribution_offset = 12;
    portals[4].width = 2;
    portals[4].height = 1;
    let distribution = [
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 1.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 2.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 3.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 2.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 4.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 6.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 3.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 6.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 9.0,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 1.0,
        },
        PortalDistributionTexel {
            function: 0.0,
            summed_area: 0.0,
        },
        PortalDistributionTexel {
            function: f32::NAN,
            summed_area: f32::NAN,
        },
        PortalDistributionTexel {
            function: 1.0e20,
            summed_area: 1.0e20,
        },
        PortalDistributionTexel {
            function: 1.0,
            summed_area: 1.0e20,
        },
    ];
    let portal_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("portal test record"),
        contents: bytemuck::cast_slice(&portals),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let distribution_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("portal test distribution"),
        contents: bytemuck::cast_slice(&distribution),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("portal test output"),
        size: (RESULT_FLOATS * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("portal test readback"),
        size: (RESULT_FLOATS * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("portal test layout"),
        entries: &[
            buffer_layout(24, true),
            buffer_layout(25, true),
            buffer_layout(26, false),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("portal test bind group"),
        layout: &layout,
        entries: &[
            binding(24, portal_buffer.as_entire_binding()),
            binding(25, distribution_buffer.as_entire_binding()),
            binding(26, output.as_entire_binding()),
        ],
    });
    let source = compose_source(PORTAL_TEST_SHADER);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("portal numerical test"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("portal numerical test"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("portal numerical test"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("portal_test"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, (RESULT_FLOATS * 4) as u64);
    context.queue.submit(Some(encoder.finish()));
    let actual = read_f32_buffer(device, &readback);

    let mapped = reference_image_from_render(&portal, [0.0, 0.0, 1.0]);
    let direction = reference_render_from_image(&portal, [0.5, 0.5]);
    let bounds = reference_image_bounds(&portal, [0.0, 0.0, 0.0]);
    let sampled = reference_sample(&portal, &distribution, [0.5, 0.5], bounds);
    let pdf = reference_pdf(&portal, &distribution, [0.5, 0.5], bounds);
    let back = reference_image_from_render(&portal, [0.0, 0.0, -1.0]);

    let mut expected = Vec::with_capacity(RESULT_FLOATS);
    push_uv_result(&mut expected, mapped);
    expected.extend_from_slice(&[
        direction.wi[0],
        direction.wi[1],
        direction.wi[2],
        direction.duv_dw,
    ]);
    expected.extend_from_slice(&[bounds.min[0], bounds.min[1], bounds.max[0], bounds.max[1]]);
    expected.extend_from_slice(&[sampled.uv[0], sampled.uv[1], sampled.pdf, pdf]);
    push_uv_result(&mut expected, back);
    for corner in [
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ] {
        push_uv_result(
            &mut expected,
            reference_image_from_render(&portal, normalize(corner)),
        );
    }
    let one_bounds = reference_image_bounds(&portals[1], [0.0, 0.0, 0.0]);
    let one_sample = reference_sample(&portals[1], &distribution, [0.5, 0.5], one_bounds);
    push_sample_result(&mut expected, one_sample);
    let zero_bounds = reference_image_bounds(&portals[2], [0.0, 0.0, 0.0]);
    let zero_sample = reference_sample(&portals[2], &distribution, [0.5, 0.5], zero_bounds);
    push_sample_result(&mut expected, zero_sample);
    let non_finite_bounds = reference_image_bounds(&portals[3], [0.0, 0.0, 0.0]);
    let non_finite_sample =
        reference_sample(&portals[3], &distribution, [0.5, 0.5], non_finite_bounds);
    expected.extend_from_slice(&[
        reference_pdf(&portal, &distribution, [0.1, 0.1], bounds),
        non_finite_sample.pdf,
        non_finite_sample.valid as u32 as f32,
        0.0,
    ]);
    let rounded_bounds = ReferenceBounds {
        min: [0.5, 0.0],
        max: [1.0, 1.0],
        valid: true,
    };
    let rounded_sample =
        reference_sample_marginal_x(&portals[4], &distribution, 0.5, rounded_bounds, 1.0);
    expected.extend_from_slice(&[
        rounded_sample.value,
        rounded_sample.valid as u32 as f32,
        0.0,
        0.0,
    ]);
    let invalid_bounds = reference_image_bounds(&portal, [0.0, 0.0, 2.0]);
    let invalid_bounds_sample =
        reference_sample(&portal, &distribution, [0.5, 0.5], invalid_bounds);
    expected.extend_from_slice(&[
        invalid_bounds.valid as u32 as f32,
        invalid_bounds_sample.pdf,
        invalid_bounds_sample.valid as u32 as f32,
        0.0,
    ]);
    let invalid_u_sample = reference_sample(&portal, &distribution, [f32::NAN, 0.5], bounds);
    push_sample_result(&mut expected, invalid_u_sample);
    expected.extend_from_slice(&[0.0; 4]);
    let outside_direction = reference_render_from_image(&portal, [0.1, 0.1]);
    let outside_mapping = reference_image_from_render(&portal, outside_direction.wi);
    expected.extend_from_slice(&[
        reference_pdf(&portal, &distribution, outside_mapping.uv, bounds) / outside_mapping.duv_dw,
        outside_mapping.uv[0],
        outside_mapping.uv[1],
        outside_mapping.valid as u32 as f32,
    ]);
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        if matches!(index, 2 | 7 | 14 | 15 | 22 | 26 | 30 | 34 | 38 | 44 | 64) && expected != 0.0 {
            assert_relative(actual, expected, 5e-5, index);
        } else {
            assert_absolute(actual, expected, 2e-5, index);
        }
    }
}

#[derive(Clone, Copy)]
struct ReferenceUv {
    uv: [f32; 2],
    duv_dw: f32,
    valid: bool,
}

#[derive(Clone, Copy)]
struct ReferenceDirection {
    wi: [f32; 3],
    duv_dw: f32,
}

#[derive(Clone, Copy)]
struct ReferenceBounds {
    min: [f32; 2],
    max: [f32; 2],
    valid: bool,
}

#[derive(Clone, Copy)]
struct ReferenceSample {
    uv: [f32; 2],
    pdf: f32,
    valid: bool,
}

#[derive(Clone, Copy)]
struct ReferenceAxisSample {
    value: f32,
    valid: bool,
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let inverse_length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().recip();
    v.map(|value| value * inverse_length)
}

fn reference_image_from_render(
    portal: &PortalImageInfiniteRecord,
    w_render: [f32; 3],
) -> ReferenceUv {
    let w = normalize(std::array::from_fn(|row| {
        portal.world_to_portal[row][0] * w_render[0]
            + portal.world_to_portal[row][1] * w_render[1]
            + portal.world_to_portal[row][2] * w_render[2]
    }));
    if w.iter().any(|value| !value.is_finite()) || w[2] <= 0.0 {
        return ReferenceUv {
            uv: [0.0; 2],
            duv_dw: 0.0,
            valid: false,
        };
    }
    let duv_dw = PI * PI * (1.0 - w[0] * w[0]) * (1.0 - w[1] * w[1]) / w[2];
    let uv = [
        (w[0].atan2(w[2]) / PI + 0.5).clamp(0.0, 1.0),
        (w[1].atan2(w[2]) / PI + 0.5).clamp(0.0, 1.0),
    ];
    ReferenceUv {
        uv,
        duv_dw,
        valid: duv_dw.is_finite() && uv.iter().all(|value| value.is_finite()),
    }
}

fn reference_render_from_image(
    portal: &PortalImageInfiniteRecord,
    uv: [f32; 2],
) -> ReferenceDirection {
    let angles = [-0.5 * PI + uv[0] * PI, -0.5 * PI + uv[1] * PI];
    let w = normalize([angles[0].tan(), angles[1].tan(), 1.0]);
    let duv_dw = PI * PI * (1.0 - w[0] * w[0]) * (1.0 - w[1] * w[1]) / w[2];
    let wi = normalize(std::array::from_fn(|column| {
        portal.world_to_portal[0][column] * w[0]
            + portal.world_to_portal[1][column] * w[1]
            + portal.world_to_portal[2][column] * w[2]
    }));
    ReferenceDirection { wi, duv_dw }
}

fn reference_image_bounds(portal: &PortalImageInfiniteRecord, point: [f32; 3]) -> ReferenceBounds {
    let directions = [portal.portal[0], portal.portal[2]].map(|corner| {
        normalize([
            corner[0] - point[0],
            corner[1] - point[1],
            corner[2] - point[2],
        ])
    });
    let mapped = directions.map(|direction| reference_image_from_render(portal, direction));
    if !mapped[0].valid || !mapped[1].valid {
        return ReferenceBounds {
            min: [0.0; 2],
            max: [0.0; 2],
            valid: false,
        };
    }
    ReferenceBounds {
        min: [
            mapped[0].uv[0].min(mapped[1].uv[0]),
            mapped[0].uv[1].min(mapped[1].uv[1]),
        ],
        max: [
            mapped[0].uv[0].max(mapped[1].uv[0]),
            mapped[0].uv[1].max(mapped[1].uv[1]),
        ],
        valid: true,
    }
}

fn reference_sat_lookup_int(
    portal: &PortalImageInfiniteRecord,
    distribution: &[PortalDistributionTexel],
    x: i32,
    y: i32,
) -> f32 {
    if x == 0 || y == 0 {
        return 0.0;
    }
    let x = (x - 1).min(portal.width as i32 - 1) as u32;
    let y = (y - 1).min(portal.height as i32 - 1) as u32;
    distribution[(portal.distribution_offset + y * portal.width + x) as usize].summed_area
}

fn reference_sat_lookup(
    portal: &PortalImageInfiniteRecord,
    distribution: &[PortalDistributionTexel],
    point: [f32; 2],
) -> f32 {
    let scaled = [
        point[0] * portal.width as f32,
        point[1] * portal.height as f32,
    ];
    let p0 = [scaled[0] as i32, scaled[1] as i32];
    let d = [scaled[0] - p0[0] as f32, scaled[1] - p0[1] as f32];
    let value = |x, y| reference_sat_lookup_int(portal, distribution, x, y);
    (1.0 - d[0]) * (1.0 - d[1]) * value(p0[0], p0[1])
        + d[0] * (1.0 - d[1]) * value(p0[0] + 1, p0[1])
        + (1.0 - d[0]) * d[1] * value(p0[0], p0[1] + 1)
        + d[0] * d[1] * value(p0[0] + 1, p0[1] + 1)
}

fn reference_sat_integral(
    portal: &PortalImageInfiniteRecord,
    distribution: &[PortalDistributionTexel],
    lower: [f32; 2],
    upper: [f32; 2],
) -> f32 {
    let sum = reference_sat_lookup(portal, distribution, upper)
        - reference_sat_lookup(portal, distribution, [lower[0], upper[1]])
        + reference_sat_lookup(portal, distribution, lower)
        - reference_sat_lookup(portal, distribution, [upper[0], lower[1]]);
    (sum / (portal.width * portal.height) as f32).max(0.0)
}

fn reference_sample_marginal_x(
    portal: &PortalImageInfiniteRecord,
    distribution: &[PortalDistributionTexel],
    u: f32,
    bounds: ReferenceBounds,
    bounds_integral: f32,
) -> ReferenceAxisSample {
    let mut x_min = bounds.min[0];
    let mut x_max = bounds.max[0];
    for _ in 0..32 {
        if (portal.width as f32 * x_max).ceil() - (portal.width as f32 * x_min).floor() <= 1.0 {
            break;
        }
        let mid = 0.5 * (x_min + x_max);
        let cdf = reference_sat_integral(portal, distribution, bounds.min, [mid, bounds.max[1]])
            / bounds_integral;
        if cdf > u {
            x_max = mid;
        } else {
            x_min = mid;
        }
    }
    let cdf = |x| {
        reference_sat_integral(portal, distribution, bounds.min, [x, bounds.max[1]])
            / bounds_integral
    };
    let px_min = cdf(x_min);
    let px_max = cdf(x_max);
    let delta = px_max - px_min;
    if !px_min.is_finite() || !px_max.is_finite() || !delta.is_finite() || delta == 0.0 {
        return ReferenceAxisSample {
            value: 0.0,
            valid: false,
        };
    }
    ReferenceAxisSample {
        value: (x_min + (x_max - x_min) * ((u - px_min) / delta)).clamp(x_min, x_max),
        valid: true,
    }
}

fn reference_sample_conditional_y(
    portal: &PortalImageInfiniteRecord,
    distribution: &[PortalDistributionTexel],
    u: f32,
    bounds: ReferenceBounds,
    x: f32,
) -> ReferenceAxisSample {
    let width = portal.width as f32;
    let cond_min = [(x * width).floor() / width, bounds.min[1]];
    let mut cond_max = [(x * width).ceil() / width, bounds.max[1]];
    if cond_min[0] == cond_max[0] {
        cond_max[0] += 1.0 / width;
    }
    let integral = reference_sat_integral(portal, distribution, cond_min, cond_max);
    if !integral.is_finite() || integral == 0.0 {
        return ReferenceAxisSample {
            value: 0.0,
            valid: false,
        };
    }
    let mut y_min = bounds.min[1];
    let mut y_max = bounds.max[1];
    for _ in 0..32 {
        if (portal.height as f32 * y_max).ceil() - (portal.height as f32 * y_min).floor() <= 1.0 {
            break;
        }
        let mid = 0.5 * (y_min + y_max);
        let cdf =
            reference_sat_integral(portal, distribution, cond_min, [cond_max[0], mid]) / integral;
        if cdf > u {
            y_max = mid;
        } else {
            y_min = mid;
        }
    }
    let cdf =
        |y| reference_sat_integral(portal, distribution, cond_min, [cond_max[0], y]) / integral;
    let py_min = cdf(y_min);
    let py_max = cdf(y_max);
    let delta = py_max - py_min;
    if !py_min.is_finite() || !py_max.is_finite() || !delta.is_finite() || delta == 0.0 {
        return ReferenceAxisSample {
            value: 0.0,
            valid: false,
        };
    }
    ReferenceAxisSample {
        value: (y_min + (y_max - y_min) * ((u - py_min) / delta)).clamp(y_min, y_max),
        valid: true,
    }
}

fn reference_sample(
    portal: &PortalImageInfiniteRecord,
    distribution: &[PortalDistributionTexel],
    u: [f32; 2],
    bounds: ReferenceBounds,
) -> ReferenceSample {
    let invalid = ReferenceSample {
        uv: [0.0; 2],
        pdf: 0.0,
        valid: false,
    };
    if !bounds.valid
        || portal.width == 0
        || portal.height == 0
        || u.iter().any(|value| !value.is_finite())
    {
        return invalid;
    }
    let integral = reference_sat_integral(portal, distribution, bounds.min, bounds.max);
    if !integral.is_finite() || integral == 0.0 {
        return invalid;
    }
    let x = reference_sample_marginal_x(portal, distribution, u[0], bounds, integral);
    if !x.valid {
        return invalid;
    }
    let y = reference_sample_conditional_y(portal, distribution, u[1], bounds, x.value);
    if !y.valid {
        return invalid;
    }
    let uv = [x.value, y.value];
    let pdf = reference_distribution_eval(portal, distribution, uv) / integral;
    if !pdf.is_finite() {
        return invalid;
    }
    ReferenceSample {
        uv,
        pdf,
        valid: true,
    }
}

fn reference_distribution_eval(
    portal: &PortalImageInfiniteRecord,
    distribution: &[PortalDistributionTexel],
    uv: [f32; 2],
) -> f32 {
    let x = (uv[0] * portal.width as f32) as u32;
    let y = (uv[1] * portal.height as f32) as u32;
    distribution[(portal.distribution_offset
        + y.min(portal.height - 1) * portal.width
        + x.min(portal.width - 1)) as usize]
        .function
}

fn reference_pdf(
    portal: &PortalImageInfiniteRecord,
    distribution: &[PortalDistributionTexel],
    uv: [f32; 2],
    bounds: ReferenceBounds,
) -> f32 {
    if !bounds.valid || portal.width == 0 || portal.height == 0 {
        return 0.0;
    }
    let integral = reference_sat_integral(portal, distribution, bounds.min, bounds.max);
    if !integral.is_finite() || integral == 0.0 {
        return 0.0;
    }
    let pdf = reference_distribution_eval(portal, distribution, uv) / integral;
    if pdf.is_finite() {
        pdf
    } else {
        0.0
    }
}

fn push_uv_result(values: &mut Vec<f32>, result: ReferenceUv) {
    values.extend_from_slice(&[
        result.uv[0],
        result.uv[1],
        result.duv_dw,
        result.valid as u32 as f32,
    ]);
}

fn push_sample_result(values: &mut Vec<f32>, result: ReferenceSample) {
    values.extend_from_slice(&[
        result.uv[0],
        result.uv[1],
        result.pdf,
        result.valid as u32 as f32,
    ]);
}

fn assert_absolute(actual: f32, expected: f32, tolerance: f32, index: usize) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "value {index}: actual {actual}, expected {expected}"
    );
}

fn assert_relative(actual: f32, expected: f32, tolerance: f32, index: usize) {
    let scale = expected.abs().max(f32::MIN_POSITIVE);
    assert!(
        (actual - expected).abs() / scale <= tolerance,
        "value {index}: actual {actual}, expected {expected}"
    );
}

fn buffer_layout(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn binding(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

fn read_f32_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    values
}
