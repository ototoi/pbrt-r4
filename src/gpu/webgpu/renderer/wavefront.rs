mod bxdf;
mod camera;
mod film;
mod intersection;
mod lighting;

use super::super::shader::ShaderStageId;
use super::resources::{RenderBuffers, RenderDimensions};
use super::{ExecutableScene, Renderer};
use crate::gpu::ir::{GpuRenderOutput, GpuRenderRequest};
use crate::gpu::webgpu::error::BackendError;
use std::time::{Duration, Instant};

const READBACK_SPP_INTERVAL: u32 = 8;
const READBACK_TIME_INTERVAL: Duration = Duration::from_secs(2);

pub fn render(
    renderer: &Renderer,
    scene: &ExecutableScene,
    request: GpuRenderRequest,
    dimensions: &RenderDimensions,
    buffers: &RenderBuffers,
    bind_group: &wgpu::BindGroup,
) -> Result<GpuRenderOutput, BackendError> {
    let mut sample_offset = 0u32;
    let mut samples_since_readback = 0u32;
    let mut last_readback = Instant::now();
    let mut latest_rgb = None;

    while sample_offset < request.sample_count {
        let mut encoder = renderer.device_context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("pbrt-r4 WebGPU wavefront dispatch"),
            },
        );
        if sample_offset == 0 {
            encoder.clear_buffer(&buffers.output, 0, None);
            encoder.clear_buffer(&buffers.wavefront, 0, None);
        }
        let mut batch_count = 0u32;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pbrt-r4 WebGPU wavefront pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bind_group, &[]);
            while sample_offset < request.sample_count {
                dispatch_sample(
                    renderer,
                    &mut pass,
                    dimensions.wavefront_workgroups,
                    scene.scene().render.integrator.max_depth,
                    sample_offset + 1 < request.sample_count,
                );
                sample_offset += 1;
                batch_count += 1;
                samples_since_readback += 1;
                if sample_offset == request.sample_count
                    || samples_since_readback >= READBACK_SPP_INTERVAL
                    || last_readback.elapsed() >= READBACK_TIME_INTERVAL
                {
                    break;
                }
            }
        }
        encoder.copy_buffer_to_buffer(
            &buffers.output,
            0,
            &buffers.readback,
            0,
            dimensions.output_size,
        );
        encoder.copy_buffer_to_buffer(
            &buffers.wavefront,
            0,
            &buffers.control_readback,
            0,
            u64::from(crate::gpu::webgpu::wavefront::WAVEFRONT_CONTROL_SIZE),
        );
        renderer.device_context.queue.submit(Some(encoder.finish()));
        latest_rgb = Some(film::readback(
            renderer,
            &buffers.readback,
            &buffers.control_readback,
            dimensions.pixel_count,
        )?);
        samples_since_readback = 0;
        last_readback = Instant::now();
        debug_assert!(batch_count > 0);
    }

    let rgb = latest_rgb
        .ok_or_else(|| BackendError::Readback("wavefront produced no film readback".to_string()))?;
    GpuRenderOutput::new(dimensions.pixel_bounds, rgb.into_boxed_slice(), request)
        .map_err(|error| BackendError::Readback(format!("invalid render output: {error:?}")))
}

fn dispatch_sample<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    workgroups: u32,
    max_depth: u32,
    advance_sample: bool,
) {
    camera::dispatch(renderer, pass, workgroups);
    for _ in 0..=max_depth {
        intersection::dispatch(renderer, pass, workgroups);
        bxdf::dispatch(renderer, pass, workgroups);
        lighting::dispatch(renderer, pass, workgroups);
        film::finish_bounce(renderer, pass, workgroups);
        film::prepare_next_bounce(renderer, pass);
    }
    film::update(renderer, pass, workgroups);
    if advance_sample {
        film::advance_sample(renderer, pass);
    }
}

fn dispatch_stage<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    stage: ShaderStageId,
    workgroups: u32,
) {
    pass.set_pipeline(renderer.pipeline.stage(stage));
    pass.dispatch_workgroups(workgroups, 1, 1);
}

fn dispatch_control_stage<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    stage: ShaderStageId,
) {
    pass.set_pipeline(renderer.pipeline.stage(stage));
    pass.dispatch_workgroups(1, 1, 1);
}
