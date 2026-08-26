use super::super::super::ir::{GpuRenderOutput, GpuRenderRequest};
use super::super::error::BackendError;
use super::super::shader::ShaderStageId;
use super::resources::{RenderBuffers, RenderDimensions};
use super::Renderer;

pub fn render(
    renderer: &Renderer,
    request: GpuRenderRequest,
    dimensions: &RenderDimensions,
    buffers: &RenderBuffers,
    bind_group: &wgpu::BindGroup,
) -> Result<GpuRenderOutput, BackendError> {
    let sample_end = request
        .sample_start
        .checked_add(u64::from(request.sample_count))
        .ok_or(BackendError::UnsupportedRenderRequest {
            reason: "sample range overflows the WebGPU sampler index",
        })?;
    if sample_end > u64::from(u32::MAX) + 1 {
        return Err(BackendError::UnsupportedRenderRequest {
            reason: "the WebGPU wavefront renderer supports 32-bit sampler indices only",
        });
    }

    let mut clear_encoder =
        renderer
            .device_context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pbrt-r4 WebGPU wavefront film clear"),
            });
    clear_encoder.clear_buffer(&buffers.output, 0, None);
    renderer
        .device_context
        .queue
        .submit(Some(clear_encoder.finish()));

    for sample_offset in 0..request.sample_count {
        let sample_index = request.sample_start as u32 + sample_offset;
        renderer.device_context.queue.write_buffer(
            &buffers.arena,
            0,
            &arena_header_bytes(sample_index, dimensions.arena_layout.capacity),
        );
        let mut encoder = renderer.device_context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("pbrt-r4 WebGPU wavefront dispatch"),
            },
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pbrt-r4 WebGPU wavefront pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_pipeline(renderer.pipeline.stage(ShaderStageId::PrepareCameraRays));
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(renderer.pipeline.stage(ShaderStageId::GenerateCameraRays));
            pass.dispatch_workgroups(dimensions.workgroups, 1, 1);
            pass.set_pipeline(renderer.pipeline.stage(ShaderStageId::IntersectClosest));
            pass.dispatch_workgroups(dimensions.workgroups, 1, 1);
            pass.set_pipeline(renderer.pipeline.stage(ShaderStageId::HandleEscapedRays));
            pass.dispatch_workgroups(dimensions.workgroups, 1, 1);
            pass.set_pipeline(
                renderer
                    .pipeline
                    .stage(ShaderStageId::EvaluateSurfaceInteraction),
            );
            pass.dispatch_workgroups(dimensions.workgroups, 1, 1);
            pass.set_pipeline(renderer.pipeline.stage(ShaderStageId::EvaluateMaterial));
            pass.dispatch_workgroups(dimensions.workgroups, 1, 1);
            pass.set_pipeline(renderer.pipeline.stage(ShaderStageId::SampleDirectLighting));
            pass.dispatch_workgroups(dimensions.workgroups, 1, 1);
            pass.set_pipeline(renderer.pipeline.stage(ShaderStageId::IntersectShadow));
            pass.dispatch_workgroups(dimensions.workgroups, 1, 1);
            pass.set_pipeline(renderer.pipeline.stage(ShaderStageId::UpdateFilm));
            pass.dispatch_workgroups(dimensions.workgroups, 1, 1);
        }
        renderer.device_context.queue.submit(Some(encoder.finish()));
    }
    let mut readback_encoder =
        renderer
            .device_context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pbrt-r4 WebGPU wavefront readback"),
            });
    readback_encoder.copy_buffer_to_buffer(
        &buffers.output,
        0,
        &buffers.readback,
        0,
        dimensions.output_size,
    );
    renderer
        .device_context
        .queue
        .submit(Some(readback_encoder.finish()));
    let rgb = readback_output(renderer, &buffers.readback, dimensions.pixel_count)?;
    let inverse_sample_count = 1.0 / request.sample_count as f32;
    let rgb: Vec<[f32; 3]> = rgb
        .into_iter()
        .map(|pixel| pixel.map(|component| component * inverse_sample_count))
        .collect();
    GpuRenderOutput::new(dimensions.pixel_bounds, rgb.into_boxed_slice(), request)
        .map_err(|error| BackendError::Readback(format!("invalid wavefront output: {error:?}")))
}

fn arena_header_bytes(sample_index: u32, capacity: u32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&sample_index.to_ne_bytes());
    bytes[4..8].copy_from_slice(&capacity.to_ne_bytes());
    bytes
}

fn readback_output(
    renderer: &Renderer,
    readback_buffer: &wgpu::Buffer,
    pixel_count: usize,
) -> Result<Vec<[f32; 3]>, BackendError> {
    let mapped = readback_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    mapped.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result.map_err(|error| error.to_string()));
    });
    renderer
        .device_context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| BackendError::Readback(error.to_string()))?;
    receiver
        .recv()
        .map_err(|error| BackendError::Readback(error.to_string()))?
        .map_err(BackendError::Readback)?;
    let bytes = mapped
        .get_mapped_range()
        .map_err(|error| BackendError::Readback(error.to_string()))?;
    let pixels = bytemuck::try_cast_slice::<u8, [f32; 4]>(&bytes)
        .map_err(|error| BackendError::Readback(error.to_string()))?;
    let rgb = pixels
        .iter()
        .take(pixel_count)
        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect();
    drop(bytes);
    readback_buffer.unmap();
    Ok(rgb)
}
