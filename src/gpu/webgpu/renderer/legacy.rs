use super::super::super::ir::{GpuRenderOutput, GpuRenderRequest};
use super::super::error::BackendError;
use super::resources::{RenderBuffers, RenderDimensions};
use super::Renderer;

pub fn render(
    renderer: &Renderer,
    request: GpuRenderRequest,
    dimensions: &RenderDimensions,
    buffers: &RenderBuffers,
    bind_group: &wgpu::BindGroup,
) -> Result<GpuRenderOutput, BackendError> {
    let mut encoder =
        renderer
            .device_context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pbrt-r4 WebGPU dispatch"),
            });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pbrt-r4 WebGPU render pass"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_pipeline(renderer.pipeline.stage(super::ShaderStageId::LegacyRender));
        pass.dispatch_workgroups(
            dimensions.legacy_workgroups[0],
            dimensions.legacy_workgroups[1],
            1,
        );
    }
    encoder.copy_buffer_to_buffer(
        &buffers.output,
        0,
        &buffers.readback,
        0,
        dimensions.output_size,
    );
    renderer.device_context.queue.submit(Some(encoder.finish()));
    let rgb = readback_output(renderer, &buffers.readback, dimensions.pixel_count)?;
    GpuRenderOutput::new(dimensions.pixel_bounds, rgb.into_boxed_slice(), request)
        .map_err(|error| BackendError::Readback(format!("invalid render output: {error:?}")))
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
