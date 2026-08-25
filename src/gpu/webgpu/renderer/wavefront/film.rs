use super::super::super::error::BackendError;
use super::super::super::shader::ShaderStageId;
use super::super::Renderer;
use super::{dispatch_control_stage, dispatch_stage};

pub fn finish_bounce<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    workgroups: u32,
) {
    dispatch_stage(renderer, pass, ShaderStageId::FinishBounce, workgroups);
}

pub fn prepare_next_bounce<'a>(renderer: &'a Renderer, pass: &mut wgpu::ComputePass<'a>) {
    dispatch_control_stage(renderer, pass, ShaderStageId::PrepareNextBounce);
}

pub fn update_film<'a>(renderer: &'a Renderer, pass: &mut wgpu::ComputePass<'a>, workgroups: u32) {
    dispatch_stage(renderer, pass, ShaderStageId::UpdateFilm, workgroups);
}

pub fn advance_sample<'a>(renderer: &'a Renderer, pass: &mut wgpu::ComputePass<'a>) {
    dispatch_control_stage(renderer, pass, ShaderStageId::AdvanceSample);
}

pub fn readback_film(
    renderer: &Renderer,
    readback_buffer: &wgpu::Buffer,
    control_readback_buffer: &wgpu::Buffer,
    pixel_count: usize,
) -> Result<Vec<[f32; 3]>, BackendError> {
    let mapped = readback_buffer.slice(..);
    let control_mapped = control_readback_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::sync_channel(2);
    let output_sender = sender.clone();
    mapped.map_async(wgpu::MapMode::Read, move |result| {
        let _ = output_sender.send((0u8, result.map_err(|error| error.to_string())));
    });
    control_mapped.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send((1u8, result.map_err(|error| error.to_string())));
    });
    renderer
        .device_context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| BackendError::Readback(error.to_string()))?;
    for _ in 0..2 {
        let (_, result) = receiver
            .recv()
            .map_err(|error| BackendError::Readback(error.to_string()))?;
        result.map_err(BackendError::Readback)?;
    }
    let control_bytes = control_mapped
        .get_mapped_range()
        .map_err(|error| BackendError::Readback(error.to_string()))?;
    let control = bytemuck::try_cast_slice::<u8, u32>(&control_bytes)
        .map_err(|error| BackendError::Readback(error.to_string()))?;
    let overflow = control.get(3).copied().unwrap_or_default() != 0;
    drop(control_bytes);
    control_readback_buffer.unmap();
    if overflow {
        readback_buffer.unmap();
        return Err(BackendError::Readback(
            "WebGPU wavefront queue overflow".to_string(),
        ));
    }
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
