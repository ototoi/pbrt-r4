use crate::util::error::PbrtError;

const HEADER_BYTES: u64 = 16;
const DEFAULT_CONTINUATION_ROUNDS: u32 = 2;
pub(super) const MAX_CONTINUATION_ROUNDS: u32 = 8;

/// Samples left out of the film because their continuations outlasted the
/// GPU rounds; the host renders them again with the waiting loop.
pub(super) struct SampleRetry {
    buffer: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl SampleRetry {
    pub(super) fn new(device: &wgpu::Device, samples_per_pixel: u32) -> Self {
        let size = HEADER_BYTES + u64::from(samples_per_pixel.max(1)) * 4;
        Self {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 sample retry"),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 sample retry readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    pub(super) fn binding_resource(&self) -> wgpu::BindingResource<'_> {
        self.buffer.as_entire_binding()
    }

    pub(super) fn clear(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&[0u32; 2]));
    }

    pub(super) fn clear_pending(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&0u32));
    }

    pub(super) fn read(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u32>, PbrtError> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pbrt-r4 sample retry readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &self.readback, 0, self.buffer.size());
        queue.submit(Some(encoder.finish()));
        let slice = self.readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| {
                PbrtError::error(&format!("WebGPU sample retry polling failed: {error}"))
            })?;
        receiver
            .recv()
            .map_err(|error| {
                PbrtError::error(&format!("WebGPU sample retry callback failed: {error}"))
            })?
            .map_err(|error| {
                PbrtError::error(&format!("WebGPU sample retry mapping failed: {error}"))
            })?;
        let mapped = slice.get_mapped_range().map_err(|error| {
            PbrtError::error(&format!("WebGPU sample retry map access failed: {error}"))
        })?;
        let words = bytemuck::try_cast_slice::<u8, u32>(&mapped)
            .map_err(|_| PbrtError::error("WebGPU sample retry readback was not u32-aligned."))?;
        let capacity = words.len() - (HEADER_BYTES / 4) as usize;
        let count = (words[1] as usize).min(capacity);
        let samples = words[4..4 + count].to_vec();
        drop(mapped);
        self.readback.unmap();
        Ok(samples)
    }
}

/// `PBRT_R4_GPU_CONTINUATION_ROUNDS` sets the initial rounds; 0 keeps every
/// continuation on the waiting loop.
pub(super) fn continuation_rounds_from_environment() -> Result<u32, PbrtError> {
    match std::env::var("PBRT_R4_GPU_CONTINUATION_ROUNDS") {
        Ok(value) => value
            .parse::<u32>()
            .ok()
            .filter(|rounds| *rounds <= MAX_CONTINUATION_ROUNDS)
            .ok_or_else(|| {
                PbrtError::error(&format!(
                    "PBRT_R4_GPU_CONTINUATION_ROUNDS must be an integer from 0 to {MAX_CONTINUATION_ROUNDS}."
                ))
            }),
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_CONTINUATION_ROUNDS),
        Err(std::env::VarError::NotUnicode(_)) => Err(PbrtError::error(
            "PBRT_R4_GPU_CONTINUATION_ROUNDS must be valid UTF-8.",
        )),
    }
}
