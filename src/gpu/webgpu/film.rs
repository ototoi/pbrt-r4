use std::sync::{Arc, RwLock};

use crate::displays::{Display, DisplayTile, MultipleDisplay};
use crate::util::base::{Float, Point2i};
use crate::util::error::PbrtError;
use crate::util::geometry::Bounds2i;
use crate::util::imageio::write_image;

use super::output::Output;

pub struct Film {
    pub resolution: [u32; 2],
    pub framebuffer: wgpu::Buffer,
    readback: wgpu::Buffer,
    display: MultipleDisplay,
    pixels: Vec<f32>,
    completed_samples: u32,
    output_matrix: [[f32; 3]; 3],
    scale: f32,
    display_mode: bool,
}

impl Film {
    pub fn new(
        device: &wgpu::Device,
        resolution: [u32; 2],
        output_matrix: [[f32; 3]; 3],
        scale: f32,
        display_mode: bool,
    ) -> Result<Self, PbrtError> {
        let pixel_count = u64::from(resolution[0])
            .checked_mul(u64::from(resolution[1]))
            .ok_or_else(|| PbrtError::error("WebGPU film resolution overflowed."))?;
        let pixel_byte_size = pixel_count
            .checked_mul(4 * std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| PbrtError::error("WebGPU framebuffer size overflowed."))?;
        let framebuffer_size = pixel_byte_size;
        Ok(Self {
            resolution,
            framebuffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 framebuffer"),
                size: framebuffer_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 framebuffer readback"),
                size: pixel_byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            display: MultipleDisplay::new(),
            pixels: vec![0.0; pixel_count as usize * 3],
            completed_samples: 0,
            output_matrix,
            scale,
            display_mode,
        })
    }

    pub fn add_display(&mut self, display: &Arc<RwLock<dyn Display>>) {
        self.display.add_display(display);
    }

    pub fn has_no_display(&self) -> bool {
        self.display.is_empty()
    }

    pub fn start(&mut self) -> Result<(), PbrtError> {
        self.display.start(
            "pbrt-r4 WebGPU diffuse",
            &[self.resolution[0] as usize, self.resolution[1] as usize],
            &["R", "G", "B"],
        )
    }

    pub fn clear(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.completed_samples = 0;
        encoder.clear_buffer(&self.framebuffer, 0, None);
    }

    pub fn complete_sample(&mut self) -> Result<(), PbrtError> {
        self.completed_samples = self
            .completed_samples
            .checked_add(1)
            .ok_or_else(|| PbrtError::error("WebGPU Film completed sample count overflowed."))?;
        Ok(())
    }

    pub fn completed_samples(&self) -> u32 {
        self.completed_samples
    }

    pub fn copy_to_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.framebuffer,
            0,
            &self.readback,
            0,
            self.readback.size(),
        );
    }

    pub fn readback(&mut self, device: &wgpu::Device) -> Result<(), PbrtError> {
        let completed_samples = self.completed_samples;
        if completed_samples == 0 {
            return Err(PbrtError::error(
                "WebGPU Film cannot read back zero samples.",
            ));
        }
        let slice = self.readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| {
                PbrtError::error(&format!("WebGPU readback polling failed: {error}"))
            })?;
        receiver
            .recv()
            .map_err(|error| {
                PbrtError::error(&format!("WebGPU readback callback failed: {error}"))
            })?
            .map_err(|error| {
                PbrtError::error(&format!("WebGPU framebuffer mapping failed: {error}"))
            })?;
        let mapped = slice.get_mapped_range().map_err(|error| {
            PbrtError::error(&format!("WebGPU framebuffer map access failed: {error}"))
        })?;
        let values = bytemuck::try_cast_slice::<u8, f32>(&mapped)
            .map_err(|_| PbrtError::error("WebGPU framebuffer readback was not f32-aligned."))?;
        for (source, destination) in values.chunks_exact(4).zip(self.pixels.chunks_exact_mut(3)) {
            let weight = source[3].max(1e-7);
            let sensor = [source[0] / weight, source[1] / weight, source[2] / weight];
            for channel in 0..3 {
                destination[channel] = if self.display_mode {
                    sensor[channel]
                } else {
                    self.scale
                        * (self.output_matrix[channel][0] * sensor[0]
                            + self.output_matrix[channel][1] * sensor[1]
                            + self.output_matrix[channel][2] * sensor[2])
                };
            }
        }
        drop(mapped);
        self.readback.unmap();
        Ok(())
    }

    pub fn update_display(&mut self) -> Result<(), PbrtError> {
        self.display.update(&DisplayTile {
            x: 0,
            y: 0,
            width: self.resolution[0] as usize,
            height: self.resolution[1] as usize,
            buffer: self.pixels.clone(),
        })
    }

    pub fn end(&mut self) -> Result<(), PbrtError> {
        self.display.end()
    }

    pub fn write_output(&self, output: &Output) -> Result<(), PbrtError> {
        let bounds = Bounds2i::from((
            (0, 0),
            (self.resolution[0] as i32, self.resolution[1] as i32),
        ));
        let resolution = Point2i::new(self.resolution[0] as i32, self.resolution[1] as i32);
        let pixels: Vec<Float> = self.pixels.iter().map(|value| *value as Float).collect();
        write_image(&output.filename, &pixels, &bounds, &resolution)
    }
}
