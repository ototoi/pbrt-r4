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
}

impl Film {
    pub fn new(device: &wgpu::Device, resolution: [u32; 2]) -> Result<Self, PbrtError> {
        let pixel_count = u64::from(resolution[0])
            .checked_mul(u64::from(resolution[1]))
            .ok_or_else(|| PbrtError::error("WebGPU film resolution overflowed."))?;
        let byte_size = pixel_count
            .checked_mul(4 * std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| PbrtError::error("WebGPU framebuffer size overflowed."))?;
        Ok(Self {
            resolution,
            framebuffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 framebuffer"),
                size: byte_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 framebuffer readback"),
                size: byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            display: MultipleDisplay::new(),
            pixels: vec![0.0; pixel_count as usize * 3],
        })
    }

    pub fn add_display(&mut self, display: &Arc<RwLock<dyn Display>>) {
        self.display.add_display(display);
    }

    pub fn start(&mut self) -> Result<(), PbrtError> {
        self.display.start(
            "pbrt-r4 WebGPU primary-ray normal",
            &[self.resolution[0] as usize, self.resolution[1] as usize],
            &["R", "G", "B"],
        )
    }

    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.framebuffer, 0, None);
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
            destination.copy_from_slice(&source[..3]);
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
