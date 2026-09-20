//! Immutable WebGPU resources shared by procedural Noise texture operations.

use crate::textures::noise::noise_permutation;

pub struct NoiseRuntimeResources {
    _texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl NoiseRuntimeResources {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        const WIDTH: u32 = 256;
        const HEIGHT: u32 = 257;
        let mut table = vec![0u8; (WIDTH * HEIGHT) as usize];
        for x in 0..WIDTH as usize {
            table[x] = noise_permutation(x);
        }
        for y in 0..WIDTH as usize {
            for x in 0..WIDTH as usize {
                let first = usize::from(noise_permutation(x));
                table[(y + 1) * WIDTH as usize + x] = noise_permutation(first + y);
            }
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pbrt-r4 Noise permutation table"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &table,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH),
                rows_per_image: Some(HEIGHT),
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
        }
    }
}
