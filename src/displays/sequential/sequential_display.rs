use crate::displays::*;

use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::imageio::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::RwLock;

pub struct SequentialDisplay {
    count: AtomicU64,
    resolution: [usize; 2],
    buffer: RwLock<Vec<f32>>,
    output_dir: PathBuf,
    channel_names: Vec<String>,
}

impl SequentialDisplay {
    pub fn new(output_dir: &str) -> Self {
        let output_dir = Path::new(output_dir).to_path_buf();
        let channel_names: Vec<String> = ["R", "G", "B"].iter().map(|s| s.to_string()).collect();
        // method body elided
        SequentialDisplay {
            count: AtomicU64::new(0),
            resolution: [640, 480],
            buffer: RwLock::new(Vec::new()),
            output_dir,
            channel_names,
            //prev_call_time: std::time::Instant::now(),
        }
    }

    pub fn update_buffer(&mut self, tile: &DisplayTile) -> Result<(), PbrtError> {
        let x0 = tile.x;
        let x1 = tile.x + tile.width;
        let y0 = tile.y;
        let y1 = tile.y + tile.height;
        let channels = self.channel_names.len();
        let mut buffer = self.buffer.write().unwrap();
        for y in y0..y1 {
            for x in x0..x1 {
                let i_offset = ((y - y0) * tile.width + (x - x0)) * channels;
                let o_offset = (y * self.resolution[0] + x) * channels;
                for i in 0..channels {
                    buffer[o_offset + i] = tile.buffer[i_offset + i];
                }
            }
        }
        Ok(())
    }
}

impl Display for SequentialDisplay {
    fn start(
        &mut self,
        _title: &str,
        resolution: &[usize; 2],
        channel_names: &[&str],
    ) -> Result<(), PbrtError> {
        let n_channels = channel_names.len();
        let n_pixels = resolution[0] * resolution[1];
        let buffer_size = n_channels * n_pixels;
        {
            let mut buffer = self.buffer.write().unwrap();
            buffer.resize(buffer_size, 0.0);
        }
        self.resolution = *resolution;
        self.count.store(0, std::sync::atomic::Ordering::Relaxed);
        self.channel_names = channel_names.iter().map(|s| s.to_string()).collect();
        Ok(())
    }

    fn update(&mut self, tile: &DisplayTile) -> Result<(), PbrtError> {
        self.update_buffer(tile)?;
        let count = self
            .count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let filename = format!("{:08}.exr", count);
        let filepath = self.output_dir.join(&filename);
        let mut pixels = Vec::new();
        {
            let buffer = self.buffer.read().unwrap();
            for i in 0..buffer.len() {
                pixels.push(buffer[i] as Float);
            }
        }
        let bounds = Bounds2i::from((
            (0, 0),
            (self.resolution[0] as i32, self.resolution[1] as i32),
        ));
        let resolution = Vector2i::from((self.resolution[0] as i32, self.resolution[1] as i32));
        write_image(filepath.to_str().unwrap(), &pixels, &bounds, &resolution)?;
        Ok(())
    }

    fn end(&mut self) -> Result<(), PbrtError> {
        // method body elided
        println!("SequentialDisplay::end()");
        Ok(())
    }
}
