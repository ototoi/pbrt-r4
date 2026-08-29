use crate::util::atomic_double::AtomicDouble;
use crate::util::base::Float;
use std::sync::atomic::Ordering;

pub type AtomicRgb = [AtomicDouble; 3];

pub fn new_atomic_rgb() -> AtomicRgb {
    std::array::from_fn(|_| AtomicDouble::default())
}

pub fn new_atomic_rgb_buffer(pixel_count: usize) -> Vec<AtomicRgb> {
    (0..pixel_count).map(|_| new_atomic_rgb()).collect()
}

pub fn clear_atomic_rgb_buffer(pixels: &[AtomicRgb]) {
    for pixel in pixels {
        for channel in pixel {
            channel.store(0.0, Ordering::Relaxed);
        }
    }
}

pub fn load_atomic_rgb(pixel: &AtomicRgb) -> [Float; 3] {
    std::array::from_fn(|channel| pixel[channel].load(Ordering::Relaxed) as Float)
}

pub fn add_atomic_rgb(pixel: &AtomicRgb, value: [Float; 3]) {
    for channel in 0..3 {
        pixel[channel].fetch_add(value[channel] as f64, Ordering::Relaxed);
    }
}
