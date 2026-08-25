//! Host-side layout for the initial fixed-slot GPU wavefront arena.

pub const WAVEFRONT_SLOT_STRIDE: u32 = 160;
pub const WAVEFRONT_CONTROL_SIZE: u32 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WavefrontLayout {
    pub capacity: u32,
    pub byte_len: u32,
}

impl WavefrontLayout {
    pub fn for_pixel_count(pixel_count: u32) -> Option<Self> {
        let byte_len = WAVEFRONT_SLOT_STRIDE
            .checked_mul(pixel_count)?
            .checked_add(WAVEFRONT_CONTROL_SIZE)?;
        Some(Self {
            capacity: pixel_count,
            byte_len,
        })
    }

    pub fn slot_offset(self, index: u32) -> Option<u32> {
        (index < self.capacity).then(|| WAVEFRONT_CONTROL_SIZE + index * WAVEFRONT_SLOT_STRIDE)
    }
}
