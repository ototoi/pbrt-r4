//! Fixed-slot arena layout for the WebGPU wavefront path.

pub const ARENA_HEADER_SIZE: u32 = 16;
pub const RAY_WORK_ITEM_STRIDE: u32 = 192;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArenaLayout {
    pub capacity: u32,
    pub byte_len: u32,
}

impl ArenaLayout {
    pub fn for_pixel_count(pixel_count: u32) -> Option<Self> {
        let byte_len =
            ARENA_HEADER_SIZE.checked_add(RAY_WORK_ITEM_STRIDE.checked_mul(pixel_count)?)?;
        Some(Self {
            capacity: pixel_count,
            byte_len,
        })
    }
}
