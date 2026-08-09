use crate::util::base::*;
use crate::util::geometry::*;

pub const ST_W: usize = 16;
pub const ST_SZ: usize = ST_W * ST_W;

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct SplatTile {
    pub dirty: bool,
    pub pixel_bounds: Bounds2i,
    pub pixels: [[Float; 3]; ST_SZ],
}

impl SplatTile {
    pub fn new(pixel_bounds: &Bounds2i) -> Self {
        SplatTile {
            dirty: true,
            pixel_bounds: *pixel_bounds,
            pixels: [[0.0; 3]; ST_SZ],
        }
    }
}

impl Default for SplatTile {
    fn default() -> Self {
        SplatTile {
            dirty: false,
            pixel_bounds: Bounds2i::default(),
            pixels: [[0.0; 3]; ST_SZ],
        }
    }
}
