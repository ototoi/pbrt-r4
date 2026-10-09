use std::time::Duration;

pub(super) const DEFAULT_DISPLAY_UPDATE_INTERVAL: Duration = Duration::from_millis(500);

/// Default side length of a square wavefront tile. Tiles are processed
/// strictly sequentially (never in parallel); the goal is bounding the size
/// of every per-pixel wavefront buffer, not speed. See
/// `docs/webgpu-tile-rendering-design_ja.md` in the devkit repo.
pub(super) const DEFAULT_GPU_TILE_SIZE: u32 = 512;

/// One rectangular sub-region of the rendered region, in full-image
/// coordinates.
#[derive(Clone, Copy)]
pub(super) struct Tile {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

/// Number of tiles along each axis, without materializing them.
pub(super) fn tile_grid_dims(
    region_width: u32,
    region_height: u32,
    tile_width: u32,
    tile_height: u32,
) -> (u32, u32) {
    (
        region_width.div_ceil(tile_width),
        region_height.div_ceil(tile_height),
    )
}

/// Iterates every tile in the region's grid lazily: a `--gpu-tile-size 1`
/// render of an 8K image would otherwise need a multi-gigabyte `Vec<Tile>`,
/// defeating the point of tiling to reduce memory use.
pub(super) fn compute_tiles(
    region_x: u32,
    region_y: u32,
    region_width: u32,
    region_height: u32,
    tile_width: u32,
    tile_height: u32,
) -> impl Iterator<Item = Tile> {
    let (tiles_x, tiles_y) = tile_grid_dims(region_width, region_height, tile_width, tile_height);
    (0..tiles_y).flat_map(move |tile_y| {
        (0..tiles_x).map(move |tile_x| {
            let x = tile_x * tile_width;
            let y = tile_y * tile_height;
            Tile {
                x: region_x + x,
                y: region_y + y,
                width: tile_width.min(region_width - x),
                height: tile_height.min(region_height - y),
            }
        })
    })
}
