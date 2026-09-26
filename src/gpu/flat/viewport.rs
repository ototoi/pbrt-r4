#[derive(Clone, Debug, PartialEq)]
pub struct Viewport {
    /// Full output image resolution. Fixed for the render; the camera and
    /// sampler are keyed to this regardless of `region_offset`/
    /// `region_resolution`.
    pub resolution: [u32; 2],
    /// Offset and extent of the rendered region ("cropwindow"/
    /// "pixelbounds") within `resolution`. Equal to `[0, 0]`/`resolution`
    /// when rendering the whole image.
    pub region_offset: [u32; 2],
    pub region_resolution: [u32; 2],
}
