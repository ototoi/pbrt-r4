//! Mapping values stored in executable Flat IR texture instructions.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvMapping {
    pub uscale: f32,
    pub vscale: f32,
    pub udelta: f32,
    pub vdelta: f32,
}

/// Transform variants store a row-major 4×4 mapping matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TextureMapping {
    Uv(UvMapping),
    Planar([f32; 16]),
    Spherical([f32; 16]),
    Cylindrical([f32; 16]),
    PointTransform([f32; 16]),
}
