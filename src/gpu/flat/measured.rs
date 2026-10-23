//! GPU-neutral packed representation of pbrt-v4 measured BSDF tables.

pub const MEASURED_ATLAS_WIDTH: u32 = 2048;
pub const MEASURED_ATLAS_HEIGHT: u32 = 2048;
pub const INVALID_MEASURED_OFFSET: u32 = u32::MAX;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeasuredBsdfResources {
    pub bsdfs: Vec<MeasuredBsdfRecord>,
    pub tables: Vec<MeasuredTableRecord>,
    pub atlas_pages: Vec<MeasuredAtlasPage>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredBsdfRecord {
    pub ndf: u32,
    pub sigma: u32,
    pub vndf: u32,
    pub luminance: u32,
    pub spectra: u32,
    pub isotropic: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredTableRecord {
    pub size: [u32; 2],
    pub parameter_count: u32,
    pub parameter_sizes: [u32; 3],
    pub parameter_strides: [u32; 3],
    pub parameter_value_offsets: [u32; 3],
    pub data_offset: u32,
    pub marginal_cdf_offset: u32,
    pub conditional_cdf_offset: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredAtlasPage {
    pub resolution: [u32; 2],
    /// RGBA texels in row-major order.
    pub texels: Vec<f32>,
}
