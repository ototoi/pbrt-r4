use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BSSRDFTableRecord {
    pub rho_offset: u32,
    pub radius_offset: u32,
    pub profile_offset: u32,
    pub rho_eff_offset: u32,
    pub cdf_offset: u32,
    pub rho_count: u32,
    pub radius_count: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BSSRDFProbeWorkItem {
    pub position: [f32; 4],
    pub normal: [f32; 4],
    pub sigma_t: [f32; 4],
    pub rho: [f32; 4],
    pub sample: [f32; 4],
    pub table_index: u32,
    pub material_root: u32,
    pub pixel_index: u32,
    pub ray_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BSSRDFProbeResult {
    pub start: [f32; 4],
    pub end: [f32; 4],
    pub position: [f32; 4],
    pub position_error: [f32; 4],
    pub normal: [f32; 4],
    pub barycentric: [f32; 4],
    pub instance_index: u32,
    pub primitive_index: u32,
    pub candidate_count: u32,
    pub valid: u32,
    pub reservoir_probability: f32,
    pub segment_valid: u32,
    pub padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BSSRDFMaterialRecord {
    pub scale: f32,
    pub eta: f32,
    pub table_index: u32,
    pub coefficient_kind: u32,
}
