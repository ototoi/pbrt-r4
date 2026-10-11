use super::Transform;

#[derive(Clone, Debug, PartialEq)]
pub struct Medium {
    pub name: String,
    pub kind: String,
    pub sigma_a: u32,
    pub sigma_s: u32,
    pub le: u32,
    pub g: f32,
    pub transform: Transform,
    pub data: MediumData,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MediumData {
    Homogeneous,
    UniformGrid(UniformGridMedium),
    Cloud(CloudMedium),
    RgbGrid(RgbGridMedium),
}

#[derive(Clone, Debug, PartialEq)]
pub struct UniformGridMedium {
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub resolution: [u32; 3],
    pub density: Vec<f32>,
    pub majorant_resolution: [u32; 3],
    pub majorant: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CloudMedium {
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub density: f32,
    pub wispiness: f32,
    pub frequency: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RgbGridMedium {
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub resolution: [u32; 3],
    pub sigma_a: Option<Vec<[f32; 3]>>,
    pub sigma_s: Option<Vec<[f32; 3]>>,
    pub scale: f32,
    pub color_space: super::texture::ColorSpace,
    pub majorant_resolution: [u32; 3],
    pub majorant: Vec<f32>,
}
