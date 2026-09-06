#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub kind: String,
    pub data: MaterialData,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MaterialData {
    Diffuse(DiffuseMaterialData),
    Dielectric(DielectricMaterialData),
    Unsupported,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiffuseMaterialData {
    pub reflectance: [f32; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct DielectricMaterialData {
    pub eta: f32,
}
